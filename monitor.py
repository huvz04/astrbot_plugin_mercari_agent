"""Deterministic local crawl, retry, and recovery orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .domain import CrawlAttemptState, CrawlJobState, Listing, WatchRule
from .storage import CrawlAttempt, Repository


class Collector(Protocol):
    """Supplies listings without deciding retry or persistence behavior."""

    async def collect(self, rule: WatchRule) -> list[Listing]: ...


class ListingGraph(Protocol):
    """The compiled LangGraph interface used by successful saved listings."""

    async def ainvoke(self, state: dict[str, object]) -> object: ...


class HttpStatusError(Exception):
    """A collector failure carrying an HTTP status without retaining a body."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class ParseError(Exception):
    """The collector response could not be interpreted as listings."""


class InvalidRule(Exception):
    """A watch rule cannot be used by the collector."""


@dataclass(frozen=True)
class RetryDecision:
    retryable: bool
    error_type: str


class RetryPolicy:
    """Classifies only known transient collector failures as retryable."""

    _RETRYABLE_TYPES = frozenset(
        {"timeout", "connection_error", "http_429", "http_5xx", "parse_error"}
    )

    def classify(self, error: BaseException) -> RetryDecision:
        if isinstance(error, TimeoutError):
            return RetryDecision(True, "timeout")
        if isinstance(error, ConnectionError):
            return RetryDecision(True, "connection_error")
        if isinstance(error, ParseError):
            return RetryDecision(True, "parse_error")
        if isinstance(error, InvalidRule):
            return RetryDecision(False, "invalid_rule")
        if isinstance(error, HttpStatusError):
            if error.status_code == 429:
                return RetryDecision(True, "http_429")
            if 500 <= error.status_code <= 599:
                return RetryDecision(True, "http_5xx")
            return RetryDecision(False, f"http_{error.status_code}")
        return RetryDecision(False, "unexpected_error")

    def is_retryable_error_type(self, error_type: str | None) -> bool:
        return error_type in self._RETRYABLE_TYPES


class MockCollector:
    """A stable, offline collector used by the skeleton and its tests."""

    async def collect(self, rule: WatchRule) -> list[Listing]:
        return [
            Listing(
                marketplace="mercari",
                external_id="mock-mercari-001",
                title="月村手毬 缶バッジ 未開封",
                price_jpy=1200,
                url="https://example.invalid/mock-mercari-001",
                discovered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        ]


class CrawlService:
    """Persists explicit crawl transitions and retries transient failures."""

    _BACKOFF_SECONDS = (5.0, 20.0, 60.0)

    def __init__(
        self,
        repository: Repository,
        collector: Collector,
        graph: ListingGraph,
        *,
        retry_policy: RetryPolicy | None = None,
        max_attempts: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = lambda: 0.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self._repository = repository
        self._collector = collector
        self._graph = graph
        self._retry_policy = retry_policy or RetryPolicy()
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter

    async def run_once(self, rule: WatchRule) -> int:
        """Create, activate, and run one durable job for *rule*."""
        job_id = self._repository.create_job(rule.id)
        self._repository.activate_job(job_id)
        await self._run_new_attempts(job_id, rule)
        return job_id

    async def resume_active_jobs(
        self, rules_by_id: Mapping[str, WatchRule]
    ) -> None:
        """Continue only due, retryable active work after a process restart."""
        now = datetime.now(timezone.utc)
        for job in self._repository.active_jobs():
            rule = rules_by_id.get(job.rule_id)
            if rule is None:
                self._repository.finish_job(job.id, CrawlJobState.CANCELLED)
                continue

            attempts = self._repository.attempts(job.id)
            if self._has_unfinished_attempt(attempts):
                continue
            if not attempts:
                await self._run_new_attempts(job.id, rule)
                continue

            latest = attempts[-1]
            if latest.state is not CrawlAttemptState.FAILED:
                continue
            if latest.attempt_no >= self._max_attempts:
                self._repository.finish_job(job.id, CrawlJobState.EXHAUSTED)
                continue
            if not self._retry_policy.is_retryable_error_type(latest.error_type):
                self._repository.finish_job(job.id, CrawlJobState.EXHAUSTED)
                continue
            if latest.next_retry_at is None or latest.next_retry_at > now:
                continue
            await self._run_new_attempts(job.id, rule)

    async def _run_new_attempts(self, job_id: int, rule: WatchRule) -> None:
        """Drive fresh attempts until the job reaches a terminal state."""
        while True:
            attempt_id = self._repository.create_attempt(job_id)
            attempt = self._repository.get_attempt(attempt_id)
            succeeded, decision, retry_delay = await self._run_attempt(attempt, rule)
            if succeeded:
                self._repository.finish_job(job_id, CrawlJobState.SUCCEEDED)
                return
            if not decision.retryable or attempt.attempt_no >= self._max_attempts:
                self._repository.finish_job(job_id, CrawlJobState.EXHAUSTED)
                return

            assert retry_delay is not None
            await self._sleep(retry_delay)

    async def _run_attempt(
        self, attempt: CrawlAttempt, rule: WatchRule
    ) -> tuple[bool, RetryDecision, float | None]:
        try:
            self._repository.advance_attempt(
                attempt.id,
                CrawlAttemptState.REQUESTING,
                started_at=datetime.now(timezone.utc),
            )
            listings = await self._collector.collect(rule)
            self._repository.advance_attempt(attempt.id, CrawlAttemptState.RECEIVED)
            self._repository.advance_attempt(attempt.id, CrawlAttemptState.PARSING)
            if not isinstance(listings, list) or not all(
                isinstance(listing, Listing) for listing in listings
            ):
                raise ParseError("collector must return a list of Listing values")
            self._repository.advance_attempt(attempt.id, CrawlAttemptState.SAVING)
            for listing in listings:
                self._repository.save_listing(listing)
            self._repository.advance_attempt(
                attempt.id,
                CrawlAttemptState.SUCCEEDED,
                item_count=len(listings),
                finished_at=datetime.now(timezone.utc),
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            decision = self._retry_policy.classify(error)
            retry_delay = (
                self._backoff_for(attempt.attempt_no)
                if decision.retryable and attempt.attempt_no < self._max_attempts
                else None
            )
            self._repository.advance_attempt(
                attempt.id,
                CrawlAttemptState.FAILED,
                http_status=(
                    error.status_code
                    if isinstance(error, HttpStatusError)
                    else None
                ),
                error_type=decision.error_type,
                error_summary=str(error),
                next_retry_at=(
                    datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
                    if retry_delay is not None
                    else None
                ),
                finished_at=datetime.now(timezone.utc),
            )
            return False, decision, retry_delay

        for listing in listings:
            try:
                await self._graph.ainvoke(
                    {"listing": listing, "watch_rule": rule}
                )
            except Exception:
                # A listing process run owns evaluator/graph failures.  The
                # crawl is already safely persisted and must remain successful.
                pass
        return True, RetryDecision(False, ""), None

    def _backoff_for(self, attempt_no: int) -> float:
        base = self._BACKOFF_SECONDS[min(attempt_no - 1, len(self._BACKOFF_SECONDS) - 1)]
        return base + self._jitter()

    @staticmethod
    def _has_unfinished_attempt(attempts: list[CrawlAttempt]) -> bool:
        return any(
            attempt.state
            not in {CrawlAttemptState.SUCCEEDED, CrawlAttemptState.FAILED}
            for attempt in attempts
        )


class Monitor:
    """One poller with a separate lock for every watch rule."""

    def __init__(
        self,
        crawl_service: CrawlService,
        rules: Iterable[WatchRule],
        *,
        poll_interval: float,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._crawl_service = crawl_service
        self._rules = tuple(rules)
        self._poll_interval = poll_interval
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._rule_locks: dict[str, asyncio.Lock] = {}

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    async def start(self) -> None:
        if self.running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._poll(), name="mercari-monitor")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run_rule_now(self, rule: WatchRule) -> int:
        lock = self._rule_locks.setdefault(rule.id, asyncio.Lock())
        async with lock:
            return await self._crawl_service.run_once(rule)

    def status_text(self) -> str:
        state = "running" if self.running else "stopped"
        return f"Mercari monitor: {state}; rules={len(self._rules)}"

    async def _poll(self) -> None:
        rules_by_id = {rule.id: rule for rule in self._rules}
        while not self._stop_event.is_set():
            await self._crawl_service.resume_active_jobs(rules_by_id)
            for rule in self._rules:
                if rule.enabled:
                    await self.run_rule_now(rule)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval
                )
            except TimeoutError:
                continue
