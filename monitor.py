"""Deterministic local crawl, retry, and recovery orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .domain import (
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    ListingRunState,
    WatchRule,
)
from .storage import CrawlAttempt, Repository, sanitize_error_summary


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
        {
            "timeout",
            "connection_error",
            "http_429",
            "http_5xx",
            "parse_error",
            "cancelled",
            "crash_recovered",
        }
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
        attempt_stale_after: timedelta = timedelta(minutes=5),
        listing_run_stale_after: timedelta = timedelta(minutes=5),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if attempt_stale_after <= timedelta(0):
            raise ValueError("attempt_stale_after must be positive")
        if listing_run_stale_after <= timedelta(0):
            raise ValueError("listing_run_stale_after must be positive")
        self._repository = repository
        self._collector = collector
        self._graph = graph
        self._retry_policy = retry_policy or RetryPolicy()
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter
        self._attempt_stale_after = attempt_stale_after
        self._listing_run_stale_after = listing_run_stale_after

    async def run_once(self, rule: WatchRule) -> int:
        """Create, activate, and run one durable job for *rule*."""
        job_id = self._repository.create_job(rule.id)
        self._repository.activate_job(job_id)
        await self._run_new_attempts(job_id, rule)
        return job_id

    async def resume_active_jobs(
        self,
        rules_by_id: Mapping[str, WatchRule],
        *,
        rule_id: str | None = None,
    ) -> None:
        """Continue only due, retryable active work after a process restart."""
        now = datetime.now(timezone.utc)
        for job in self._repository.active_jobs(rule_id):
            rule = rules_by_id.get(job.rule_id)
            if rule is None:
                self._repository.finish_job(job.id, CrawlJobState.CANCELLED)
                continue

            attempts = self._repository.attempts(job.id)
            if not attempts:
                await self._run_new_attempts(job.id, rule)
                continue

            latest = attempts[-1]
            if latest.state is CrawlAttemptState.SUCCEEDED:
                self._repository.finish_job(job.id, CrawlJobState.SUCCEEDED)
                continue
            if latest.state not in {
                CrawlAttemptState.SUCCEEDED,
                CrawlAttemptState.FAILED,
            }:
                if latest.updated_at > now - self._attempt_stale_after:
                    continue
                self._repository.advance_attempt(
                    latest.id,
                    CrawlAttemptState.FAILED,
                    error_type="crash_recovered",
                    error_summary="stale attempt recovered after restart",
                    next_retry_at=now,
                    finished_at=now,
                )
                latest = self._repository.get_attempt(latest.id)
            if latest.state is not CrawlAttemptState.FAILED:
                continue
            if latest.attempt_no >= self._max_attempts:
                self._repository.finish_job(job.id, CrawlJobState.EXHAUSTED)
                continue
            if not self._retry_policy.is_retryable_error_type(latest.error_type):
                self._repository.finish_job(job.id, CrawlJobState.EXHAUSTED)
                continue
            if latest.next_retry_at is not None and latest.next_retry_at > now:
                continue
            await self._run_new_attempts(job.id, rule)

    async def run_scheduled(self, rule: WatchRule) -> int:
        """Recover a rule first and create work only when no ACTIVE Job remains."""
        await self.resume_active_jobs({rule.id: rule}, rule_id=rule.id)
        active = self._repository.active_job_for_rule(rule.id)
        if active is not None:
            return active.id
        return await self.run_once(rule)

    def active_rule_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(job.rule_id for job in self._repository.active_jobs()))

    async def _run_new_attempts(self, job_id: int, rule: WatchRule) -> None:
        """Drive fresh attempts until the job reaches a terminal state."""
        while True:
            attempt_id = self._repository.create_attempt(job_id)
            attempt = self._repository.get_attempt(attempt_id)
            succeeded, decision, retry_delay, run_ids = await self._run_attempt(
                attempt, rule
            )
            if succeeded:
                self._repository.finish_job(job_id, CrawlJobState.SUCCEEDED)
                await self._process_run_ids(run_ids)
                return
            if not decision.retryable or attempt.attempt_no >= self._max_attempts:
                self._repository.finish_job(job_id, CrawlJobState.EXHAUSTED)
                return

            assert retry_delay is not None
            await self._sleep(retry_delay)

    async def _run_attempt(
        self, attempt: CrawlAttempt, rule: WatchRule
    ) -> tuple[bool, RetryDecision, float | None, list[int]]:
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
            run_ids: list[int] = []
            for listing in listings:
                run_id, _ = self._repository.persist_listing_work(
                    listing,
                    rule,
                    origin_job_id=attempt.job_id,
                    origin_attempt_id=attempt.id,
                )
                run_ids.append(run_id)
            self._repository.advance_attempt(
                attempt.id,
                CrawlAttemptState.SUCCEEDED,
                item_count=len(listings),
                finished_at=datetime.now(timezone.utc),
            )
        except asyncio.CancelledError:
            cancelled_at = datetime.now(timezone.utc)
            self._repository.advance_attempt(
                attempt.id,
                CrawlAttemptState.FAILED,
                error_type="cancelled",
                error_summary="crawl cancelled",
                next_retry_at=cancelled_at,
                finished_at=cancelled_at,
            )
            raise
        except Exception as error:
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
            return False, decision, retry_delay, []

        return True, RetryDecision(False, ""), None, run_ids

    async def drain_pending_listing_work(self) -> list[Exception]:
        """Process safe durable work while isolating failures by run."""
        self._repository.recover_stale_listing_work(
            stale_after=self._listing_run_stale_after
        )
        return await self._process_run_ids(
            [run.id for run in self._repository.pending_listing_work()]
        )

    async def _process_run_ids(self, run_ids: list[int]) -> list[Exception]:
        errors: list[Exception] = []
        for run_id in run_ids:
            try:
                run = self._repository.get_listing_run(run_id)
                if run.state is not ListingRunState.DISCOVERED:
                    continue
                if run.rule_snapshot_json is None:
                    raise ValueError("listing work has no rule snapshot")
                rule = WatchRule.model_validate_json(run.rule_snapshot_json)
                listing = self._repository.get_listing(run.listing_id)
                result = await self._graph.ainvoke(
                    {
                        "listing": listing,
                        "watch_rule": rule,
                        "process_run_id": run.id,
                    }
                )
                if isinstance(result, Mapping):
                    errors.extend(
                        RuntimeError(str(error))
                        for error in result.get("errors", ())
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                errors.append(error)
                try:
                    current = self._repository.get_listing_run(run_id)
                    if current.state is ListingRunState.DISCOVERED:
                        self._repository.advance_listing_run(
                            run_id,
                            ListingRunState.FAILED,
                            error_summary=str(error),
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as cleanup_error:
                    errors.append(
                        RuntimeError(
                            sanitize_error_summary(str(cleanup_error))
                            or "listing work cleanup failed"
                        )
                    )
        return errors

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
        error_delay: float = 1.0,
        maintenance: (
            Callable[[], Awaitable[list[Exception]]] | None
        ) = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if error_delay <= 0:
            raise ValueError("error_delay must be positive")
        self._crawl_service = crawl_service
        self._rules = tuple(rules)
        self._poll_interval = poll_interval
        self._error_delay = min(error_delay, poll_interval, 5.0)
        self._maintenance = maintenance
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._rule_locks: dict[str, asyncio.Lock] = {}
        self._maintenance_lock = asyncio.Lock()
        self._last_poll_error: str | None = None
        self._last_maintenance_error: str | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def last_poll_error(self) -> str | None:
        return self._last_poll_error

    @property
    def last_maintenance_error(self) -> str | None:
        return self._last_maintenance_error

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
        finally:
            self._task = None

    async def run_rule_now(self, rule: WatchRule) -> int:
        lock = self._rule_locks.setdefault(rule.id, asyncio.Lock())
        async with lock:
            return await self._crawl_service.run_scheduled(rule)

    def status_text(self) -> str:
        state = "running" if self.running else "stopped"
        return f"Mercari monitor: {state}; rules={len(self._rules)}"

    async def run_maintenance(self) -> bool:
        """Run one isolated safe-work drain under the maintenance lock."""
        if self._maintenance is None:
            return False
        async with self._maintenance_lock:
            try:
                errors = await self._maintenance()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                errors = [error]
        for error in errors:
            self._last_maintenance_error = (
                sanitize_error_summary(str(error)) or "maintenance error"
            )
        return bool(errors)

    async def _poll(self) -> None:
        rules_by_id = {rule.id: rule for rule in self._rules}
        startup_failed = await self.run_maintenance()
        try:
            active_rule_ids = self._crawl_service.active_rule_ids()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            active_rule_ids = ()
            startup_failed = True
            self._record_poll_error(error)
        for rule_id in active_rule_ids:
            try:
                lock = self._rule_locks.setdefault(rule_id, asyncio.Lock())
                async with lock:
                    await self._crawl_service.resume_active_jobs(
                        rules_by_id,
                        rule_id=rule_id,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                startup_failed = True
                self._record_poll_error(error)
        if startup_failed:
            await self._wait(self._error_delay)
        while not self._stop_event.is_set():
            cycle_failed = await self.run_maintenance()
            for rule in self._rules:
                if rule.enabled:
                    try:
                        await self.run_rule_now(rule)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        cycle_failed = True
                        self._record_poll_error(error)
            await self._wait(
                self._error_delay if cycle_failed else self._poll_interval
            )

    def _record_poll_error(self, error: Exception) -> None:
        self._last_poll_error = (
            sanitize_error_summary(str(error)) or "poll error"
        )

    async def _wait(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except TimeoutError:
            return
