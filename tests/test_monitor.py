from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from astrbot_plugin_mercari_agent.domain import (
    AgentDecision,
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    ListingRunState,
    WatchRule,
)
from astrbot_plugin_mercari_agent.graph import build_listing_graph
from astrbot_plugin_mercari_agent.monitor import (
    CrawlService,
    HttpStatusError,
    MockCollector,
    Monitor,
    ParseError,
    RetryPolicy,
)
from astrbot_plugin_mercari_agent.storage import (
    CrawlAttemptRow,
    CrawlJobRow,
    ListingProcessRunRow,
    Repository,
)
from sqlalchemy import func, select, update


class SequenceCollector:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    async def collect(self, rule: WatchRule) -> list[Listing]:
        self.calls += 1
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response  # type: ignore[return-value]


class RecordingGraph:
    def __init__(self, fails: bool = False) -> None:
        self.fails = fails
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
        self.calls.append(state)
        if self.fails:
            raise RuntimeError("graph unavailable")
        return state


class BlockingCollector:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def collect(self, rule: WatchRule) -> list[Listing]:
        self.started.set()
        await self.release.wait()
        return []


class BlockingGraph:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
        self.started.set()
        await self.release.wait()
        return state


class EmptyRetriever:
    def retrieve(self, query: str) -> list[object]:
        return []


class ThrowingEvaluator:
    async def evaluate(self, listing: Listing, evidence: list[object]) -> object:
        raise RuntimeError("Authorization: Bearer private-token")


class PassingEvaluator:
    async def evaluate(self, listing: Listing, evidence: list[object]) -> AgentDecision:
        return AgentDecision(
            score=88,
            recommendation="HIGH_PRIORITY",
            reasons=("wanted",),
            risks=(),
            retrieved_evidence=(),
            model_name="test-evaluator",
            prompt_version="test-v1",
        )


class SuccessfulNotifier:
    async def send(self, target_session: str, text: str) -> bool:
        return True


class FailingPollService:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    def active_rule_ids(self) -> tuple[str, ...]:
        return ("rule-1",)

    async def resume_active_jobs(
        self,
        rules_by_id: dict[str, WatchRule],
        *,
        rule_id: str | None = None,
    ) -> None:
        self.started.set()
        raise RuntimeError("poller failed")

    async def run_scheduled(self, rule: WatchRule) -> int:
        return 0


@pytest.fixture
def repository(tmp_path):
    repo = Repository.open(tmp_path / "mercari.sqlite3")
    yield repo
    repo.dispose()


@pytest.fixture
def rule() -> WatchRule:
    return WatchRule(
        id="rule-1",
        name="手毬 badge",
        include_keywords=("缶バッジ",),
        interval_seconds=60,
        target_session="aiocqhttp:group:123",
    )


@pytest.fixture
def listing() -> Listing:
    return Listing(
        marketplace="mercari",
        external_id="m-001",
        title="月村手毬 缶バッジ 未開封",
        price_jpy=1200,
        url="https://example.invalid/items/m-001",
        discovered_at=datetime.now(timezone.utc),
    )


def _service(
    repository: Repository,
    collector: SequenceCollector | MockCollector,
    graph: RecordingGraph | None = None,
    sleeps: list[float] | None = None,
) -> CrawlService:
    async def sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return CrawlService(
        repository,
        collector,
        graph or RecordingGraph(),
        sleep=sleep,
        jitter=lambda: 0.0,
    )


def test_timeout_then_success_creates_two_attempts_and_succeeds(
    repository: Repository, rule: WatchRule, listing: Listing
) -> None:
    service = _service(
        repository,
        SequenceCollector([TimeoutError("temporary"), [listing]]),
    )

    job_id = asyncio.run(service.run_once(rule))

    assert [item.state for item in repository.attempts(job_id)] == [
        CrawlAttemptState.FAILED,
        CrawlAttemptState.SUCCEEDED,
    ]
    assert repository.latest_job().state is CrawlJobState.SUCCEEDED


def test_three_retryable_failures_exhaust_the_job(
    repository: Repository, rule: WatchRule
) -> None:
    service = _service(
        repository,
        SequenceCollector([TimeoutError("one"), TimeoutError("two"), TimeoutError("three")]),
    )

    job_id = asyncio.run(service.run_once(rule))

    assert [item.state for item in repository.attempts(job_id)] == [
        CrawlAttemptState.FAILED,
        CrawlAttemptState.FAILED,
        CrawlAttemptState.FAILED,
    ]
    assert repository.latest_job().state is CrawlJobState.EXHAUSTED


def test_http_403_fails_once_without_retry(
    repository: Repository, rule: WatchRule
) -> None:
    service = _service(repository, SequenceCollector([HttpStatusError(403)]))

    job_id = asyncio.run(service.run_once(rule))

    attempts = repository.attempts(job_id)
    assert len(attempts) == 1
    assert attempts[0].state is CrawlAttemptState.FAILED
    assert attempts[0].error_type == "http_403"
    assert repository.latest_job().state is CrawlJobState.EXHAUSTED


def test_zero_jitter_uses_five_then_twenty_second_backoff(
    repository: Repository, rule: WatchRule
) -> None:
    sleeps: list[float] = []
    service = _service(
        repository,
        SequenceCollector([TimeoutError("one"), TimeoutError("two"), TimeoutError("three")]),
        sleeps=sleeps,
    )

    asyncio.run(service.run_once(rule))

    assert sleeps == [5.0, 20.0]


def test_mock_collector_returns_the_stable_offline_listing(rule: WatchRule) -> None:
    listings = asyncio.run(MockCollector().collect(rule))

    assert len(listings) == 1
    listing = listings[0]
    assert (
        listing.marketplace,
        listing.external_id,
        listing.title,
        listing.price_jpy,
        listing.url,
    ) == (
        "mercari",
        "mock-mercari-001",
        "月村手毬 缶バッジ 未開封",
        1200,
        "https://example.invalid/mock-mercari-001",
    )

    assert listings == asyncio.run(MockCollector().collect(rule))


def test_successful_collection_invokes_graph_once_per_listing(
    repository: Repository, rule: WatchRule, listing: Listing
) -> None:
    second = listing.model_copy(update={"external_id": "m-002"})
    graph = RecordingGraph()
    service = _service(repository, SequenceCollector([[listing, second]]), graph)

    asyncio.run(service.run_once(rule))

    assert [call["listing"] for call in graph.calls] == [listing, second]
    assert [call["watch_rule"] for call in graph.calls] == [rule, rule]


def test_graph_failure_does_not_rewind_successful_crawl(
    repository: Repository, rule: WatchRule, listing: Listing
) -> None:
    service = _service(
        repository,
        SequenceCollector([[listing]]),
        RecordingGraph(fails=True),
    )

    job_id = asyncio.run(service.run_once(rule))

    assert repository.latest_job().state is CrawlJobState.SUCCEEDED
    assert repository.attempts(job_id)[0].state is CrawlAttemptState.SUCCEEDED


def test_job_is_succeeded_before_downstream_graph_can_be_cancelled(
    repository: Repository, rule: WatchRule, listing: Listing
) -> None:
    graph = BlockingGraph()
    service = _service(repository, SequenceCollector([[listing]]), graph)  # type: ignore[arg-type]

    async def exercise() -> None:
        task = asyncio.create_task(service.run_once(rule))
        await graph.started.wait()
        try:
            assert repository.latest_job().state is CrawlJobState.SUCCEEDED
            assert repository.attempts(repository.latest_job().id)[0].state is CrawlAttemptState.SUCCEEDED
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())


def test_compiled_graph_evaluator_failure_fails_process_run_without_rewinding_crawl(
    repository: Repository, rule: WatchRule, listing: Listing
) -> None:
    graph = build_listing_graph(
        repository,
        EmptyRetriever(),
        ThrowingEvaluator(),
        SuccessfulNotifier(),
    )
    service = _service(repository, SequenceCollector([[listing]]), graph)  # type: ignore[arg-type]

    job_id = asyncio.run(service.run_once(rule))

    with repository._sessions() as session:
        process_run = session.scalar(select(ListingProcessRunRow))
    assert process_run is not None
    assert ListingRunState(process_run.state) is ListingRunState.FAILED
    assert process_run.error_summary == "sensitive error detail redacted"
    assert repository.latest_job().state is CrawlJobState.SUCCEEDED
    assert repository.attempts(job_id)[0].state is CrawlAttemptState.SUCCEEDED


def test_queue_failure_fails_process_run_without_rewinding_successful_crawl(
    repository: Repository, rule: WatchRule, listing: Listing, monkeypatch
) -> None:
    graph = build_listing_graph(
        repository,
        EmptyRetriever(),
        PassingEvaluator(),
        SuccessfulNotifier(),
    )

    def fail_queue(*args: object, **kwargs: object) -> tuple[int, bool]:
        raise RuntimeError("Authorization: Bearer private-token")

    monkeypatch.setattr(repository, "queue_notification_for_run", fail_queue)
    service = _service(repository, SequenceCollector([[listing]]), graph)  # type: ignore[arg-type]

    job_id = asyncio.run(service.run_once(rule))

    with repository._sessions() as session:
        process_run = session.scalar(select(ListingProcessRunRow))
    assert process_run is not None
    assert ListingRunState(process_run.state) is ListingRunState.FAILED
    assert process_run.error_summary == "sensitive error detail redacted"
    assert repository.latest_job().state is CrawlJobState.SUCCEEDED
    assert repository.attempts(job_id)[0].state is CrawlAttemptState.SUCCEEDED


def test_hard_filter_transition_failure_fails_process_run_without_rewinding_crawl(
    repository: Repository, rule: WatchRule, listing: Listing, monkeypatch
) -> None:
    original_advance = repository.advance_listing_run

    def fail_rule_transition(
        run_id: int,
        target: ListingRunState,
        error_summary: object = None,
    ) -> None:
        if target is ListingRunState.RULE_EVALUATED:
            raise RuntimeError("Cookie: private-value")
        original_advance(run_id, target, error_summary)

    monkeypatch.setattr(repository, "advance_listing_run", fail_rule_transition)
    graph = build_listing_graph(
        repository,
        EmptyRetriever(),
        PassingEvaluator(),
        SuccessfulNotifier(),
    )
    service = _service(repository, SequenceCollector([[listing]]), graph)  # type: ignore[arg-type]

    job_id = asyncio.run(service.run_once(rule))

    with repository._sessions() as session:
        process_run = session.scalar(select(ListingProcessRunRow))
    assert process_run is not None
    assert ListingRunState(process_run.state) is ListingRunState.FAILED
    assert process_run.error_summary == "sensitive error detail redacted"
    assert repository.latest_job().state is CrawlJobState.SUCCEEDED
    assert repository.attempts(job_id)[0].state is CrawlAttemptState.SUCCEEDED


def test_deduplicate_transition_failure_fails_process_run_without_rewinding_crawl(
    repository: Repository, rule: WatchRule, listing: Listing, monkeypatch
) -> None:
    original_advance = repository.advance_listing_run

    def fail_normalize_transition(
        run_id: int,
        target: ListingRunState,
        error_summary: object = None,
    ) -> None:
        if target is ListingRunState.NORMALIZED:
            raise RuntimeError("Cookie: private-value")
        original_advance(run_id, target, error_summary)

    monkeypatch.setattr(repository, "advance_listing_run", fail_normalize_transition)
    graph = build_listing_graph(
        repository,
        EmptyRetriever(),
        PassingEvaluator(),
        SuccessfulNotifier(),
    )
    service = _service(repository, SequenceCollector([[listing]]), graph)  # type: ignore[arg-type]

    job_id = asyncio.run(service.run_once(rule))

    with repository._sessions() as session:
        process_run = session.scalar(select(ListingProcessRunRow))
    assert process_run is not None
    assert ListingRunState(process_run.state) is ListingRunState.FAILED
    assert process_run.error_summary == "sensitive error detail redacted"
    assert repository.latest_job().state is CrawlJobState.SUCCEEDED
    assert repository.attempts(job_id)[0].state is CrawlAttemptState.SUCCEEDED


def test_recovery_continues_due_active_job_with_a_new_attempt(
    repository: Repository, rule: WatchRule, listing: Listing
) -> None:
    job_id = repository.create_job(rule.id)
    repository.activate_job(job_id)
    first_attempt = repository.create_attempt(job_id)
    repository.advance_attempt(first_attempt, CrawlAttemptState.REQUESTING)
    repository.advance_attempt(
        first_attempt,
        CrawlAttemptState.FAILED,
        error_type="timeout",
        next_retry_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        finished_at=datetime.now(timezone.utc),
    )
    service = _service(repository, SequenceCollector([[listing]]))

    asyncio.run(service.resume_active_jobs({rule.id: rule}))

    assert [item.state for item in repository.attempts(job_id)] == [
        CrawlAttemptState.FAILED,
        CrawlAttemptState.SUCCEEDED,
    ]
    assert repository.latest_job().state is CrawlJobState.SUCCEEDED


def test_recovery_cancels_active_job_with_no_matching_rule(repository: Repository) -> None:
    job_id = repository.create_job("missing")
    repository.activate_job(job_id)
    service = _service(repository, SequenceCollector([]))

    asyncio.run(service.resume_active_jobs({}))

    assert repository.latest_job().id == job_id
    assert repository.latest_job().state is CrawlJobState.CANCELLED


def test_monitor_lifecycle_is_idempotent_and_leaves_no_running_task(
    repository: Repository, rule: WatchRule
) -> None:
    monitor = Monitor(
        _service(repository, SequenceCollector([])),
        [rule],
        poll_interval=60,
    )

    async def exercise() -> None:
        await monitor.start()
        first_task = monitor.task
        await monitor.start()
        assert monitor.task is first_task
        assert monitor.running is True
        await monitor.stop()
        await monitor.stop()

    asyncio.run(exercise())

    assert monitor.running is False
    assert monitor.task is None


def test_monitor_stop_clears_task_when_the_poller_already_failed(
    rule: WatchRule,
) -> None:
    service = FailingPollService()
    monitor = Monitor(service, [rule], poll_interval=60)  # type: ignore[arg-type]

    async def exercise() -> None:
        await monitor.start()
        await service.started.wait()
        with pytest.raises(RuntimeError, match="poller failed"):
            await monitor.stop()

    asyncio.run(exercise())

    assert monitor.running is False
    assert monitor.task is None


def test_cancelling_a_crawl_terminalizes_attempt_and_recovery_retries_it(
    repository: Repository, rule: WatchRule, listing: Listing
) -> None:
    collector = BlockingCollector()
    service = _service(repository, collector)

    async def exercise() -> None:
        task = asyncio.create_task(service.run_once(rule))
        await collector.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    job_id = repository.latest_job().id
    first_attempt = repository.attempts(job_id)[0]
    assert repository.latest_job().state is CrawlJobState.ACTIVE
    assert first_attempt.state is CrawlAttemptState.FAILED
    assert first_attempt.error_type == "cancelled"
    assert first_attempt.next_retry_at <= datetime.now(timezone.utc)

    recovery = _service(repository, SequenceCollector([[listing]]))
    asyncio.run(recovery.resume_active_jobs({rule.id: rule}))

    assert [attempt.state for attempt in repository.attempts(job_id)] == [
        CrawlAttemptState.FAILED,
        CrawlAttemptState.SUCCEEDED,
    ]
    assert repository.latest_job().state is CrawlJobState.SUCCEEDED


@pytest.mark.parametrize(
    ("error", "retryable", "error_type"),
    [
        (TimeoutError(), True, "timeout"),
        (ConnectionError(), True, "connection_error"),
        (HttpStatusError(429), True, "http_429"),
        (HttpStatusError(503), True, "http_5xx"),
        (ParseError(), True, "parse_error"),
        (HttpStatusError(401), False, "http_401"),
        (RuntimeError(), False, "unexpected_error"),
    ],
)
def test_retry_policy_classifies_deterministic_failures(
    error: BaseException, retryable: bool, error_type: str
) -> None:
    decision = RetryPolicy().classify(error)
    assert (decision.retryable, decision.error_type) == (retryable, error_type)


@pytest.mark.parametrize(
    "stale_state",
    [
        CrawlAttemptState.CREATED,
        CrawlAttemptState.REQUESTING,
        CrawlAttemptState.RECEIVED,
        CrawlAttemptState.PARSING,
        CrawlAttemptState.SAVING,
    ],
)
def test_restart_recovery_fails_each_stale_nonterminal_attempt_then_retries(
    repository: Repository,
    rule: WatchRule,
    listing: Listing,
    stale_state: CrawlAttemptState,
) -> None:
    job_id = repository.create_job(rule.id)
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)
    for stage in (
        CrawlAttemptState.REQUESTING,
        CrawlAttemptState.RECEIVED,
        CrawlAttemptState.PARSING,
        CrawlAttemptState.SAVING,
    ):
        if stale_state is CrawlAttemptState.CREATED:
            break
        repository.advance_attempt(attempt_id, stage)
        if stage is stale_state:
            break
    stale_at = datetime.now(timezone.utc) - timedelta(hours=1)
    with repository._engine.begin() as connection:
        connection.execute(
            update(CrawlAttemptRow)
            .where(CrawlAttemptRow.id == attempt_id)
            .values(updated_at=stale_at)
        )
    collector = SequenceCollector([[listing]])
    service = CrawlService(
        repository,
        collector,
        RecordingGraph(),
        max_attempts=3,
        attempt_stale_after=timedelta(seconds=30),
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda: 0.0,
    )

    asyncio.run(service.resume_active_jobs({rule.id: rule}))

    attempts = repository.attempts(job_id)
    assert [attempt.state for attempt in attempts] == [
        CrawlAttemptState.FAILED,
        CrawlAttemptState.SUCCEEDED,
    ]
    assert attempts[0].error_type == "crash_recovered"
    assert attempts[0].error_summary == "stale attempt recovered after restart"
    assert collector.calls == 1
    assert repository.get_job(job_id).state is CrawlJobState.SUCCEEDED


def test_restart_recovery_finishes_active_job_after_succeeded_attempt(
    repository: Repository,
    rule: WatchRule,
) -> None:
    job_id = repository.create_job(rule.id)
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)
    for stage in (
        CrawlAttemptState.REQUESTING,
        CrawlAttemptState.RECEIVED,
        CrawlAttemptState.PARSING,
        CrawlAttemptState.SAVING,
        CrawlAttemptState.SUCCEEDED,
    ):
        repository.advance_attempt(attempt_id, stage)
    collector = SequenceCollector([])
    service = _service(repository, collector)

    asyncio.run(service.resume_active_jobs({rule.id: rule}))

    assert repository.get_job(job_id).state is CrawlJobState.SUCCEEDED
    assert len(repository.attempts(job_id)) == 1
    assert collector.calls == 0


def test_restart_recovery_does_not_steal_a_fresh_nonterminal_attempt(
    repository: Repository,
    rule: WatchRule,
) -> None:
    job_id = repository.create_job(rule.id)
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)
    repository.advance_attempt(attempt_id, CrawlAttemptState.REQUESTING)
    collector = SequenceCollector([])
    service = CrawlService(
        repository,
        collector,
        RecordingGraph(),
        attempt_stale_after=timedelta(hours=1),
    )

    asyncio.run(service.resume_active_jobs({rule.id: rule}))

    assert repository.get_job(job_id).state is CrawlJobState.ACTIVE
    assert repository.get_attempt(attempt_id).state is CrawlAttemptState.REQUESTING
    assert collector.calls == 0


def test_poll_cycle_preserves_future_backoff_without_creating_another_job(
    repository: Repository,
    rule: WatchRule,
) -> None:
    job_id = repository.create_job(rule.id)
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)
    repository.advance_attempt(attempt_id, CrawlAttemptState.REQUESTING)
    repository.advance_attempt(
        attempt_id,
        CrawlAttemptState.FAILED,
        error_type="timeout",
        error_summary="temporary",
        next_retry_at=datetime.now(timezone.utc) + timedelta(hours=1),
        finished_at=datetime.now(timezone.utc),
    )
    service = _service(repository, SequenceCollector([]))
    monitor = Monitor(service, [rule], poll_interval=60)

    async def exercise() -> None:
        await monitor.start()
        for _ in range(20):
            await asyncio.sleep(0)
        await monitor.stop()

    asyncio.run(exercise())

    with repository._sessions() as session:
        job_count = session.scalar(select(func.count()).select_from(CrawlJobRow))
        attempt_count = session.scalar(
            select(func.count()).select_from(CrawlAttemptRow)
        )
    assert job_count == 1
    assert attempt_count == 1
    assert repository.get_job(job_id).state is CrawlJobState.ACTIVE
