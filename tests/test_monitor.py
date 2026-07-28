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


class CountingNotifier:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, target_session: str, text: str) -> bool:
        self.calls += 1
        return True


class SimulatedProcessCrash(BaseException):
    pass


class CrashBeforeGraphWork:
    async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
        raise SimulatedProcessCrash("process stopped before graph work")


class CrashDuringRetrieval:
    def retrieve(self, query: str) -> list[object]:
        raise SimulatedProcessCrash("process stopped during retrieval")


class CrashDuringEvaluation:
    async def evaluate(
        self,
        listing: Listing,
        evidence: list[object],
    ) -> object:
        raise SimulatedProcessCrash("process stopped during evaluation")


class FailingPollService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.continued = asyncio.Event()

    def active_rule_ids(self) -> tuple[str, ...]:
        return ("rule-1",)

    async def resume_active_jobs(
        self,
        rules_by_id: dict[str, WatchRule],
        *,
        rule_id: str | None = None,
    ) -> None:
        self.started.set()
        raise RuntimeError("Authorization: private-token")

    async def run_scheduled(self, rule: WatchRule) -> int:
        self.continued.set()
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
        decision_version="test-v1",
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
    def fail_normalize_claim(run_id: int) -> bool:
        raise RuntimeError("Cookie: private-value")

    monkeypatch.setattr(
        repository,
        "claim_discovered_listing_run",
        fail_normalize_claim,
    )
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


def test_monitor_records_startup_recovery_error_and_continues_polling(
    rule: WatchRule,
) -> None:
    service = FailingPollService()
    monitor = Monitor(
        service,
        [rule],
        poll_interval=60,
        error_delay=0.001,
    )  # type: ignore[arg-type]

    async def exercise() -> None:
        await monitor.start()
        await asyncio.wait_for(service.started.wait(), timeout=1)
        await asyncio.wait_for(service.continued.wait(), timeout=1)
        assert monitor.running is True
        await monitor.stop()

    asyncio.run(exercise())

    assert monitor.running is False
    assert monitor.task is None
    assert monitor.last_poll_error == "sensitive error detail redacted"


def test_monitor_retries_maintenance_and_exposes_only_sanitized_error(
    rule: WatchRule,
) -> None:
    class IdleService:
        def active_rule_ids(self) -> tuple[str, ...]:
            return ()

        async def resume_active_jobs(
            self,
            rules_by_id: dict[str, WatchRule],
            *,
            rule_id: str | None = None,
        ) -> None:
            return None

        async def run_scheduled(self, rule: WatchRule) -> int:
            return 0

    maintenance_calls = 0
    second_cycle = asyncio.Event()

    async def maintenance() -> list[Exception]:
        nonlocal maintenance_calls
        maintenance_calls += 1
        if maintenance_calls == 1:
            return [RuntimeError("Authorization: private-token")]
        second_cycle.set()
        return []

    monitor = Monitor(
        IdleService(),
        [rule.model_copy(update={"enabled": False})],
        poll_interval=60,
        error_delay=0.001,
        maintenance=maintenance,
    )  # type: ignore[arg-type]

    async def exercise() -> None:
        await monitor.start()
        await asyncio.wait_for(second_cycle.wait(), timeout=1)
        assert monitor.running is True
        await monitor.stop()

    asyncio.run(exercise())

    assert maintenance_calls == 2
    assert (
        monitor.last_maintenance_error
        == "sensitive error detail redacted"
    )
    assert monitor.running is False


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


def _drain_reopened_listing_work(
    database,
    run_id: int,
) -> tuple[Repository, CountingNotifier]:
    reopened = Repository.open(database)
    notifier = CountingNotifier()
    graph = build_listing_graph(
        reopened,
        EmptyRetriever(),
        PassingEvaluator(),
        notifier,
    )
    service = CrawlService(
        reopened,
        SequenceCollector([]),
        graph,
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda: 0.0,
    )

    errors = asyncio.run(service.drain_pending_listing_work())

    assert errors == []
    assert reopened.get_listing_run(run_id).state is ListingRunState.NOTIFIED
    assert reopened.count_notifications() == 1
    assert notifier.calls == 1
    return reopened, notifier


def _recover_reopened_inflight_work(
    database,
    old_run_id: int,
) -> tuple[Repository, CountingNotifier]:
    reopened = Repository.open(database)
    notifier = CountingNotifier()
    collector = SequenceCollector([])
    graph = build_listing_graph(
        reopened,
        EmptyRetriever(),
        PassingEvaluator(),
        notifier,
    )
    service = CrawlService(
        reopened,
        collector,
        graph,
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda: 0.0,
        listing_run_stale_after=timedelta(seconds=30),
    )

    errors = asyncio.run(service.drain_pending_listing_work())

    old = reopened.get_listing_run(old_run_id)
    with reopened._sessions() as session:
        runs = list(
            session.scalars(
                select(ListingProcessRunRow).order_by(
                    ListingProcessRunRow.run_no
                )
            )
        )
    assert errors == []
    assert collector.calls == 0
    assert old.state is ListingRunState.FAILED
    assert old.error_summary == (
        "crash_recovered: stale in-flight listing work"
    )
    assert len(runs) == 2
    assert runs[1].run_no == runs[0].run_no + 1
    assert ListingRunState(runs[1].state) is ListingRunState.NOTIFIED
    assert notifier.calls == 1
    return reopened, notifier


@pytest.mark.parametrize(
    ("crash_point", "abandoned_state"),
    [
        ("after_claim", ListingRunState.NORMALIZED),
        ("retrieval", ListingRunState.RULE_EVALUATED),
        ("evaluation", ListingRunState.RAG_RETRIEVED),
    ],
)
def test_reopen_recovers_stale_inflight_work_without_recollection(
    tmp_path,
    rule: WatchRule,
    listing: Listing,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
    abandoned_state: ListingRunState,
) -> None:
    database = tmp_path / f"crash-{crash_point}.sqlite3"
    repository = Repository.open(database)
    retriever: object = EmptyRetriever()
    evaluator: object = PassingEvaluator()
    expected_error: type[BaseException] = SimulatedProcessCrash

    if crash_point == "after_claim":
        original_advance = repository.advance_listing_run

        def crash_after_claim(
            run_id: int,
            target: ListingRunState,
            error_summary: object = None,
        ) -> None:
            if target is ListingRunState.DEDUP_CHECKED:
                raise SimulatedProcessCrash("process stopped after claim")
            original_advance(run_id, target, error_summary)

        monkeypatch.setattr(
            repository,
            "advance_listing_run",
            crash_after_claim,
        )
    elif crash_point == "retrieval":
        retriever = CrashDuringRetrieval()
    else:
        evaluator = CrashDuringEvaluation()

    graph = build_listing_graph(
        repository,
        retriever,
        evaluator,
        SuccessfulNotifier(),
    )
    service = CrawlService(
        repository,
        SequenceCollector([[listing]]),
        graph,
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda: 0.0,
    )

    with pytest.raises(expected_error):
        asyncio.run(service.run_once(rule))

    with repository._sessions() as session:
        abandoned = session.scalar(select(ListingProcessRunRow))
    assert abandoned is not None
    assert ListingRunState(abandoned.state) is abandoned_state
    stale_at = datetime.now(timezone.utc) - timedelta(hours=1)
    with repository._engine.begin() as connection:
        connection.execute(
            update(ListingProcessRunRow)
            .where(ListingProcessRunRow.id == abandoned.id)
            .values(updated_at=stale_at)
        )
    repository.dispose()

    reopened, _ = _recover_reopened_inflight_work(
        database,
        abandoned.id,
    )
    reopened.dispose()


def test_crash_after_durable_run_creation_leaves_recoverable_listing_work(
    tmp_path,
    rule: WatchRule,
    listing: Listing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "crash-after-durable-run.sqlite3"
    repository = Repository.open(database)
    original_advance = repository.advance_attempt

    def crash_before_attempt_success(
        attempt_id: int,
        target: CrawlAttemptState,
        **fields: object,
    ) -> None:
        if target is CrawlAttemptState.SUCCEEDED:
            if len(repository.pending_listing_work()) != 1:
                raise SimulatedProcessCrash(
                    "durable run missing before attempt success"
                )
            raise SimulatedProcessCrash("after durable run commit")
        original_advance(attempt_id, target, **fields)

    monkeypatch.setattr(repository, "advance_attempt", crash_before_attempt_success)
    service = _service(repository, SequenceCollector([[listing]]))

    with pytest.raises(SimulatedProcessCrash, match="durable run"):
        asyncio.run(service.run_once(rule))

    run = repository.pending_listing_work()[0]
    assert repository.get_attempt(run.origin_attempt_id).state is CrawlAttemptState.SAVING
    assert repository.get_job(run.origin_job_id).state is CrawlJobState.ACTIVE
    repository.dispose()

    reopened, _ = _drain_reopened_listing_work(database, run.id)
    reopened.dispose()


def test_crash_after_attempt_success_keeps_listing_work_for_restart(
    tmp_path,
    rule: WatchRule,
    listing: Listing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "crash-after-attempt-success.sqlite3"
    repository = Repository.open(database)
    original_finish = repository.finish_job

    def crash_before_job_success(
        job_id: int,
        target: CrawlJobState,
    ) -> None:
        if target is CrawlJobState.SUCCEEDED:
            raise SimulatedProcessCrash("after attempt success")
        original_finish(job_id, target)

    monkeypatch.setattr(repository, "finish_job", crash_before_job_success)
    service = _service(repository, SequenceCollector([[listing]]))

    with pytest.raises(SimulatedProcessCrash, match="attempt success"):
        asyncio.run(service.run_once(rule))

    run = repository.pending_listing_work()[0]
    assert repository.get_attempt(run.origin_attempt_id).state is CrawlAttemptState.SUCCEEDED
    assert repository.get_job(run.origin_job_id).state is CrawlJobState.ACTIVE
    repository.dispose()

    reopened = Repository.open(database)
    recovery = _service(reopened, SequenceCollector([]))
    asyncio.run(recovery.resume_active_jobs({rule.id: rule}))
    assert reopened.get_job(run.origin_job_id).state is CrawlJobState.SUCCEEDED
    reopened.dispose()

    drained, _ = _drain_reopened_listing_work(database, run.id)
    drained.dispose()


def test_crash_after_job_success_keeps_listing_work_for_restart(
    tmp_path,
    rule: WatchRule,
    listing: Listing,
) -> None:
    database = tmp_path / "crash-after-job-success.sqlite3"
    repository = Repository.open(database)
    service = CrawlService(
        repository,
        SequenceCollector([[listing]]),
        CrashBeforeGraphWork(),
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda: 0.0,
    )

    with pytest.raises(SimulatedProcessCrash, match="before graph"):
        asyncio.run(service.run_once(rule))

    run = repository.pending_listing_work()[0]
    assert repository.get_attempt(run.origin_attempt_id).state is CrawlAttemptState.SUCCEEDED
    assert repository.get_job(run.origin_job_id).state is CrawlJobState.SUCCEEDED
    repository.dispose()

    reopened, _ = _drain_reopened_listing_work(database, run.id)
    reopened.dispose()


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


def test_one_rule_failure_does_not_block_later_rule_or_next_cycle() -> None:
    first = WatchRule(
        id="first",
        name="first",
        interval_seconds=1,
        target_session="session:first",
    )
    second = first.model_copy(
        update={"id": "second", "name": "second", "target_session": "session:second"}
    )

    class PerRuleFailureService:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.second_ran = asyncio.Event()
            self.first_recovered = asyncio.Event()

        def active_rule_ids(self) -> tuple[str, ...]:
            return ()

        async def resume_active_jobs(
            self,
            rules_by_id: dict[str, WatchRule],
            *,
            rule_id: str | None = None,
        ) -> None:
            return None

        async def run_scheduled(self, rule: WatchRule) -> int:
            self.calls.append(rule.id)
            if rule.id == "first" and self.calls.count("first") == 1:
                raise RuntimeError("Cookie: private-value")
            if rule.id == "second":
                self.second_ran.set()
            if rule.id == "first":
                self.first_recovered.set()
            return len(self.calls)

    service = PerRuleFailureService()
    monitor = Monitor(
        service,
        [first, second],
        poll_interval=0.01,
        error_delay=0.001,
    )  # type: ignore[arg-type]

    async def exercise() -> None:
        await monitor.start()
        await asyncio.wait_for(service.second_ran.wait(), timeout=1)
        await asyncio.wait_for(service.first_recovered.wait(), timeout=1)
        await monitor.stop()

    asyncio.run(exercise())

    assert service.calls[:3] == ["first", "second", "first"]
    assert monitor.last_poll_error == "sensitive error detail redacted"


def test_one_startup_recovery_failure_does_not_block_another_rule() -> None:
    first = WatchRule(
        id="first",
        name="first",
        interval_seconds=1,
        target_session="session:first",
    )
    second = first.model_copy(
        update={"id": "second", "name": "second", "target_session": "session:second"}
    )

    class PerRuleRecoveryFailureService:
        def __init__(self) -> None:
            self.recovery_calls: list[str] = []
            self.second_recovered = asyncio.Event()

        def active_rule_ids(self) -> tuple[str, ...]:
            return ("first", "second")

        async def resume_active_jobs(
            self,
            rules_by_id: dict[str, WatchRule],
            *,
            rule_id: str | None = None,
        ) -> None:
            assert rule_id is not None
            self.recovery_calls.append(rule_id)
            if rule_id == "first":
                raise RuntimeError("Authorization: private-token")
            self.second_recovered.set()

        async def run_scheduled(self, rule: WatchRule) -> int:
            return 0

    service = PerRuleRecoveryFailureService()
    monitor = Monitor(
        service,
        [first, second],
        poll_interval=60,
        error_delay=0.001,
    )  # type: ignore[arg-type]

    async def exercise() -> None:
        await monitor.start()
        await asyncio.wait_for(service.second_recovered.wait(), timeout=1)
        await monitor.stop()

    asyncio.run(exercise())

    assert service.recovery_calls == ["first", "second"]
    assert monitor.last_poll_error == "sensitive error detail redacted"
