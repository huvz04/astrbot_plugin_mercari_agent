from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3
from threading import Barrier, Event

import pytest
from sqlalchemy import event, insert, text, update
from sqlalchemy.exc import IntegrityError

from astrbot_plugin_mercari_agent.domain import (
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    ListingRunState,
    NotificationState,
    TransitionError,
    WatchRule,
)
from astrbot_plugin_mercari_agent.storage import (
    ConcurrentStateChange,
    CrawlAttemptRow,
    ListingProcessRunRow,
    ListingRow,
    Repository,
)


@pytest.fixture
def repository(tmp_path):
    repo = Repository.open(tmp_path / "mercari.sqlite3")
    yield repo
    repo.dispose()


@pytest.fixture
def listing() -> Listing:
    return Listing(
        marketplace="mercari",
        external_id="m123",
        title="月村手毬 缶バッジ",
        price_jpy=1200,
        url="https://example.invalid/item/m123",
        discovered_at=datetime.now(timezone.utc),
    )


def _durable_rule(rule_id: str) -> WatchRule:
    return WatchRule(
        id=rule_id,
        name=rule_id,
        include_keywords=("月村手毬",),
        max_price_jpy=1500,
        interval_seconds=60,
        target_session=f"session:{rule_id}",
    )


def _saving_origin(
    repository: Repository,
    rule_id: str,
) -> tuple[int, int]:
    job_id = repository.create_job(rule_id)
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)
    for state in (
        CrawlAttemptState.REQUESTING,
        CrawlAttemptState.RECEIVED,
        CrawlAttemptState.PARSING,
        CrawlAttemptState.SAVING,
    ):
        repository.advance_attempt(attempt_id, state)
    return job_id, attempt_id


def _advance_listing_run_to(
    repository: Repository,
    run_id: int,
    target: ListingRunState,
) -> None:
    for state in (
        ListingRunState.NORMALIZED,
        ListingRunState.DEDUP_CHECKED,
        ListingRunState.RULE_EVALUATED,
        ListingRunState.RAG_RETRIEVED,
        ListingRunState.AGENT_EVALUATED,
        ListingRunState.NOTIFICATION_QUEUED,
    ):
        repository.advance_listing_run(run_id, state)
        if state is target:
            return
    raise AssertionError(f"unsupported test target {target}")


def test_retry_creates_a_new_attempt_without_rewriting_failure(repository) -> None:
    job_id = repository.create_job("rule-1")
    repository.activate_job(job_id)
    first = repository.create_attempt(job_id)
    repository.advance_attempt(first, CrawlAttemptState.REQUESTING)
    repository.advance_attempt(first, CrawlAttemptState.FAILED, error_type="timeout")

    second = repository.create_attempt(job_id)

    assert second != first
    assert repository.get_attempt(first).state is CrawlAttemptState.FAILED
    assert repository.get_attempt(second).attempt_no == 2


def test_listing_unique_key_is_durable(repository, listing) -> None:
    first_id, first_created = repository.save_listing(listing)
    second_id, second_created = repository.save_listing(listing)

    assert second_id == first_id
    assert first_created is True
    assert second_created is False


def test_persist_listing_work_atomically_records_rule_and_origin(
    repository: Repository,
    listing: Listing,
) -> None:
    rule = WatchRule(
        id="rule-durable",
        name="durable rule",
        include_keywords=("月村手毬",),
        exclude_keywords=("ジャンク",),
        max_price_jpy=1500,
        interval_seconds=60,
        target_session="aiocqhttp:group:123",
    )
    job_id = repository.create_job(rule.id)
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)
    for state in (
        CrawlAttemptState.REQUESTING,
        CrawlAttemptState.RECEIVED,
        CrawlAttemptState.PARSING,
        CrawlAttemptState.SAVING,
    ):
        repository.advance_attempt(attempt_id, state)

    run_id, created = repository.persist_listing_work(
        listing,
        rule,
        origin_job_id=job_id,
        origin_attempt_id=attempt_id,
    )

    run = repository.get_listing_run(run_id)
    stored_listing = repository.get_listing(run.listing_id)
    assert created is True
    assert stored_listing == listing
    assert run.state is ListingRunState.DISCOVERED
    assert run.rule_snapshot_json == rule.model_dump_json()
    assert WatchRule.model_validate_json(run.rule_snapshot_json) == rule
    assert run.origin_job_id == job_id
    assert run.origin_attempt_id == attempt_id
    assert [work.id for work in repository.pending_listing_work()] == [run_id]


def test_persist_listing_work_rejects_invalid_origin_without_saving_listing(
    repository: Repository,
    listing: Listing,
) -> None:
    rule = WatchRule(
        id="rule-invalid-origin",
        name="invalid origin",
        interval_seconds=60,
        target_session="session",
    )
    job_id = repository.create_job(rule.id)
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)

    with pytest.raises(ConcurrentStateChange, match="SAVING"):
        repository.persist_listing_work(
            listing,
            rule,
            origin_job_id=job_id,
            origin_attempt_id=attempt_id,
        )

    with repository._sessions() as session:
        assert session.query(ListingRow).count() == 0
        assert session.query(ListingProcessRunRow).count() == 0

    repository.save_listing(listing)
    with pytest.raises(ConcurrentStateChange, match="SAVING"):
        repository.persist_listing_work(
            listing,
            rule,
            origin_job_id=job_id,
            origin_attempt_id=attempt_id,
        )

    with repository._sessions() as session:
        assert session.query(ListingRow).count() == 1
        assert session.query(ListingProcessRunRow).count() == 0


def test_exact_job_and_dispatched_notification_queries_are_scoped(
    repository, listing
) -> None:
    first_job_id = repository.create_job("rule-test")
    second_job_id = repository.create_job("rule-other")
    listing_id, _ = repository.save_listing(listing)
    run_id, _ = repository.get_or_create_listing_run(listing_id, "rule-test")
    for state in (
        ListingRunState.NORMALIZED,
        ListingRunState.DEDUP_CHECKED,
        ListingRunState.RULE_EVALUATED,
        ListingRunState.RAG_RETRIEVED,
        ListingRunState.AGENT_EVALUATED,
    ):
        repository.advance_listing_run(run_id, state)
    notification_id, _ = repository.queue_notification_for_run(
        run_id,
        decision_version="mercari-v1",
        target_session="aiocqhttp:group:123",
        message_text="message",
    )

    assert repository.get_job(first_job_id).id == first_job_id
    assert repository.get_job(second_job_id).id == second_job_id
    assert repository.count_dispatched_notifications("rule-test") == 0
    repository.claim_notification(notification_id)
    repository.finalize_notification_sent(notification_id)
    assert repository.count_dispatched_notifications("rule-test") == 1
    assert repository.count_dispatched_notifications("rule-other") == 0


def test_illegal_attempt_transition_leaves_stored_state_unchanged(repository) -> None:
    job_id = repository.create_job("rule-1")
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)

    with pytest.raises(TransitionError):
        repository.advance_attempt(attempt_id, CrawlAttemptState.SUCCEEDED)

    assert repository.get_attempt(attempt_id).state is CrawlAttemptState.CREATED


def test_unknown_attempt_fields_are_rejected(repository) -> None:
    job_id = repository.create_job("rule-1")
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)

    with pytest.raises(ValueError, match="unsupported attempt fields"):
        repository.advance_attempt(
            attempt_id, CrawlAttemptState.REQUESTING, raw_response="secret response body"
        )

    assert repository.get_attempt(attempt_id).state is CrawlAttemptState.CREATED


def test_attempt_field_sanitization_redacts_secrets_and_response_bodies(repository) -> None:
    job_id = repository.create_job("rule-1")
    repository.activate_job(job_id)
    def fail_attempt(unsafe_summary: str, **fields: object) -> int:
        attempt_id = repository.create_attempt(job_id)
        repository.advance_attempt(
            attempt_id,
            CrawlAttemptState.FAILED,
            error_summary=unsafe_summary,
            **fields,
        )

        return attempt_id

    secret_attempts = [
        fail_attempt(
            unsafe_summary,
            http_status=401,
            error_type="RequestError",
            item_count=0,
        )
        for unsafe_summary in (
            "Cookie: session=private-value",
            "Authorization: Bearer top-secret-token",
            "token=private-value",
            "password=private-value",
            "secret=private-value",
        )
    ]
    body_attempt = fail_attempt('{"complete": "response body"}')
    long_body_attempt = fail_attempt("x" * 241)

    secret = repository.get_attempt(secret_attempts[0])
    body = repository.get_attempt(body_attempt)
    long_body = repository.get_attempt(long_body_attempt)
    assert secret.http_status == 401
    assert secret.item_count == 0
    assert secret.error_type == "RequestError"
    assert secret.error_summary == "sensitive error detail redacted"
    for attempt_id in secret_attempts:
        assert repository.get_attempt(attempt_id).error_summary == "sensitive error detail redacted"
    assert body.error_summary == "response body omitted"
    assert long_body.error_summary == "error summary omitted"


def test_attempt_timestamps_are_normalized_to_utc_on_read(repository) -> None:
    job_id = repository.create_job("rule-1")
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)
    tokyo_time = datetime(2026, 7, 28, 15, 30, tzinfo=timezone(timedelta(hours=9)))

    repository.advance_attempt(
        attempt_id,
        CrawlAttemptState.REQUESTING,
        started_at=tokyo_time,
        next_retry_at=tokyo_time,
    )

    attempt = repository.get_attempt(attempt_id)
    expected = datetime(2026, 7, 28, 6, 30, tzinfo=timezone.utc)
    assert attempt.started_at == expected
    assert attempt.next_retry_at == expected
    assert attempt.started_at.tzinfo is timezone.utc


def test_attempt_timestamps_reject_naive_datetimes(repository) -> None:
    job_id = repository.create_job("rule-1")
    repository.activate_job(job_id)
    attempt_id = repository.create_attempt(job_id)

    with pytest.raises(ValueError, match="timezone-aware"):
        repository.advance_attempt(
            attempt_id,
            CrawlAttemptState.REQUESTING,
            started_at=datetime(2026, 7, 28, 6, 30),
        )

    assert repository.get_attempt(attempt_id).state is CrawlAttemptState.CREATED


def test_all_required_tables_exist(repository) -> None:
    assert set(repository.table_names()) == {
        "crawl_attempts",
        "crawl_jobs",
        "listing_process_runs",
        "listings",
        "notifications",
    }


def test_status_snapshot_reports_latest_job_and_attempt_count(repository) -> None:
    job_id = repository.create_job("rule-1")
    repository.activate_job(job_id)
    repository.create_attempt(job_id)
    repository.finish_job(job_id, CrawlJobState.CANCELLED)

    snapshot = repository.get_status()

    assert snapshot.job_state is CrawlJobState.CANCELLED
    assert snapshot.attempt_count == 1
    assert snapshot.notification_count == 0


def test_separate_repository_callers_cannot_create_two_unfinished_attempts(tmp_path) -> None:
    database = tmp_path / "mercari.sqlite3"
    first_repo = Repository.open(database)
    second_repo = Repository.open(database)
    job_id = first_repo.create_job("rule-1")
    first_repo.activate_job(job_id)
    first_committed = Event()

    def create_first() -> int | Exception:
        try:
            return first_repo.create_attempt(job_id)
        except Exception as error:
            return error
        finally:
            first_committed.set()

    def create_second() -> int | Exception:
        first_committed.wait()
        try:
            return second_repo.create_attempt(job_id)
        except Exception as error:
            return error

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(create_first)
            second = executor.submit(create_second)
            results = [
                first.result(),
                second.result(),
            ]
    finally:
        first_repo.dispose()
        second_repo.dispose()

    assert sum(isinstance(result, int) for result in results) == 1
    assert sum(isinstance(result, ConcurrentStateChange) for result in results) == 1


def test_open_migrates_old_duplicate_attempts_and_installs_partial_index(tmp_path) -> None:
    database = tmp_path / "old-mercari.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE crawl_jobs (
                id INTEGER PRIMARY KEY,
                rule_id VARCHAR NOT NULL,
                state VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                started_at DATETIME,
                finished_at DATETIME
            );
            CREATE TABLE crawl_attempts (
                id INTEGER PRIMARY KEY,
                job_id INTEGER NOT NULL,
                attempt_no INTEGER NOT NULL,
                state VARCHAR NOT NULL,
                http_status INTEGER,
                error_type VARCHAR,
                error_summary VARCHAR,
                next_retry_at DATETIME,
                item_count INTEGER,
                started_at DATETIME,
                finished_at DATETIME,
                CONSTRAINT uq_attempt_number UNIQUE (job_id, attempt_no)
            );
            INSERT INTO crawl_jobs VALUES (1, 'rule-1', 'ACTIVE', '2026-01-01 00:00:00', '2026-01-01 00:00:00', NULL);
            INSERT INTO crawl_attempts VALUES (1, 1, 1, 'CREATED', NULL, NULL, NULL, NULL, NULL, NULL, NULL);
            INSERT INTO crawl_attempts VALUES (2, 1, 2, 'REQUESTING', NULL, NULL, NULL, NULL, NULL, '2026-01-01 00:01:00', NULL);
            """
        )
        connection.commit()
    finally:
        connection.close()

    repository = Repository.open(database)
    try:
        attempts = repository.attempts(1)
        assert [attempt.state for attempt in attempts] == [
            CrawlAttemptState.FAILED,
            CrawlAttemptState.REQUESTING,
        ]
        assert attempts[0].error_type == "migration_recovered"
        assert attempts[0].error_summary == "migration closed duplicate unfinished attempt"
        assert attempts[0].finished_at is not None
        assert attempts[0].finished_at.tzinfo is timezone.utc
        with pytest.raises(IntegrityError):
            with repository._engine.begin() as engine_connection:
                engine_connection.execute(
                    insert(CrawlAttemptRow).values(
                        job_id=1,
                        attempt_no=3,
                        state=CrawlAttemptState.CREATED.value,
                    )
                )
        with repository._engine.connect() as engine_connection:
            index_names = {
                row[1]
                for row in engine_connection.execute(text("PRAGMA index_list('crawl_attempts')"))
            }
        assert "uq_unfinished_attempt_per_job" in index_names
    finally:
        repository.dispose()

    reopened = Repository.open(database)
    try:
        assert [attempt.state for attempt in reopened.attempts(1)] == [
            CrawlAttemptState.FAILED,
            CrawlAttemptState.REQUESTING,
        ]
    finally:
        reopened.dispose()


def test_failed_listing_run_creates_the_next_run_number(
    repository: Repository, listing: Listing
) -> None:
    listing_id, _ = repository.save_listing(listing)
    first_id, first_created = repository.get_or_create_listing_run(
        listing_id, "rule-1"
    )
    repository.advance_listing_run(first_id, ListingRunState.FAILED, "temporary")

    second_id, second_created = repository.get_or_create_listing_run(
        listing_id, "rule-1"
    )

    assert first_created is True
    assert second_created is True
    assert second_id != first_id
    assert repository.get_listing_run(first_id).run_no == 1
    assert repository.get_listing_run(second_id).run_no == 2
    assert repository.get_listing_run(first_id).state is ListingRunState.FAILED


def test_stale_listing_run_is_failed_and_retried(
    repository: Repository, listing: Listing
) -> None:
    listing_id, _ = repository.save_listing(listing)
    first_id, _ = repository.get_or_create_listing_run(listing_id, "rule-1")
    stale_at = datetime.now(timezone.utc) - timedelta(hours=1)
    with repository._engine.begin() as connection:
        connection.execute(
            update(ListingProcessRunRow)
            .where(ListingProcessRunRow.id == first_id)
            .values(updated_at=stale_at)
        )

    second_id, created = repository.get_or_create_listing_run(
        listing_id,
        "rule-1",
        stale_after=timedelta(seconds=30),
    )

    assert created is True
    assert second_id != first_id
    first = repository.get_listing_run(first_id)
    assert first.state is ListingRunState.FAILED
    assert first.error_summary == "stale process run recovered after restart"
    assert repository.get_listing_run(second_id).run_no == 2


@pytest.mark.parametrize(
    "stale_state",
    [
        ListingRunState.NORMALIZED,
        ListingRunState.DEDUP_CHECKED,
        ListingRunState.RULE_EVALUATED,
        ListingRunState.RAG_RETRIEVED,
        ListingRunState.AGENT_EVALUATED,
    ],
)
def test_recover_stale_snapshotted_inflight_run_copies_durable_work(
    repository: Repository,
    listing: Listing,
    stale_state: ListingRunState,
) -> None:
    rule = _durable_rule(f"rule-{stale_state.value.lower()}")
    job_id, attempt_id = _saving_origin(repository, rule.id)
    run_id, _ = repository.persist_listing_work(
        listing,
        rule,
        origin_job_id=job_id,
        origin_attempt_id=attempt_id,
    )
    _advance_listing_run_to(repository, run_id, stale_state)
    stale_at = datetime.now(timezone.utc) - timedelta(hours=1)
    with repository._engine.begin() as connection:
        connection.execute(
            update(ListingProcessRunRow)
            .where(ListingProcessRunRow.id == run_id)
            .values(updated_at=stale_at)
        )

    replacement_ids = repository.recover_stale_listing_work(
        stale_after=timedelta(seconds=30)
    )

    assert len(replacement_ids) == 1
    old = repository.get_listing_run(run_id)
    replacement = repository.get_listing_run(replacement_ids[0])
    assert old.state is ListingRunState.FAILED
    assert old.error_summary == (
        "crash_recovered: stale in-flight listing work"
    )
    assert replacement.listing_id == old.listing_id
    assert replacement.watch_rule_id == old.watch_rule_id
    assert replacement.run_no == old.run_no + 1
    assert replacement.state is ListingRunState.DISCOVERED
    assert replacement.rule_snapshot_json == old.rule_snapshot_json
    assert replacement.origin_job_id == old.origin_job_id
    assert replacement.origin_attempt_id == old.origin_attempt_id


def test_recover_stale_listing_work_does_not_steal_fresh_inflight_run(
    repository: Repository,
    listing: Listing,
) -> None:
    rule = _durable_rule("rule-fresh")
    job_id, attempt_id = _saving_origin(repository, rule.id)
    run_id, _ = repository.persist_listing_work(
        listing,
        rule,
        origin_job_id=job_id,
        origin_attempt_id=attempt_id,
    )
    _advance_listing_run_to(
        repository,
        run_id,
        ListingRunState.RAG_RETRIEVED,
    )

    replacement_ids = repository.recover_stale_listing_work(
        stale_after=timedelta(hours=1)
    )

    assert replacement_ids == []
    assert (
        repository.get_listing_run(run_id).state
        is ListingRunState.RAG_RETRIEVED
    )
    with repository._sessions() as session:
        assert session.query(ListingProcessRunRow).count() == 1


def test_recover_stale_listing_work_excludes_outbox_owned_run(
    repository: Repository,
    listing: Listing,
) -> None:
    rule = _durable_rule("rule-outbox-owned")
    job_id, attempt_id = _saving_origin(repository, rule.id)
    run_id, _ = repository.persist_listing_work(
        listing,
        rule,
        origin_job_id=job_id,
        origin_attempt_id=attempt_id,
    )
    _advance_listing_run_to(
        repository,
        run_id,
        ListingRunState.AGENT_EVALUATED,
    )
    notification_id, _ = repository.queue_notification_for_run(
        run_id,
        decision_version=rule.decision_version,
        target_session=rule.target_session,
        message_text="queued",
    )
    stale_at = datetime.now(timezone.utc) - timedelta(hours=1)
    with repository._engine.begin() as connection:
        connection.execute(
            update(ListingProcessRunRow)
            .where(ListingProcessRunRow.id == run_id)
            .values(updated_at=stale_at)
        )

    replacement_ids = repository.recover_stale_listing_work(
        stale_after=timedelta(seconds=30)
    )

    assert replacement_ids == []
    assert (
        repository.get_listing_run(run_id).state
        is ListingRunState.NOTIFICATION_QUEUED
    )

    claimed = repository.claim_notification(notification_id)
    assert claimed is not None
    assert (
        repository.get_notification(notification_id).state
        is NotificationState.SENDING
    )
    assert repository.recover_stale_listing_work(
        stale_after=timedelta(seconds=30)
    ) == []

    repository.mark_notification_verify_required(
        notification_id,
        "uncertain platform result",
    )
    assert (
        repository.get_notification(notification_id).state
        is NotificationState.VERIFY_REQUIRED
    )
    assert (
        repository.get_listing_run(run_id).state
        is ListingRunState.FAILED
    )
    assert repository.recover_stale_listing_work(
        stale_after=timedelta(seconds=30)
    ) == []
    with repository._sessions() as session:
        assert session.query(ListingProcessRunRow).count() == 1


@pytest.mark.parametrize(
    "terminal_state",
    [ListingRunState.REJECTED, ListingRunState.NOTIFIED],
)
def test_terminal_listing_run_suppresses_reprocessing(
    repository: Repository,
    listing: Listing,
    terminal_state: ListingRunState,
) -> None:
    listing_id, _ = repository.save_listing(listing)
    run_id, _ = repository.get_or_create_listing_run(listing_id, "rule-1")
    path = [
        ListingRunState.NORMALIZED,
        ListingRunState.DEDUP_CHECKED,
        ListingRunState.RULE_EVALUATED,
    ]
    if terminal_state is ListingRunState.REJECTED:
        path.append(ListingRunState.REJECTED)
    else:
        path.extend(
            [
                ListingRunState.RAG_RETRIEVED,
                ListingRunState.AGENT_EVALUATED,
                ListingRunState.NOTIFICATION_QUEUED,
                ListingRunState.NOTIFIED,
            ]
        )
    for state in path:
        repository.advance_listing_run(run_id, state)

    repeated_id, created = repository.get_or_create_listing_run(
        listing_id, "rule-1"
    )

    assert repeated_id == run_id
    assert created is False


def test_concurrent_listing_run_get_or_create_returns_one_fresh_run(
    tmp_path, listing: Listing
) -> None:
    database = tmp_path / "listing-run-race.sqlite3"
    first_repo = Repository.open(database)
    second_repo = Repository.open(database)
    listing_id, _ = first_repo.save_listing(listing)
    gate = Event()

    def create(repo: Repository) -> tuple[int, bool]:
        gate.wait()
        return repo.get_or_create_listing_run(listing_id, "rule-1")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(create, first_repo),
                executor.submit(create, second_repo),
            ]
            gate.set()
            results = [future.result() for future in futures]
        run_ids = {run_id for run_id, _ in results}
        assert len(run_ids) == 1
        assert sum(created for _, created in results) == 1
        run = first_repo.get_listing_run(next(iter(run_ids)))
        assert run.run_no == 1
        assert run.state is ListingRunState.DISCOVERED
    finally:
        first_repo.dispose()
        second_repo.dispose()


def test_concurrent_first_listing_persistence_keeps_both_rule_runs(
    tmp_path,
    listing: Listing,
) -> None:
    database = tmp_path / "first-listing-two-rules.sqlite3"
    first_repo = Repository.open(database)
    second_repo = Repository.open(database)
    first_rule = _durable_rule("rule-concurrent-first")
    second_rule = _durable_rule("rule-concurrent-second")
    first_job, first_attempt = _saving_origin(first_repo, first_rule.id)
    second_job, second_attempt = _saving_origin(
        second_repo,
        second_rule.id,
    )
    both_at_insert = Barrier(2, timeout=5)
    first_committed = Event()

    def is_listing_insert(statement: str) -> bool:
        return statement.lstrip().lower().startswith(
            "insert into listings "
        )

    def first_insert_hook(
        connection,
        cursor,
        statement: str,
        parameters,
        context,
        executemany: bool,
    ) -> None:
        if is_listing_insert(statement):
            both_at_insert.wait()

    def second_insert_hook(
        connection,
        cursor,
        statement: str,
        parameters,
        context,
        executemany: bool,
    ) -> None:
        if is_listing_insert(statement):
            both_at_insert.wait()
            assert first_committed.wait(timeout=5)

    event.listen(
        first_repo._engine,
        "before_cursor_execute",
        first_insert_hook,
    )
    event.listen(
        second_repo._engine,
        "before_cursor_execute",
        second_insert_hook,
    )

    def persist_first() -> tuple[int, bool] | Exception:
        try:
            return first_repo.persist_listing_work(
                listing,
                first_rule,
                origin_job_id=first_job,
                origin_attempt_id=first_attempt,
            )
        except Exception as error:
            return error
        finally:
            first_committed.set()

    def persist_second() -> tuple[int, bool] | Exception:
        try:
            return second_repo.persist_listing_work(
                listing,
                second_rule,
                origin_job_id=second_job,
                origin_attempt_id=second_attempt,
            )
        except Exception as error:
            return error

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(persist_first)
            second_future = executor.submit(persist_second)
            results = [
                first_future.result(timeout=10),
                second_future.result(timeout=10),
            ]
        assert all(isinstance(result, tuple) for result in results)
        successful = [
            result for result in results if isinstance(result, tuple)
        ]
        assert all(created for _, created in successful)
        assert len({run_id for run_id, _ in successful}) == 2
        with first_repo._sessions() as session:
            listings = list(session.query(ListingRow))
            runs = list(
                session.query(ListingProcessRunRow).order_by(
                    ListingProcessRunRow.id
                )
            )
        assert len(listings) == 1
        assert len(runs) == 2
        assert {run.watch_rule_id for run in runs} == {
            first_rule.id,
            second_rule.id,
        }
        expected = {
            first_rule.id: (first_rule, first_job, first_attempt),
            second_rule.id: (second_rule, second_job, second_attempt),
        }
        for run in runs:
            rule, job_id, attempt_id = expected[run.watch_rule_id]
            assert ListingRunState(run.state) is ListingRunState.DISCOVERED
            assert run.rule_snapshot_json == rule.model_dump_json()
            assert run.origin_job_id == job_id
            assert run.origin_attempt_id == attempt_id
    finally:
        event.remove(
            first_repo._engine,
            "before_cursor_execute",
            first_insert_hook,
        )
        event.remove(
            second_repo._engine,
            "before_cursor_execute",
            second_insert_hook,
        )
        first_repo.dispose()
        second_repo.dispose()


def test_only_one_caller_claims_a_discovered_listing_run(
    tmp_path,
    listing: Listing,
) -> None:
    database = tmp_path / "listing-run-claim-race.sqlite3"
    first_repo = Repository.open(database)
    second_repo = Repository.open(database)
    listing_id, _ = first_repo.save_listing(listing)
    run_id, _ = first_repo.get_or_create_listing_run(
        listing_id,
        "rule-1",
    )
    gate = Event()

    def claim(repo: Repository) -> bool:
        gate.wait()
        return repo.claim_discovered_listing_run(run_id)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(claim, first_repo),
                executor.submit(claim, second_repo),
            ]
            gate.set()
            results = [future.result() for future in futures]
        assert sorted(results) == [False, True]
        assert (
            first_repo.get_listing_run(run_id).state
            is ListingRunState.NORMALIZED
        )
    finally:
        first_repo.dispose()
        second_repo.dispose()


def test_open_idempotently_migrates_legacy_listing_run_uniqueness(tmp_path) -> None:
    database = tmp_path / "legacy-listing-runs.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE listings (
                id INTEGER PRIMARY KEY,
                marketplace VARCHAR NOT NULL,
                external_id VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                price_jpy INTEGER NOT NULL,
                url VARCHAR NOT NULL,
                image_url VARCHAR,
                seller_name VARCHAR,
                published_at DATETIME,
                discovered_at DATETIME NOT NULL,
                CONSTRAINT uq_listing_external UNIQUE (marketplace, external_id)
            );
            CREATE TABLE listing_process_runs (
                id INTEGER PRIMARY KEY,
                listing_id INTEGER NOT NULL,
                watch_rule_id VARCHAR NOT NULL,
                state VARCHAR NOT NULL,
                error_summary VARCHAR,
                created_at DATETIME NOT NULL,
                CONSTRAINT uq_listing_run_rule UNIQUE (listing_id, watch_rule_id)
            );
            CREATE TABLE notifications (
                id INTEGER PRIMARY KEY,
                listing_id INTEGER NOT NULL,
                watch_rule_id VARCHAR NOT NULL,
                decision_version VARCHAR NOT NULL,
                target_session VARCHAR NOT NULL,
                message_text VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                sent_at DATETIME,
                CONSTRAINT uq_notification_idempotency
                    UNIQUE (listing_id, watch_rule_id, decision_version)
            );
            INSERT INTO listings VALUES (
                1, 'mercari', 'legacy-1', 'legacy', '', 100,
                'https://example.invalid/legacy-1', NULL, NULL, NULL,
                '2026-01-01 00:00:00'
            );
            INSERT INTO listing_process_runs VALUES (
                1, 1, 'rule-1', 'FAILED', 'temporary', '2026-01-01 00:00:00'
            );
            INSERT INTO notifications VALUES (
                1, 1, 'rule-1', 'legacy-unsent',
                'aiocqhttp:group:1', 'message', '2026-01-01 00:00:00', NULL
            );
            INSERT INTO notifications VALUES (
                2, 1, 'rule-1', 'legacy-sent',
                'aiocqhttp:group:1', 'message', '2026-01-01 00:00:00',
                '2026-01-01 00:01:00'
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    repository = Repository.open(database)
    try:
        legacy = repository.get_listing_run(1)
        assert legacy.run_no == 1
        assert (
            repository.get_notification(1).state
            is NotificationState.VERIFY_REQUIRED
        )
        assert repository.get_notification(1).process_run_id == 1
        assert repository.get_notification(2).state is NotificationState.SENT
        second_id, created = repository.get_or_create_listing_run(1, "rule-1")
        assert created is True
        assert repository.get_listing_run(second_id).run_no == 2
    finally:
        repository.dispose()

    reopened = Repository.open(database)
    try:
        assert reopened.get_listing_run(1).run_no == 1
        assert reopened.get_listing_run(2).run_no == 2
        assert (
            reopened.get_notification(1).state
            is NotificationState.VERIFY_REQUIRED
        )
        assert reopened.get_notification(2).state is NotificationState.SENT
        with reopened._engine.connect() as engine_connection:
            columns = {
                row[1]
                for row in engine_connection.execute(
                    text("PRAGMA table_info('listing_process_runs')")
                )
            }
        assert {
            "run_no",
            "updated_at",
            "rule_snapshot_json",
            "origin_job_id",
            "origin_attempt_id",
        } <= columns
    finally:
        reopened.dispose()
