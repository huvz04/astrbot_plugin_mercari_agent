from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3
from threading import Event

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError

from astrbot_plugin_mercari_agent.domain import (
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    TransitionError,
)
from astrbot_plugin_mercari_agent.storage import (
    ConcurrentStateChange,
    CrawlAttemptRow,
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


def test_exact_job_and_sent_notification_queries_are_scoped(
    repository, listing
) -> None:
    first_job_id = repository.create_job("rule-test")
    second_job_id = repository.create_job("rule-other")
    listing_id, _ = repository.save_listing(listing)
    notification_id, _ = repository.queue_notification(
        listing_id,
        "rule-test",
        "mercari-v1",
        "aiocqhttp:group:123",
        "message",
    )

    assert repository.get_job(first_job_id).id == first_job_id
    assert repository.get_job(second_job_id).id == second_job_id
    assert repository.count_sent_notifications("rule-test") == 0
    repository.mark_notification_sent(notification_id)
    assert repository.count_sent_notifications("rule-test") == 1
    assert repository.count_sent_notifications("rule-other") == 0


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
