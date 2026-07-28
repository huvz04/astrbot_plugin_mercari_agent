from datetime import datetime, timedelta, timezone

import pytest

from astrbot_plugin_mercari_agent.domain import (
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    TransitionError,
)
from astrbot_plugin_mercari_agent.storage import Repository


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
    secret_attempts = [repository.create_attempt(job_id) for _ in range(5)]
    body_attempt = repository.create_attempt(job_id)
    long_body_attempt = repository.create_attempt(job_id)

    for attempt_id, unsafe_summary in zip(
        secret_attempts,
        (
            "Cookie: session=private-value",
            "Authorization: Bearer top-secret-token",
            "token=private-value",
            "password=private-value",
            "secret=private-value",
        ),
        strict=True,
    ):
        repository.advance_attempt(
            attempt_id,
            CrawlAttemptState.FAILED,
            http_status=401,
            error_type="RequestError",
            error_summary=unsafe_summary,
            item_count=0,
        )
    repository.advance_attempt(
        body_attempt,
        CrawlAttemptState.FAILED,
        error_summary='{"complete": "response body"}',
    )
    repository.advance_attempt(
        long_body_attempt,
        CrawlAttemptState.FAILED,
        error_summary="x" * 241,
    )

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
