from datetime import datetime, timezone

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
