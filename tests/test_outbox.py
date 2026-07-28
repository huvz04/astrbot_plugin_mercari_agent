from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import event

from astrbot_plugin_mercari_agent.domain import (
    Listing,
    ListingRunState,
    NotificationState,
)
from astrbot_plugin_mercari_agent.graph import (
    drain_queued_notifications,
    dispatch_notification,
    recover_notification_outbox,
)
from astrbot_plugin_mercari_agent.storage import Repository


class RecordingNotifier:
    def __init__(self, outcome: bool | BaseException = True) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str]] = []

    async def send(self, target_session: str, text: str) -> bool:
        self.calls.append((target_session, text))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.fixture
def repository(tmp_path: Path):
    value = Repository.open(tmp_path / "outbox.sqlite3")
    yield value
    value.dispose()


def _ready_run(
    repository: Repository,
    *,
    external_id: str = "outbox-1",
    rule_id: str = "rule-1",
) -> tuple[int, int]:
    listing_id, _ = repository.save_listing(
        Listing(
            marketplace="mercari",
            external_id=external_id,
            title=f"listing {external_id}",
            price_jpy=1000,
            url=f"https://example.invalid/{external_id}",
            discovered_at=datetime.now(timezone.utc),
        )
    )
    run_id, created = repository.get_or_create_listing_run(listing_id, rule_id)
    assert created is True
    for state in (
        ListingRunState.NORMALIZED,
        ListingRunState.DEDUP_CHECKED,
        ListingRunState.RULE_EVALUATED,
        ListingRunState.RAG_RETRIEVED,
        ListingRunState.AGENT_EVALUATED,
    ):
        repository.advance_listing_run(run_id, state)
    return listing_id, run_id


def _queue(repository: Repository, run_id: int) -> tuple[int, bool]:
    return repository.queue_notification_for_run(
        run_id,
        decision_version="decision-v1",
        target_session="aiocqhttp:group:123",
        message_text="hello",
    )


def test_queue_and_run_transition_roll_back_together_on_database_failure(
    repository: Repository,
) -> None:
    _, run_id = _ready_run(repository)

    def fail_run_update(
        connection,
        cursor,
        statement: str,
        parameters,
        context,
        executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("UPDATE LISTING_PROCESS_RUNS"):
            raise RuntimeError("injected queue boundary failure")

    event.listen(repository._engine, "before_cursor_execute", fail_run_update)
    try:
        with pytest.raises(RuntimeError, match="queue boundary"):
            _queue(repository, run_id)
    finally:
        event.remove(repository._engine, "before_cursor_execute", fail_run_update)

    assert repository.count_notifications() == 0
    assert (
        repository.get_listing_run(run_id).state
        is ListingRunState.AGENT_EVALUATED
    )


def test_successful_dispatch_atomically_marks_notification_and_run(
    repository: Repository,
) -> None:
    _, run_id = _ready_run(repository)
    notification_id, dispatchable = _queue(repository, run_id)
    notifier = RecordingNotifier(True)

    result = asyncio.run(
        dispatch_notification(repository, notifier, notification_id)
    )

    notification = repository.get_notification(notification_id)
    assert dispatchable is True
    assert result is NotificationState.SENT
    assert notification.state is NotificationState.SENT
    assert notification.attempt_at is not None
    assert notification.sent_at is not None
    assert notification.last_error is None
    assert repository.get_listing_run(run_id).state is ListingRunState.NOTIFIED
    assert notifier.calls == [("aiocqhttp:group:123", "hello")]


def test_known_failure_can_requeue_same_notification_for_a_new_run(
    repository: Repository,
) -> None:
    listing_id, first_run_id = _ready_run(repository)
    notification_id, _ = _queue(repository, first_run_id)
    failed = RecordingNotifier(False)

    first_result = asyncio.run(
        dispatch_notification(repository, failed, notification_id)
    )
    second_run_id, created = repository.get_or_create_listing_run(
        listing_id, "rule-1"
    )
    assert created is True
    for state in (
        ListingRunState.NORMALIZED,
        ListingRunState.DEDUP_CHECKED,
        ListingRunState.RULE_EVALUATED,
        ListingRunState.RAG_RETRIEVED,
        ListingRunState.AGENT_EVALUATED,
    ):
        repository.advance_listing_run(second_run_id, state)
    repeated_id, dispatchable = _queue(repository, second_run_id)
    succeeded = RecordingNotifier(True)

    second_result = asyncio.run(
        dispatch_notification(repository, succeeded, repeated_id)
    )

    assert first_result is NotificationState.FAILED_KNOWN
    assert repeated_id == notification_id
    assert dispatchable is True
    assert second_result is NotificationState.SENT
    assert repository.count_notifications() == 1
    assert (
        repository.get_listing_run(first_run_id).state
        is ListingRunState.FAILED
    )
    assert (
        repository.get_listing_run(second_run_id).state
        is ListingRunState.NOTIFIED
    )


def test_notifier_exception_requires_verification_and_is_never_auto_resent(
    repository: Repository,
) -> None:
    _, run_id = _ready_run(repository)
    notification_id, _ = _queue(repository, run_id)
    notifier = RecordingNotifier(RuntimeError("Authorization: secret"))

    result = asyncio.run(
        dispatch_notification(repository, notifier, notification_id)
    )
    asyncio.run(recover_notification_outbox(repository, notifier))

    notification = repository.get_notification(notification_id)
    assert result is NotificationState.VERIFY_REQUIRED
    assert notification.state is NotificationState.VERIFY_REQUIRED
    assert notification.last_error == "sensitive error detail redacted"
    assert repository.get_listing_run(run_id).state is ListingRunState.FAILED
    assert len(notifier.calls) == 1


def test_startup_dispatches_queued_but_reconciles_sending_without_resend(
    repository: Repository,
) -> None:
    _, queued_run_id = _ready_run(repository, external_id="queued")
    queued_id, _ = _queue(repository, queued_run_id)
    _, sending_run_id = _ready_run(repository, external_id="sending")
    sending_id, _ = _queue(repository, sending_run_id)
    repository.claim_notification(sending_id)
    notifier = RecordingNotifier(True)

    asyncio.run(recover_notification_outbox(repository, notifier))

    assert repository.get_notification(queued_id).state is NotificationState.SENT
    assert (
        repository.get_notification(sending_id).state
        is NotificationState.VERIFY_REQUIRED
    )
    assert (
        repository.get_listing_run(queued_run_id).state
        is ListingRunState.NOTIFIED
    )
    assert (
        repository.get_listing_run(sending_run_id).state
        is ListingRunState.FAILED
    )
    assert notifier.calls == [("aiocqhttp:group:123", "hello")]


def test_claim_failure_never_calls_external_notifier(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id = _ready_run(repository)
    notification_id, _ = _queue(repository, run_id)
    notifier = RecordingNotifier(True)

    def fail_claim(notification_id: int):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(repository, "claim_notification", fail_claim)

    result = asyncio.run(
        dispatch_notification(repository, notifier, notification_id)
    )

    assert result is NotificationState.QUEUED
    assert notifier.calls == []
    assert (
        repository.get_notification(notification_id).state
        is NotificationState.QUEUED
    )
    assert (
        repository.get_listing_run(run_id).state
        is ListingRunState.NOTIFICATION_QUEUED
    )


def test_two_normal_cycles_retry_transient_claim_failure_exactly_once(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id = _ready_run(repository)
    notification_id, _ = _queue(repository, run_id)
    notifier = RecordingNotifier(True)
    real_claim = repository.claim_notification
    claim_calls = 0

    def transient_claim(notification_id: int):
        nonlocal claim_calls
        claim_calls += 1
        if claim_calls == 1:
            raise RuntimeError("Authorization: private")
        return real_claim(notification_id)

    monkeypatch.setattr(repository, "claim_notification", transient_claim)

    first_errors = asyncio.run(
        drain_queued_notifications(repository, notifier)
    )
    second_errors = asyncio.run(
        drain_queued_notifications(repository, notifier)
    )
    third_errors = asyncio.run(
        drain_queued_notifications(repository, notifier)
    )

    assert [str(error) for error in first_errors] == [
        "Authorization: private"
    ]
    assert second_errors == []
    assert third_errors == []
    assert claim_calls == 2
    assert notifier.calls == [("aiocqhttp:group:123", "hello")]
    assert (
        repository.get_notification(notification_id).state
        is NotificationState.SENT
    )
    assert repository.get_listing_run(run_id).state is ListingRunState.NOTIFIED


def test_finalization_failure_after_platform_acceptance_requires_verification(
    repository: Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_id = _ready_run(repository)
    notification_id, _ = _queue(repository, run_id)
    notifier = RecordingNotifier(True)

    def fail_finalize(notification_id: int) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        repository,
        "finalize_notification_sent",
        fail_finalize,
    )

    result = asyncio.run(
        dispatch_notification(repository, notifier, notification_id)
    )
    asyncio.run(recover_notification_outbox(repository, notifier))

    assert result is NotificationState.VERIFY_REQUIRED
    assert (
        repository.get_notification(notification_id).state
        is NotificationState.VERIFY_REQUIRED
    )
    assert repository.get_listing_run(run_id).state is ListingRunState.FAILED
    assert len(notifier.calls) == 1
