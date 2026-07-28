from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from astrbot_plugin_mercari_agent.domain import (
    ATTEMPT_TRANSITIONS,
    JOB_TRANSITIONS,
    LISTING_RUN_TRANSITIONS,
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    ListingRunState,
    TransitionError,
    assert_transition,
)


class DomainTests(unittest.TestCase):
    def test_attempt_can_progress_but_cannot_rewind(self) -> None:
        assert_transition(
            CrawlAttemptState.REQUESTING,
            CrawlAttemptState.RECEIVED,
            ATTEMPT_TRANSITIONS,
        )

        with self.assertRaisesRegex(TransitionError, "illegal transition"):
            assert_transition(
                CrawlAttemptState.RECEIVED,
                CrawlAttemptState.REQUESTING,
                ATTEMPT_TRANSITIONS,
            )

    def test_listing_requires_non_blank_external_id(self) -> None:
        with self.assertRaises(ValidationError):
            Listing(
                marketplace="mercari",
                external_id="   ",
                title="月村手毬 缶バッジ",
                price_jpy=1200,
                url="https://example.invalid/item/1",
                discovered_at=datetime.now(timezone.utc),
            )

    def test_terminal_states_reject_all_outgoing_transitions(self) -> None:
        transition_sets = (
            (
                JOB_TRANSITIONS,
                (
                    CrawlJobState.SUCCEEDED,
                    CrawlJobState.EXHAUSTED,
                    CrawlJobState.CANCELLED,
                ),
                CrawlJobState,
            ),
            (
                ATTEMPT_TRANSITIONS,
                (CrawlAttemptState.SUCCEEDED, CrawlAttemptState.FAILED),
                CrawlAttemptState,
            ),
            (
                LISTING_RUN_TRANSITIONS,
                (
                    ListingRunState.REJECTED,
                    ListingRunState.NOTIFIED,
                    ListingRunState.FAILED,
                ),
                ListingRunState,
            ),
        )
        for transitions, terminal_states, enum_type in transition_sets:
            for state in terminal_states:
                for target in enum_type:
                    with self.subTest(state=state, target=target):
                        with self.assertRaises(TransitionError):
                            assert_transition(state, target, transitions)
