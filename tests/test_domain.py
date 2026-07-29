from datetime import datetime, timezone
import json
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
    WatchRule,
    assert_transition,
    with_stable_rule_id,
)


class DomainTests(unittest.TestCase):
    def test_states_remain_plain_string_values(self) -> None:
        state = CrawlJobState.PENDING

        self.assertIsInstance(state, str)
        self.assertEqual(str(state), "PENDING")
        self.assertEqual(json.dumps({"state": state}), '{"state": "PENDING"}')

    def test_rule_fingerprint_tracks_only_material_configuration(self) -> None:
        base = WatchRule(
            id="placeholder",
            name="base",
            include_keywords=("缶バッジ", "月村手毬"),
            exclude_keywords=("欠品", "ジャンク"),
            max_price_jpy=1500,
            interval_seconds=60,
            target_session="aiocqhttp:group:123",
        )
        versioned = with_stable_rule_id(base)
        reordered = with_stable_rule_id(
            base.model_copy(
                update={
                    "id": "another-placeholder",
                    "name": "renamed",
                    "include_keywords": tuple(reversed(base.include_keywords)),
                    "exclude_keywords": tuple(reversed(base.exclude_keywords)),
                    "interval_seconds": 3600,
                    "enabled": False,
                }
            )
        )

        self.assertEqual(versioned.id, reordered.id)
        self.assertTrue(versioned.id.startswith("mercari-rule-v1-"))
        for field, value in (
            ("include_keywords", ("藤田ことね",)),
            ("exclude_keywords", ("破損",)),
            ("max_price_jpy", 2500),
            ("target_session", "aiocqhttp:group:456"),
            ("marketplace", "mercari-jp"),
            ("collector_identity", "mock-collector-v2"),
            ("decision_version", "mercari-v2"),
        ):
            with self.subTest(field=field):
                changed = with_stable_rule_id(
                    base.model_copy(update={field: value})
                )
                self.assertNotEqual(versioned.id, changed.id)

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
