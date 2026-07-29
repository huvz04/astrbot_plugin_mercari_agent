"""Immutable domain values and forward-only state transitions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Literal, TypeVar
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_MARKETPLACE = "mercari"
MOCK_COLLECTOR_IDENTITY = "mock-collector-v1"
DEFAULT_DECISION_VERSION = "mercari-v1"
_RULE_ID_PREFIX = "mercari-rule-v1-"


class StringEnum(str, Enum):
    """String-valued enum with consistent behavior on Python 3.10+."""

    def __str__(self) -> str:
        return self.value


class CrawlJobState(StringEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    EXHAUSTED = "EXHAUSTED"
    CANCELLED = "CANCELLED"


class CrawlAttemptState(StringEnum):
    CREATED = "CREATED"
    REQUESTING = "REQUESTING"
    RECEIVED = "RECEIVED"
    PARSING = "PARSING"
    SAVING = "SAVING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ListingRunState(StringEnum):
    DISCOVERED = "DISCOVERED"
    NORMALIZED = "NORMALIZED"
    DEDUP_CHECKED = "DEDUP_CHECKED"
    RULE_EVALUATED = "RULE_EVALUATED"
    REJECTED = "REJECTED"
    RAG_RETRIEVED = "RAG_RETRIEVED"
    AGENT_EVALUATED = "AGENT_EVALUATED"
    NOTIFICATION_QUEUED = "NOTIFICATION_QUEUED"
    NOTIFIED = "NOTIFIED"
    FAILED = "FAILED"


class NotificationState(StringEnum):
    QUEUED = "QUEUED"
    SENDING = "SENDING"
    SENT = "SENT"
    FAILED_KNOWN = "FAILED_KNOWN"
    VERIFY_REQUIRED = "VERIFY_REQUIRED"


JOB_TRANSITIONS: dict[CrawlJobState, frozenset[CrawlJobState]] = {
    CrawlJobState.PENDING: frozenset({CrawlJobState.ACTIVE}),
    CrawlJobState.ACTIVE: frozenset(
        {
            CrawlJobState.SUCCEEDED,
            CrawlJobState.EXHAUSTED,
            CrawlJobState.CANCELLED,
        }
    ),
    CrawlJobState.SUCCEEDED: frozenset(),
    CrawlJobState.EXHAUSTED: frozenset(),
    CrawlJobState.CANCELLED: frozenset(),
}

ATTEMPT_TRANSITIONS: dict[CrawlAttemptState, frozenset[CrawlAttemptState]] = {
    CrawlAttemptState.CREATED: frozenset(
        {CrawlAttemptState.REQUESTING, CrawlAttemptState.FAILED}
    ),
    CrawlAttemptState.REQUESTING: frozenset(
        {CrawlAttemptState.RECEIVED, CrawlAttemptState.FAILED}
    ),
    CrawlAttemptState.RECEIVED: frozenset(
        {CrawlAttemptState.PARSING, CrawlAttemptState.FAILED}
    ),
    CrawlAttemptState.PARSING: frozenset(
        {CrawlAttemptState.SAVING, CrawlAttemptState.FAILED}
    ),
    CrawlAttemptState.SAVING: frozenset(
        {CrawlAttemptState.SUCCEEDED, CrawlAttemptState.FAILED}
    ),
    CrawlAttemptState.SUCCEEDED: frozenset(),
    CrawlAttemptState.FAILED: frozenset(),
}

LISTING_RUN_TRANSITIONS: dict[ListingRunState, frozenset[ListingRunState]] = {
    ListingRunState.DISCOVERED: frozenset(
        {ListingRunState.NORMALIZED, ListingRunState.FAILED}
    ),
    ListingRunState.NORMALIZED: frozenset(
        {ListingRunState.DEDUP_CHECKED, ListingRunState.FAILED}
    ),
    ListingRunState.DEDUP_CHECKED: frozenset(
        {ListingRunState.RULE_EVALUATED, ListingRunState.FAILED}
    ),
    ListingRunState.RULE_EVALUATED: frozenset(
        {
            ListingRunState.REJECTED,
            ListingRunState.RAG_RETRIEVED,
            ListingRunState.FAILED,
        }
    ),
    ListingRunState.REJECTED: frozenset(),
    ListingRunState.RAG_RETRIEVED: frozenset(
        {ListingRunState.AGENT_EVALUATED, ListingRunState.FAILED}
    ),
    ListingRunState.AGENT_EVALUATED: frozenset(
        {ListingRunState.NOTIFICATION_QUEUED, ListingRunState.FAILED}
    ),
    ListingRunState.NOTIFICATION_QUEUED: frozenset(
        {ListingRunState.NOTIFIED, ListingRunState.FAILED}
    ),
    ListingRunState.NOTIFIED: frozenset(),
    ListingRunState.FAILED: frozenset(),
}

NOTIFICATION_TRANSITIONS: dict[NotificationState, frozenset[NotificationState]] = {
    NotificationState.QUEUED: frozenset({NotificationState.SENDING}),
    NotificationState.SENDING: frozenset(
        {
            NotificationState.SENT,
            NotificationState.FAILED_KNOWN,
            NotificationState.VERIFY_REQUIRED,
        }
    ),
    NotificationState.FAILED_KNOWN: frozenset({NotificationState.QUEUED}),
    NotificationState.SENT: frozenset(),
    NotificationState.VERIFY_REQUIRED: frozenset(),
}


class TransitionError(ValueError):
    """Raised when a state machine is asked to move backward or skip a state."""


State = TypeVar("State", bound=StringEnum)


def assert_transition(
    current: State,
    target: State,
    transitions: dict[State, frozenset[State]],
) -> None:
    """Raise unless *target* is an explicit direct successor of *current*."""
    if target not in transitions.get(current, frozenset()):
        raise TransitionError(f"illegal transition: {current} -> {target}")


class DomainValue(BaseModel):
    """Base class for immutable, validated values shared across the plugin."""

    model_config = ConfigDict(frozen=True)


class WatchRule(DomainValue):
    id: str
    name: str
    include_keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    max_price_jpy: int | None = Field(default=None, ge=0)
    interval_seconds: int = Field(ge=1)
    target_session: str
    enabled: bool = True
    marketplace: str = DEFAULT_MARKETPLACE
    collector_identity: str = MOCK_COLLECTOR_IDENTITY
    decision_version: str = DEFAULT_DECISION_VERSION


def _canonical_rule_keywords(values: tuple[str, ...]) -> list[str]:
    return sorted(
        {
            " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
            for value in values
            if " ".join(unicodedata.normalize("NFKC", value).split())
        }
    )


def with_stable_rule_id(rule: WatchRule) -> WatchRule:
    """Return *rule* with a stable ID derived from material behavior."""
    material = {
        "collector_identity": rule.collector_identity.strip(),
        "decision_version": rule.decision_version.strip(),
        "exclude_keywords": _canonical_rule_keywords(
            rule.exclude_keywords
        ),
        "include_keywords": _canonical_rule_keywords(
            rule.include_keywords
        ),
        "marketplace": rule.marketplace.strip(),
        "max_price_jpy": rule.max_price_jpy,
        "target_session": rule.target_session.strip(),
    }
    payload = json.dumps(
        material,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return rule.model_copy(update={"id": f"{_RULE_ID_PREFIX}{digest[:32]}"})


class Listing(DomainValue):
    marketplace: str
    external_id: str
    title: str
    description: str = ""
    price_jpy: int = Field(ge=0)
    url: str
    image_url: str | None = None
    seller_name: str | None = None
    published_at: datetime | None = None
    discovered_at: datetime

    @field_validator("external_id", "title")
    @classmethod
    def require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class AgentDecision(DomainValue):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    score: int = Field(ge=0, le=100)
    recommendation: Literal["SKIP", "REVIEW", "HIGH_PRIORITY"]
    reasons: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    retrieved_evidence: tuple[str, ...] = ()
    model_name: str
    prompt_version: str
