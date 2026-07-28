"""SQLite persistence for forward-only Mercari crawl work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, create_engine, func, inspect, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import TypeDecorator

from .domain import (
    ATTEMPT_TRANSITIONS,
    JOB_TRANSITIONS,
    LISTING_RUN_TRANSITIONS,
    NOTIFICATION_TRANSITIONS,
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    ListingRunState,
    NotificationState,
    WatchRule,
    assert_transition,
)


class ConcurrentStateChange(RuntimeError):
    """Raised when a row changed after its expected state was read."""


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


class UtcDateTime(TypeDecorator[datetime]):
    """Persist UTC instants in SQLite and restore their timezone on read."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        normalized = _as_utc(value)
        return normalized.replace(tzinfo=None) if normalized is not None else None

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


_SENSITIVE_ERROR_TEXT = re.compile(
    r"(?:cookie|authorization|bearer|token|password|secret|set-cookie)", re.IGNORECASE
)
_UNFINISHED_ATTEMPT_STATES = (
    CrawlAttemptState.CREATED.value,
    CrawlAttemptState.REQUESTING.value,
    CrawlAttemptState.RECEIVED.value,
    CrawlAttemptState.PARSING.value,
    CrawlAttemptState.SAVING.value,
)
_UNFINISHED_ATTEMPT_INDEX = "uq_unfinished_attempt_per_job"
_UNFINISHED_ATTEMPT_WHERE = (
    "state IN ('CREATED', 'REQUESTING', 'RECEIVED', 'PARSING', 'SAVING')"
)


def _sanitize_error_summary(value: object) -> str | None:
    """Keep a concise diagnostic without retaining credentials or response bodies."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("error_summary must be a string")
    compact = " ".join(value.split())
    lowered = compact.lower()
    if _SENSITIVE_ERROR_TEXT.search(compact):
        return "sensitive error detail redacted"
    if (
        lowered.startswith(("{", "[", "<"))
        or "response body" in lowered
        or "<!doctype" in lowered
    ):
        return "response body omitted"
    if len(compact) > 240:
        return "error summary omitted"
    return compact or None


def sanitize_error_summary(value: object) -> str | None:
    """Public boundary for retaining safe operational diagnostics."""
    return _sanitize_error_summary(value)


def _sanitize_error_type(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("error_type must be a string")
    if _SENSITIVE_ERROR_TEXT.search(value):
        return "redacted"
    return "".join(value.split())[:80] or None


class CrawlJobRow(Base):
    __tablename__ = "crawl_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_id: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime())


class CrawlAttemptRow(Base):
    __tablename__ = "crawl_attempts"
    __table_args__ = (
        UniqueConstraint("job_id", "attempt_no", name="uq_attempt_number"),
        Index(
            _UNFINISHED_ATTEMPT_INDEX,
            "job_id",
            unique=True,
            sqlite_where=text(_UNFINISHED_ATTEMPT_WHERE),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("crawl_jobs.id"), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(String)
    error_summary: Mapped[str | None] = mapped_column(String)
    next_retry_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    item_count: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)


class ListingRow(Base):
    __tablename__ = "listings"
    __table_args__ = (UniqueConstraint("marketplace", "external_id", name="uq_listing_external"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    marketplace: Mapped[str] = mapped_column(String, nullable=False)
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    price_jpy: Mapped[int] = mapped_column(Integer, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    image_url: Mapped[str | None] = mapped_column(String)
    seller_name: Mapped[str | None] = mapped_column(String)
    published_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    discovered_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)


class ListingProcessRunRow(Base):
    __tablename__ = "listing_process_runs"
    __table_args__ = (
        UniqueConstraint(
            "listing_id",
            "watch_rule_id",
            "run_no",
            name="uq_listing_run_rule_number",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id"), nullable=False)
    watch_rule_id: Mapped[str] = mapped_column(String, nullable=False)
    run_no: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(String)
    rule_snapshot_json: Mapped[str | None] = mapped_column(String)
    origin_job_id: Mapped[int | None] = mapped_column(
        ForeignKey("crawl_jobs.id")
    )
    origin_attempt_id: Mapped[int | None] = mapped_column(
        ForeignKey("crawl_attempts.id")
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)


class NotificationRow(Base):
    __tablename__ = "notifications"
    __table_args__ = (
        UniqueConstraint(
            "listing_id", "watch_rule_id", "decision_version", name="uq_notification_idempotency"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id"), nullable=False)
    watch_rule_id: Mapped[str] = mapped_column(String, nullable=False)
    decision_version: Mapped[str] = mapped_column(String, nullable=False)
    process_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("listing_process_runs.id")
    )
    target_session: Mapped[str] = mapped_column(String, nullable=False)
    message_text: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    attempt_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    sent_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    last_error: Mapped[str | None] = mapped_column(String)


@dataclass(frozen=True)
class CrawlJob:
    id: int
    rule_id: str
    state: CrawlJobState


@dataclass(frozen=True)
class CrawlAttempt:
    id: int
    job_id: int
    attempt_no: int
    state: CrawlAttemptState
    http_status: int | None
    error_type: str | None
    error_summary: str | None
    next_retry_at: datetime | None
    item_count: int | None
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True)
class ActiveCrawlJob:
    """An active job together with its latest persisted retry deadline."""

    id: int
    rule_id: str
    state: CrawlJobState
    next_retry_at: datetime | None


@dataclass(frozen=True)
class ListingProcessRun:
    id: int
    listing_id: int
    watch_rule_id: str
    run_no: int
    state: ListingRunState
    error_summary: str | None
    rule_snapshot_json: str | None
    origin_job_id: int | None
    origin_attempt_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Notification:
    id: int
    listing_id: int
    watch_rule_id: str
    decision_version: str
    process_run_id: int | None
    target_session: str
    message_text: str
    state: NotificationState
    created_at: datetime
    attempt_at: datetime | None
    sent_at: datetime | None
    last_error: str | None


@dataclass(frozen=True)
class StatusSnapshot:
    job_state: CrawlJobState | None
    attempt_count: int
    notification_count: int


class Repository:
    """Small transaction boundary around the plugin's SQLite database."""

    _ATTEMPT_FIELDS = frozenset(
        {
            "http_status",
            "error_type",
            "error_summary",
            "next_retry_at",
            "item_count",
            "started_at",
            "finished_at",
        }
    )
    _ATTEMPT_TIMESTAMP_FIELDS = frozenset(
        {"next_retry_at", "started_at", "finished_at"}
    )

    def __init__(self, engine) -> None:
        self._engine = engine
        self._sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @classmethod
    def open(cls, path: str | Path) -> "Repository":
        database = str(Path(path).resolve())
        engine = create_engine(f"sqlite:///{database}")
        Base.metadata.create_all(engine)
        repository = cls(engine)
        repository._migrate_attempt_updated_at()
        repository._migrate_listing_run_sequence()
        repository._migrate_listing_work_metadata()
        repository._migrate_notification_outbox()
        repository._migrate_unfinished_attempt_index()
        return repository

    def _migrate_attempt_updated_at(self) -> None:
        """Add and conservatively backfill the Attempt recovery heartbeat."""
        with self._engine.begin() as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    text("PRAGMA table_info('crawl_attempts')")
                )
            }
            if "updated_at" not in columns:
                connection.execute(
                    text("ALTER TABLE crawl_attempts ADD COLUMN updated_at DATETIME")
                )
            connection.execute(
                text(
                    "UPDATE crawl_attempts "
                    "SET updated_at = CASE "
                    f"WHEN state IN ({','.join(repr(value) for value in _UNFINISHED_ATTEMPT_STATES)}) "
                    "THEN CURRENT_TIMESTAMP "
                    "ELSE COALESCE(finished_at, started_at, CURRENT_TIMESTAMP) END "
                    "WHERE updated_at IS NULL"
                )
            )

    def _migrate_listing_run_sequence(self) -> None:
        """Rebuild legacy two-column run uniqueness into retryable sequences."""
        with self._engine.begin() as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    text("PRAGMA table_info('listing_process_runs')")
                )
            }
            if {"run_no", "updated_at"} <= columns:
                return
            connection.execute(text("DROP TABLE IF EXISTS listing_process_runs_v2"))
            connection.execute(
                text(
                    """
                    CREATE TABLE listing_process_runs_v2 (
                        id INTEGER NOT NULL PRIMARY KEY,
                        listing_id INTEGER NOT NULL
                            REFERENCES listings (id),
                        watch_rule_id VARCHAR NOT NULL,
                        run_no INTEGER NOT NULL,
                        state VARCHAR NOT NULL,
                        error_summary VARCHAR,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        CONSTRAINT uq_listing_run_rule_number
                            UNIQUE (listing_id, watch_rule_id, run_no)
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO listing_process_runs_v2 (
                        id, listing_id, watch_rule_id, run_no, state,
                        error_summary, created_at, updated_at
                    )
                    SELECT
                        id, listing_id, watch_rule_id, 1, state,
                        error_summary, created_at, created_at
                    FROM listing_process_runs
                    """
                )
            )
            old_count = connection.scalar(
                text("SELECT COUNT(*) FROM listing_process_runs")
            )
            new_count = connection.scalar(
                text("SELECT COUNT(*) FROM listing_process_runs_v2")
            )
            if old_count != new_count:
                raise RuntimeError("listing process run migration count mismatch")
            connection.execute(text("DROP TABLE listing_process_runs"))
            connection.execute(
                text(
                    "ALTER TABLE listing_process_runs_v2 "
                    "RENAME TO listing_process_runs"
                )
            )

    def _migrate_listing_work_metadata(self) -> None:
        """Add immutable recovery metadata to existing listing runs."""
        with self._engine.begin() as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    text("PRAGMA table_info('listing_process_runs')")
                )
            }
            additions = {
                "rule_snapshot_json": "VARCHAR",
                "origin_job_id": "INTEGER REFERENCES crawl_jobs (id)",
                "origin_attempt_id": "INTEGER REFERENCES crawl_attempts (id)",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        text(
                            f"ALTER TABLE listing_process_runs "
                            f"ADD COLUMN {name} {definition}"
                        )
                    )

    def _migrate_notification_outbox(self) -> None:
        """Add conservative delivery state to legacy notification rows."""
        with self._engine.begin() as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    text("PRAGMA table_info('notifications')")
                )
            }
            additions = {
                "process_run_id": "INTEGER REFERENCES listing_process_runs (id)",
                "state": "VARCHAR",
                "attempt_at": "DATETIME",
                "last_error": "VARCHAR",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        text(
                            f"ALTER TABLE notifications "
                            f"ADD COLUMN {name} {definition}"
                        )
                    )
            connection.execute(
                text(
                    "UPDATE notifications "
                    "SET process_run_id = ("
                    "  SELECT listing_process_runs.id "
                    "  FROM listing_process_runs "
                    "  WHERE listing_process_runs.listing_id = notifications.listing_id "
                    "    AND listing_process_runs.watch_rule_id = notifications.watch_rule_id "
                    "  ORDER BY listing_process_runs.run_no DESC "
                    "  LIMIT 1"
                    ") "
                    "WHERE process_run_id IS NULL"
                )
            )
            connection.execute(
                text(
                    "UPDATE notifications "
                    "SET state = CASE "
                    "WHEN sent_at IS NOT NULL THEN 'SENT' "
                    "ELSE 'VERIFY_REQUIRED' END "
                    "WHERE state IS NULL"
                )
            )

    def _migrate_unfinished_attempt_index(self) -> None:
        """Repair old duplicate work before enforcing the partial uniqueness rule."""
        with self._sessions.begin() as session:
            duplicate_job_ids = session.scalars(
                select(CrawlAttemptRow.job_id)
                .where(CrawlAttemptRow.state.in_(_UNFINISHED_ATTEMPT_STATES))
                .group_by(CrawlAttemptRow.job_id)
                .having(func.count(CrawlAttemptRow.id) > 1)
            )
            for job_id in duplicate_job_ids:
                unfinished = list(
                    session.scalars(
                        select(CrawlAttemptRow)
                        .where(
                            CrawlAttemptRow.job_id == job_id,
                            CrawlAttemptRow.state.in_(_UNFINISHED_ATTEMPT_STATES),
                        )
                        .order_by(
                            CrawlAttemptRow.attempt_no.desc(),
                            CrawlAttemptRow.id.desc(),
                        )
                    )
                )
                for attempt in unfinished[1:]:
                    session.execute(
                        update(CrawlAttemptRow)
                        .where(
                            CrawlAttemptRow.id == attempt.id,
                            CrawlAttemptRow.state == attempt.state,
                        )
                        .values(
                            state=CrawlAttemptState.FAILED.value,
                            error_type="migration_recovered",
                            error_summary="migration closed duplicate unfinished attempt",
                            next_retry_at=None,
                            finished_at=_utc_now(),
                        )
                    )
            session.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {_UNFINISHED_ATTEMPT_INDEX} "
                    f"ON crawl_attempts (job_id) WHERE {_UNFINISHED_ATTEMPT_WHERE}"
                )
            )

    def create_job(self, rule_id: str) -> int:
        with self._sessions.begin() as session:
            row = CrawlJobRow(rule_id=rule_id, state=CrawlJobState.PENDING.value, created_at=_utc_now())
            session.add(row)
            session.flush()
            return row.id

    def activate_job(self, job_id: int) -> None:
        self._advance_job(job_id, CrawlJobState.ACTIVE, started_at=_utc_now())

    def finish_job(self, job_id: int, target: CrawlJobState) -> None:
        self._advance_job(job_id, target, finished_at=_utc_now())

    def _advance_job(self, job_id: int, target: CrawlJobState, **fields: object) -> None:
        with self._sessions.begin() as session:
            row = session.get(CrawlJobRow, job_id)
            if row is None:
                raise KeyError(f"job {job_id} does not exist")
            current = CrawlJobState(row.state)
            assert_transition(current, target, JOB_TRANSITIONS)
            updated = session.execute(
                update(CrawlJobRow)
                .where(CrawlJobRow.id == job_id, CrawlJobRow.state == current.value)
                .values(state=target.value, **fields)
            )
            if updated.rowcount != 1:
                raise ConcurrentStateChange(f"job {job_id} changed concurrently")

    def create_attempt(self, job_id: int) -> int:
        try:
            with self._sessions.begin() as session:
                job = session.get(CrawlJobRow, job_id)
                if job is None:
                    raise KeyError(f"job {job_id} does not exist")
                if CrawlJobState(job.state) is not CrawlJobState.ACTIVE:
                    raise ConcurrentStateChange(
                        f"job {job_id} is not active for a new attempt"
                    )
                unfinished = session.scalar(
                    select(CrawlAttemptRow.id).where(
                        CrawlAttemptRow.job_id == job_id,
                        CrawlAttemptRow.state.in_(_UNFINISHED_ATTEMPT_STATES),
                    )
                )
                if unfinished is not None:
                    raise ConcurrentStateChange(
                        f"job {job_id} already has an unfinished attempt"
                    )
                current_max = session.scalar(
                    select(func.max(CrawlAttemptRow.attempt_no)).where(CrawlAttemptRow.job_id == job_id)
                )
                row = CrawlAttemptRow(
                    job_id=job_id,
                    attempt_no=(current_max or 0) + 1,
                    state=CrawlAttemptState.CREATED.value,
                    updated_at=_utc_now(),
                )
                session.add(row)
                session.flush()
                return row.id
        except IntegrityError as exc:
            raise ConcurrentStateChange(f"attempt number for job {job_id} changed concurrently") from exc

    def advance_attempt(self, attempt_id: int, target: CrawlAttemptState, **fields: object) -> None:
        unknown_fields = set(fields) - self._ATTEMPT_FIELDS
        if unknown_fields:
            raise ValueError(f"unsupported attempt fields: {sorted(unknown_fields)}")
        if "error_type" in fields:
            fields["error_type"] = _sanitize_error_type(fields["error_type"])
        if "error_summary" in fields:
            fields["error_summary"] = _sanitize_error_summary(fields["error_summary"])
        for name in self._ATTEMPT_TIMESTAMP_FIELDS & fields.keys():
            fields[name] = _as_utc(fields[name])
        with self._sessions.begin() as session:
            row = session.get(CrawlAttemptRow, attempt_id)
            if row is None:
                raise KeyError(f"attempt {attempt_id} does not exist")
            current = CrawlAttemptState(row.state)
            assert_transition(current, target, ATTEMPT_TRANSITIONS)
            updated = session.execute(
                update(CrawlAttemptRow)
                .where(CrawlAttemptRow.id == attempt_id, CrawlAttemptRow.state == current.value)
                .values(state=target.value, updated_at=_utc_now(), **fields)
            )
            if updated.rowcount != 1:
                raise ConcurrentStateChange(f"attempt {attempt_id} changed concurrently")

    def save_listing(self, listing: Listing) -> tuple[int, bool]:
        try:
            with self._sessions.begin() as session:
                existing = session.scalar(
                    select(ListingRow).where(
                        ListingRow.marketplace == listing.marketplace,
                        ListingRow.external_id == listing.external_id,
                    )
                )
                if existing is not None:
                    return existing.id, False
                listing_values = listing.model_dump()
                listing_values["published_at"] = _as_utc(listing_values["published_at"])
                listing_values["discovered_at"] = _as_utc(listing_values["discovered_at"])
                row = ListingRow(**listing_values)
                session.add(row)
                session.flush()
                return row.id, True
        except IntegrityError:
            with self._sessions() as session:
                existing = session.scalar(
                    select(ListingRow).where(
                        ListingRow.marketplace == listing.marketplace,
                        ListingRow.external_id == listing.external_id,
                    )
                )
                if existing is None:
                    raise
                return existing.id, False

    def persist_listing_work(
        self,
        listing: Listing,
        rule: WatchRule,
        *,
        origin_job_id: int,
        origin_attempt_id: int,
        stale_after: timedelta = timedelta(minutes=5),
    ) -> tuple[int, bool]:
        """Atomically persist a Listing and its recoverable DISCOVERED run."""
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        try:
            with self._sessions.begin() as session:
                origin_job = session.get(CrawlJobRow, origin_job_id)
                origin_attempt = session.get(
                    CrawlAttemptRow,
                    origin_attempt_id,
                )
                if (
                    origin_job is None
                    or origin_attempt is None
                    or origin_attempt.job_id != origin_job_id
                    or origin_job.rule_id != rule.id
                    or CrawlJobState(origin_job.state)
                    is not CrawlJobState.ACTIVE
                    or CrawlAttemptState(origin_attempt.state)
                    is not CrawlAttemptState.SAVING
                ):
                    raise ConcurrentStateChange(
                        "listing work origin must be an ACTIVE Job "
                        "with its SAVING Attempt"
                    )
                listing_row = session.scalar(
                    select(ListingRow).where(
                        ListingRow.marketplace == listing.marketplace,
                        ListingRow.external_id == listing.external_id,
                    )
                )
                if listing_row is None:
                    listing_values = listing.model_dump()
                    listing_values["published_at"] = _as_utc(
                        listing_values["published_at"]
                    )
                    listing_values["discovered_at"] = _as_utc(
                        listing_values["discovered_at"]
                    )
                    listing_row = ListingRow(**listing_values)
                    session.add(listing_row)
                    session.flush()

                latest = session.scalar(
                    select(ListingProcessRunRow)
                    .where(
                        ListingProcessRunRow.listing_id == listing_row.id,
                        ListingProcessRunRow.watch_rule_id == rule.id,
                    )
                    .order_by(
                        ListingProcessRunRow.run_no.desc(),
                        ListingProcessRunRow.id.desc(),
                    )
                    .limit(1)
                )
                now = _utc_now()
                next_run_no = 1
                if latest is not None:
                    latest_state = ListingRunState(latest.state)
                    if latest_state in {
                        ListingRunState.NOTIFIED,
                        ListingRunState.REJECTED,
                    }:
                        return latest.id, False
                    if latest_state is not ListingRunState.FAILED:
                        if latest.updated_at > now - stale_after:
                            return latest.id, False
                        assert_transition(
                            latest_state,
                            ListingRunState.FAILED,
                            LISTING_RUN_TRANSITIONS,
                        )
                        latest.state = ListingRunState.FAILED.value
                        latest.error_summary = (
                            "stale process run recovered after restart"
                        )
                        latest.updated_at = now
                    next_run_no = latest.run_no + 1

                run = ListingProcessRunRow(
                    listing_id=listing_row.id,
                    watch_rule_id=rule.id,
                    run_no=next_run_no,
                    state=ListingRunState.DISCOVERED.value,
                    rule_snapshot_json=rule.model_dump_json(),
                    origin_job_id=origin_job_id,
                    origin_attempt_id=origin_attempt_id,
                    created_at=now,
                    updated_at=now,
                )
                session.add(run)
                session.flush()
                return run.id, True
        except (ConcurrentStateChange, IntegrityError):
            with self._sessions() as session:
                existing_listing_id = session.scalar(
                    select(ListingRow.id).where(
                        ListingRow.marketplace == listing.marketplace,
                        ListingRow.external_id == listing.external_id,
                    )
                )
                if existing_listing_id is None:
                    raise
                existing = session.scalar(
                    select(ListingProcessRunRow)
                    .where(
                        ListingProcessRunRow.listing_id == existing_listing_id,
                        ListingProcessRunRow.watch_rule_id == rule.id,
                    )
                    .order_by(
                        ListingProcessRunRow.run_no.desc(),
                        ListingProcessRunRow.id.desc(),
                    )
                    .limit(1)
                )
                if existing is None:
                    raise
                return existing.id, False

    def get_listing(self, listing_id: int) -> Listing:
        with self._sessions() as session:
            row = session.get(ListingRow, listing_id)
            if row is None:
                raise KeyError(f"listing {listing_id} does not exist")
            return Listing(
                marketplace=row.marketplace,
                external_id=row.external_id,
                title=row.title,
                description=row.description,
                price_jpy=row.price_jpy,
                url=row.url,
                image_url=row.image_url,
                seller_name=row.seller_name,
                published_at=row.published_at,
                discovered_at=row.discovered_at,
            )

    def pending_listing_work(self) -> list[ListingProcessRun]:
        """Return only safe, immutable DISCOVERED work in insertion order."""
        with self._sessions() as session:
            rows = session.scalars(
                select(ListingProcessRunRow)
                .where(
                    ListingProcessRunRow.state
                    == ListingRunState.DISCOVERED.value,
                    ListingProcessRunRow.rule_snapshot_json.is_not(None),
                )
                .order_by(ListingProcessRunRow.id)
            )
            return [self._listing_run_value(row) for row in rows]

    def create_listing_run(self, listing_id: int, watch_rule_id: str) -> int:
        run_id, _ = self.get_or_create_listing_run(
            listing_id,
            watch_rule_id,
        )
        return run_id

    def get_or_create_listing_run(
        self,
        listing_id: int,
        watch_rule_id: str,
        *,
        stale_after: timedelta = timedelta(minutes=5),
    ) -> tuple[int, bool]:
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        key_filter = (
            ListingProcessRunRow.listing_id == listing_id,
            ListingProcessRunRow.watch_rule_id == watch_rule_id,
        )
        try:
            with self._sessions.begin() as session:
                if session.get(ListingRow, listing_id) is None:
                    raise KeyError(f"listing {listing_id} does not exist")
                latest = session.scalar(
                    select(ListingProcessRunRow)
                    .where(*key_filter)
                    .order_by(
                        ListingProcessRunRow.run_no.desc(),
                        ListingProcessRunRow.id.desc(),
                    )
                    .limit(1)
                )
                now = _utc_now()
                next_run_no = 1
                if latest is not None:
                    latest_state = ListingRunState(latest.state)
                    if latest_state in {
                        ListingRunState.NOTIFIED,
                        ListingRunState.REJECTED,
                    }:
                        return latest.id, False
                    if latest_state is not ListingRunState.FAILED:
                        if latest.updated_at > now - stale_after:
                            return latest.id, False
                        assert_transition(
                            latest_state,
                            ListingRunState.FAILED,
                            LISTING_RUN_TRANSITIONS,
                        )
                        recovered = session.execute(
                            update(ListingProcessRunRow)
                            .where(
                                ListingProcessRunRow.id == latest.id,
                                ListingProcessRunRow.state == latest.state,
                            )
                            .values(
                                state=ListingRunState.FAILED.value,
                                error_summary=_sanitize_error_summary(
                                    "stale process run recovered after restart"
                                ),
                                updated_at=now,
                            )
                        )
                        if recovered.rowcount != 1:
                            raise ConcurrentStateChange(
                                f"listing process run {latest.id} changed concurrently"
                            )
                    next_run_no = latest.run_no + 1
                row = ListingProcessRunRow(
                    listing_id=listing_id,
                    watch_rule_id=watch_rule_id,
                    run_no=next_run_no,
                    state=ListingRunState.DISCOVERED.value,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                session.flush()
                return row.id, True
        except (ConcurrentStateChange, IntegrityError):
            with self._sessions() as session:
                existing = session.scalar(
                    select(ListingProcessRunRow)
                    .where(*key_filter)
                    .order_by(
                        ListingProcessRunRow.run_no.desc(),
                        ListingProcessRunRow.id.desc(),
                    )
                    .limit(1)
                )
                if existing is None:
                    raise
                return existing.id, False

    def advance_listing_run(
        self,
        run_id: int,
        target: ListingRunState,
        error_summary: object = None,
    ) -> None:
        sanitized_error = _sanitize_error_summary(error_summary)
        with self._sessions.begin() as session:
            row = session.get(ListingProcessRunRow, run_id)
            if row is None:
                raise KeyError(f"listing process run {run_id} does not exist")
            current = ListingRunState(row.state)
            assert_transition(current, target, LISTING_RUN_TRANSITIONS)
            updated = session.execute(
                update(ListingProcessRunRow)
                .where(
                    ListingProcessRunRow.id == run_id,
                    ListingProcessRunRow.state == current.value,
                )
                .values(
                    state=target.value,
                    error_summary=sanitized_error,
                    updated_at=_utc_now(),
                )
            )
            if updated.rowcount != 1:
                raise ConcurrentStateChange(
                    f"listing process run {run_id} changed concurrently"
                )

    def claim_discovered_listing_run(self, run_id: int) -> bool:
        """Atomically claim DISCOVERED work by advancing it to NORMALIZED."""
        with self._sessions.begin() as session:
            updated = session.execute(
                update(ListingProcessRunRow)
                .where(
                    ListingProcessRunRow.id == run_id,
                    ListingProcessRunRow.state
                    == ListingRunState.DISCOVERED.value,
                )
                .values(
                    state=ListingRunState.NORMALIZED.value,
                    updated_at=_utc_now(),
                )
            )
            return updated.rowcount == 1

    def get_listing_run(self, run_id: int) -> ListingProcessRun:
        with self._sessions() as session:
            row = session.get(ListingProcessRunRow, run_id)
            if row is None:
                raise KeyError(f"listing process run {run_id} does not exist")
            return self._listing_run_value(row)

    def queue_notification_for_run(
        self,
        run_id: int,
        *,
        decision_version: str,
        target_session: str,
        message_text: str,
    ) -> tuple[int, bool]:
        """Atomically queue/requeue an idempotent notification and its run."""
        try:
            with self._sessions.begin() as session:
                run = session.get(ListingProcessRunRow, run_id)
                if run is None:
                    raise KeyError(f"listing process run {run_id} does not exist")
                run_state = ListingRunState(run.state)
                if run_state is not ListingRunState.AGENT_EVALUATED:
                    raise ConcurrentStateChange(
                        f"listing process run {run_id} is not ready to queue"
                    )
                key_filter = (
                    NotificationRow.listing_id == run.listing_id,
                    NotificationRow.watch_rule_id == run.watch_rule_id,
                    NotificationRow.decision_version == decision_version,
                )
                existing = session.scalar(
                    select(NotificationRow).where(*key_filter)
                )
                now = _utc_now()
                if existing is None:
                    notification = NotificationRow(
                        listing_id=run.listing_id,
                        watch_rule_id=run.watch_rule_id,
                        decision_version=decision_version,
                        process_run_id=run.id,
                        target_session=target_session,
                        message_text=message_text,
                        state=NotificationState.QUEUED.value,
                        created_at=now,
                    )
                    session.add(notification)
                    session.flush()
                else:
                    notification = existing
                    notification_state = NotificationState(notification.state)
                    if notification_state not in {
                        NotificationState.FAILED_KNOWN,
                        NotificationState.QUEUED,
                    }:
                        diagnostic = (
                            "notification verification required"
                            if notification_state
                            in {
                                NotificationState.SENDING,
                                NotificationState.VERIFY_REQUIRED,
                            }
                            else "notification already dispatched"
                        )
                        failed = session.execute(
                            update(ListingProcessRunRow)
                            .where(
                                ListingProcessRunRow.id == run.id,
                                ListingProcessRunRow.state
                                == ListingRunState.AGENT_EVALUATED.value,
                            )
                            .values(
                                state=ListingRunState.FAILED.value,
                                error_summary=diagnostic,
                                updated_at=now,
                            )
                        )
                        if failed.rowcount != 1:
                            raise ConcurrentStateChange(
                                f"listing process run {run.id} changed concurrently"
                            )
                        return notification.id, False
                    if notification_state is NotificationState.FAILED_KNOWN:
                        assert_transition(
                            notification_state,
                            NotificationState.QUEUED,
                            NOTIFICATION_TRANSITIONS,
                        )
                    requeued = session.execute(
                        update(NotificationRow)
                        .where(
                            NotificationRow.id == notification.id,
                            NotificationRow.state == notification.state,
                        )
                        .values(
                            process_run_id=run.id,
                            target_session=target_session,
                            message_text=message_text,
                            state=NotificationState.QUEUED.value,
                            attempt_at=None,
                            sent_at=None,
                            last_error=None,
                        )
                    )
                    if requeued.rowcount != 1:
                        raise ConcurrentStateChange(
                            f"notification {notification.id} changed concurrently"
                        )
                queued = session.execute(
                    update(ListingProcessRunRow)
                    .where(
                        ListingProcessRunRow.id == run.id,
                        ListingProcessRunRow.state
                        == ListingRunState.AGENT_EVALUATED.value,
                    )
                    .values(
                        state=ListingRunState.NOTIFICATION_QUEUED.value,
                        error_summary=None,
                        updated_at=now,
                    )
                )
                if queued.rowcount != 1:
                    raise ConcurrentStateChange(
                        f"listing process run {run.id} changed concurrently"
                    )
                return notification.id, True
        except IntegrityError:
            with self._sessions() as session:
                run = session.get(ListingProcessRunRow, run_id)
                if run is None:
                    raise
                existing = session.scalar(
                    select(NotificationRow).where(
                        NotificationRow.listing_id == run.listing_id,
                        NotificationRow.watch_rule_id == run.watch_rule_id,
                        NotificationRow.decision_version == decision_version,
                    )
                )
                if existing is None:
                    raise
                return existing.id, False

    def claim_notification(self, notification_id: int) -> Notification | None:
        with self._sessions.begin() as session:
            row = session.get(NotificationRow, notification_id)
            if row is None:
                raise KeyError(f"notification {notification_id} does not exist")
            current = NotificationState(row.state)
            if current is not NotificationState.QUEUED:
                return None
            assert_transition(
                current,
                NotificationState.SENDING,
                NOTIFICATION_TRANSITIONS,
            )
            updated = session.execute(
                update(NotificationRow)
                .where(
                    NotificationRow.id == notification_id,
                    NotificationRow.state == NotificationState.QUEUED.value,
                )
                .values(
                    state=NotificationState.SENDING.value,
                    attempt_at=_utc_now(),
                    last_error=None,
                )
            )
            if updated.rowcount != 1:
                raise ConcurrentStateChange(
                    f"notification {notification_id} changed concurrently"
                )
        return self.get_notification(notification_id)

    def finalize_notification_sent(self, notification_id: int) -> None:
        self._finalize_notification(
            notification_id,
            NotificationState.SENT,
            ListingRunState.NOTIFIED,
            error=None,
        )

    def finalize_notification_known_failure(
        self,
        notification_id: int,
        error: object,
    ) -> None:
        self._finalize_notification(
            notification_id,
            NotificationState.FAILED_KNOWN,
            ListingRunState.FAILED,
            error=error,
        )

    def mark_notification_verify_required(
        self,
        notification_id: int,
        error: object,
    ) -> None:
        self._finalize_notification(
            notification_id,
            NotificationState.VERIFY_REQUIRED,
            ListingRunState.FAILED,
            error=error,
        )

    def _finalize_notification(
        self,
        notification_id: int,
        target: NotificationState,
        run_target: ListingRunState,
        *,
        error: object,
    ) -> None:
        sanitized_error = _sanitize_error_summary(error)
        with self._sessions.begin() as session:
            notification = session.get(NotificationRow, notification_id)
            if notification is None:
                raise KeyError(f"notification {notification_id} does not exist")
            current = NotificationState(notification.state)
            if current is target:
                return
            assert_transition(current, target, NOTIFICATION_TRANSITIONS)
            if notification.process_run_id is None:
                raise ConcurrentStateChange(
                    f"notification {notification_id} has no process run"
                )
            run = session.get(
                ListingProcessRunRow,
                notification.process_run_id,
            )
            if run is None:
                raise KeyError(
                    f"listing process run {notification.process_run_id} does not exist"
                )
            current_run = ListingRunState(run.state)
            assert_transition(current_run, run_target, LISTING_RUN_TRANSITIONS)
            now = _utc_now()
            notification_update = session.execute(
                update(NotificationRow)
                .where(
                    NotificationRow.id == notification.id,
                    NotificationRow.state == NotificationState.SENDING.value,
                )
                .values(
                    state=target.value,
                    sent_at=now if target is NotificationState.SENT else None,
                    last_error=sanitized_error,
                )
            )
            run_update = session.execute(
                update(ListingProcessRunRow)
                .where(
                    ListingProcessRunRow.id == run.id,
                    ListingProcessRunRow.state == current_run.value,
                )
                .values(
                    state=run_target.value,
                    error_summary=sanitized_error,
                    updated_at=now,
                )
            )
            if notification_update.rowcount != 1 or run_update.rowcount != 1:
                raise ConcurrentStateChange(
                    f"notification {notification_id} changed concurrently"
                )

    def reconcile_sending_notifications(self) -> None:
        with self._sessions() as session:
            ids = list(
                session.scalars(
                    select(NotificationRow.id).where(
                        NotificationRow.state
                        == NotificationState.SENDING.value
                    )
                )
            )
        for notification_id in ids:
            self.mark_notification_verify_required(
                notification_id,
                "dispatch result unknown after restart",
            )

    def queued_notifications(self) -> list[Notification]:
        with self._sessions() as session:
            rows = session.scalars(
                select(NotificationRow)
                .where(NotificationRow.state == NotificationState.QUEUED.value)
                .order_by(NotificationRow.id)
            )
            return [self._notification_value(row) for row in rows]

    def get_notification(self, notification_id: int) -> Notification:
        with self._sessions() as session:
            row = session.get(NotificationRow, notification_id)
            if row is None:
                raise KeyError(f"notification {notification_id} does not exist")
            return self._notification_value(row)

    def count_notifications(self) -> int:
        with self._sessions() as session:
            return (
                session.scalar(
                    select(func.count()).select_from(NotificationRow)
                )
                or 0
            )

    def get_attempt(self, attempt_id: int) -> CrawlAttempt:
        with self._sessions() as session:
            row = session.get(CrawlAttemptRow, attempt_id)
            if row is None:
                raise KeyError(f"attempt {attempt_id} does not exist")
            return self._attempt_value(row)

    def get_job(self, job_id: int) -> CrawlJob:
        with self._sessions() as session:
            row = session.get(CrawlJobRow, job_id)
            if row is None:
                raise KeyError(f"job {job_id} does not exist")
            return CrawlJob(
                id=row.id,
                rule_id=row.rule_id,
                state=CrawlJobState(row.state),
            )

    def count_dispatched_notifications(self, watch_rule_id: str) -> int:
        with self._sessions() as session:
            return (
                session.scalar(
                    select(func.count())
                    .select_from(NotificationRow)
                    .where(
                        NotificationRow.watch_rule_id == watch_rule_id,
                        NotificationRow.state == NotificationState.SENT.value,
                    )
                )
                or 0
            )

    def count_sent_notifications(self, watch_rule_id: str) -> int:
        """Compatibility alias; user-facing callers should say dispatched."""
        return self.count_dispatched_notifications(watch_rule_id)

    def attempts(self, job_id: int) -> list[CrawlAttempt]:
        with self._sessions() as session:
            rows = session.scalars(
                select(CrawlAttemptRow)
                .where(CrawlAttemptRow.job_id == job_id)
                .order_by(CrawlAttemptRow.attempt_no)
            )
            return [self._attempt_value(row) for row in rows]

    def latest_job(self) -> CrawlJob | None:
        with self._sessions() as session:
            row = session.scalar(select(CrawlJobRow).order_by(CrawlJobRow.id.desc()))
            if row is None:
                return None
            return CrawlJob(id=row.id, rule_id=row.rule_id, state=CrawlJobState(row.state))

    def active_jobs(self, rule_id: str | None = None) -> list[ActiveCrawlJob]:
        """Return unfinished jobs with the retry deadline of their latest attempt."""
        with self._sessions() as session:
            statement = (
                select(CrawlJobRow)
                .where(CrawlJobRow.state == CrawlJobState.ACTIVE.value)
                .order_by(CrawlJobRow.id)
            )
            if rule_id is not None:
                statement = statement.where(CrawlJobRow.rule_id == rule_id)
            jobs = session.scalars(statement)
            active_jobs: list[ActiveCrawlJob] = []
            for job in jobs:
                attempt = session.scalar(
                    select(CrawlAttemptRow)
                    .where(CrawlAttemptRow.job_id == job.id)
                    .order_by(CrawlAttemptRow.attempt_no.desc())
                    .limit(1)
                )
                active_jobs.append(
                    ActiveCrawlJob(
                        id=job.id,
                        rule_id=job.rule_id,
                        state=CrawlJobState(job.state),
                        next_retry_at=(attempt.next_retry_at if attempt else None),
                    )
                )
            return active_jobs

    def active_job_for_rule(self, rule_id: str) -> ActiveCrawlJob | None:
        jobs = self.active_jobs(rule_id)
        return jobs[-1] if jobs else None

    def table_names(self) -> list[str]:
        return inspect(self._engine).get_table_names()

    def dispose(self) -> None:
        self._engine.dispose()

    def get_status(self) -> StatusSnapshot:
        with self._sessions() as session:
            latest = session.scalar(select(CrawlJobRow).order_by(CrawlJobRow.id.desc()))
            return StatusSnapshot(
                job_state=CrawlJobState(latest.state) if latest else None,
                attempt_count=session.scalar(select(func.count()).select_from(CrawlAttemptRow)) or 0,
                notification_count=session.scalar(select(func.count()).select_from(NotificationRow)) or 0,
            )

    @staticmethod
    def _attempt_value(row: CrawlAttemptRow) -> CrawlAttempt:
        return CrawlAttempt(
            id=row.id,
            job_id=row.job_id,
            attempt_no=row.attempt_no,
            state=CrawlAttemptState(row.state),
            http_status=row.http_status,
            error_type=row.error_type,
            error_summary=row.error_summary,
            next_retry_at=row.next_retry_at,
            item_count=row.item_count,
            started_at=row.started_at,
            finished_at=row.finished_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _listing_run_value(row: ListingProcessRunRow) -> ListingProcessRun:
        return ListingProcessRun(
            id=row.id,
            listing_id=row.listing_id,
            watch_rule_id=row.watch_rule_id,
            run_no=row.run_no,
            state=ListingRunState(row.state),
            error_summary=row.error_summary,
            rule_snapshot_json=row.rule_snapshot_json,
            origin_job_id=row.origin_job_id,
            origin_attempt_id=row.origin_attempt_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _notification_value(row: NotificationRow) -> Notification:
        return Notification(
            id=row.id,
            listing_id=row.listing_id,
            watch_rule_id=row.watch_rule_id,
            decision_version=row.decision_version,
            process_run_id=row.process_run_id,
            target_session=row.target_session,
            message_text=row.message_text,
            state=NotificationState(row.state),
            created_at=row.created_at,
            attempt_at=row.attempt_at,
            sent_at=row.sent_at,
            last_error=row.last_error,
        )
