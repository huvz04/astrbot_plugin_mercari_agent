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
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    ListingRunState,
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
    target_session: Mapped[str] = mapped_column(String, nullable=False)
    message_text: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(UtcDateTime())


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
    created_at: datetime
    updated_at: datetime


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

    def get_listing_run(self, run_id: int) -> ListingProcessRun:
        with self._sessions() as session:
            row = session.get(ListingProcessRunRow, run_id)
            if row is None:
                raise KeyError(f"listing process run {run_id} does not exist")
            return ListingProcessRun(
                id=row.id,
                listing_id=row.listing_id,
                watch_rule_id=row.watch_rule_id,
                run_no=row.run_no,
                state=ListingRunState(row.state),
                error_summary=row.error_summary,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    def queue_notification(
        self,
        listing_id: int,
        watch_rule_id: str,
        decision_version: str,
        target_session: str,
        message_text: str,
    ) -> tuple[int, bool]:
        key_filter = (
            NotificationRow.listing_id == listing_id,
            NotificationRow.watch_rule_id == watch_rule_id,
            NotificationRow.decision_version == decision_version,
        )
        try:
            with self._sessions.begin() as session:
                existing = session.scalar(
                    select(NotificationRow).where(*key_filter)
                )
                if existing is not None:
                    return existing.id, False
                row = NotificationRow(
                    listing_id=listing_id,
                    watch_rule_id=watch_rule_id,
                    decision_version=decision_version,
                    target_session=target_session,
                    message_text=message_text,
                    created_at=_utc_now(),
                )
                session.add(row)
                session.flush()
                return row.id, True
        except IntegrityError:
            with self._sessions() as session:
                existing = session.scalar(
                    select(NotificationRow).where(*key_filter)
                )
                if existing is None:
                    raise
                return existing.id, False

    def mark_notification_sent(self, notification_id: int) -> None:
        with self._sessions.begin() as session:
            row = session.get(NotificationRow, notification_id)
            if row is None:
                raise KeyError(f"notification {notification_id} does not exist")
            if row.sent_at is not None:
                return
            updated = session.execute(
                update(NotificationRow)
                .where(
                    NotificationRow.id == notification_id,
                    NotificationRow.sent_at.is_(None),
                )
                .values(sent_at=_utc_now())
            )
            if updated.rowcount != 1:
                raise ConcurrentStateChange(
                    f"notification {notification_id} changed concurrently"
                )

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

    def count_sent_notifications(self, watch_rule_id: str) -> int:
        with self._sessions() as session:
            return (
                session.scalar(
                    select(func.count())
                    .select_from(NotificationRow)
                    .where(
                        NotificationRow.watch_rule_id == watch_rule_id,
                        NotificationRow.sent_at.is_not(None),
                    )
                )
                or 0
            )

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
