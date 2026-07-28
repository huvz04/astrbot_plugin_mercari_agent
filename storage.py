"""SQLite persistence for forward-only Mercari crawl work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, create_engine, func, inspect, select, update
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
    __table_args__ = (UniqueConstraint("job_id", "attempt_no", name="uq_attempt_number"),)

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
            name="uq_listing_run_rule",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id"), nullable=False)
    watch_rule_id: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)


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
    state: ListingRunState
    error_summary: str | None


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
        return cls(engine)

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
                if session.get(CrawlJobRow, job_id) is None:
                    raise KeyError(f"job {job_id} does not exist")
                current_max = session.scalar(
                    select(func.max(CrawlAttemptRow.attempt_no)).where(CrawlAttemptRow.job_id == job_id)
                )
                row = CrawlAttemptRow(
                    job_id=job_id,
                    attempt_no=(current_max or 0) + 1,
                    state=CrawlAttemptState.CREATED.value,
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
                .values(state=target.value, **fields)
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
    ) -> tuple[int, bool]:
        key_filter = (
            ListingProcessRunRow.listing_id == listing_id,
            ListingProcessRunRow.watch_rule_id == watch_rule_id,
        )
        try:
            with self._sessions.begin() as session:
                if session.get(ListingRow, listing_id) is None:
                    raise KeyError(f"listing {listing_id} does not exist")
                existing = session.scalar(
                    select(ListingProcessRunRow).where(*key_filter)
                )
                if existing is not None:
                    return existing.id, False
                row = ListingProcessRunRow(
                    listing_id=listing_id,
                    watch_rule_id=watch_rule_id,
                    state=ListingRunState.DISCOVERED.value,
                    created_at=_utc_now(),
                )
                session.add(row)
                session.flush()
                return row.id, True
        except IntegrityError:
            with self._sessions() as session:
                existing = session.scalar(
                    select(ListingProcessRunRow).where(*key_filter)
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
                state=ListingRunState(row.state),
                error_summary=row.error_summary,
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

    def active_jobs(self) -> list[ActiveCrawlJob]:
        """Return unfinished jobs with the retry deadline of their latest attempt."""
        with self._sessions() as session:
            jobs = session.scalars(
                select(CrawlJobRow)
                .where(CrawlJobRow.state == CrawlJobState.ACTIVE.value)
                .order_by(CrawlJobRow.id)
            )
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
        )
