"""SQLite persistence for forward-only Mercari crawl work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, create_engine, func, inspect, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .domain import (
    ATTEMPT_TRANSITIONS,
    JOB_TRANSITIONS,
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    assert_transition,
)


class ConcurrentStateChange(RuntimeError):
    """Raised when a row changed after its expected state was read."""


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CrawlJobRow(Base):
    __tablename__ = "crawl_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_id: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


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
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    item_count: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


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
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ListingProcessRunRow(Base):
    __tablename__ = "listing_process_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id"), nullable=False)
    watch_rule_id: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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
                row = ListingRow(**listing.model_dump())
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
        )
