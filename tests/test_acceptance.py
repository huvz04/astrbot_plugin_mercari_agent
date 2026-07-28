"""Offline production-core acceptance coverage for the plugin skeleton."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math

import pytest
from sqlalchemy import func, select

from astrbot_plugin_mercari_agent.domain import (
    AgentDecision,
    CrawlAttemptState,
    CrawlJobState,
    Listing,
    ListingRunState,
    NotificationState,
    WatchRule,
)
from astrbot_plugin_mercari_agent.graph import build_listing_graph
from astrbot_plugin_mercari_agent.monitor import CrawlService, MockCollector
from astrbot_plugin_mercari_agent.rag import Evidence, MarkdownChromaRetriever
from astrbot_plugin_mercari_agent.storage import (
    ListingProcessRunRow,
    ListingRow,
    NotificationRow,
    Repository,
)


class DeterministicEmbeddings:
    """Offline embedding collaborator that avoids importing AstrBot modules."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((byte / 255.0) * 2.0) - 1.0 for byte in digest]
        magnitude = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / magnitude for value in values]


class TimeoutThenMockCollector:
    """Fail once, then return the skeleton's production mock listing."""

    def __init__(self) -> None:
        self.calls = 0
        self._mock = MockCollector()

    async def collect(self, rule: WatchRule) -> list[Listing]:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("offline transient timeout")
        return await self._mock.collect(rule)


class EvidenceBackedEvaluator:
    async def evaluate(
        self, listing: Listing, evidence: list[Evidence]
    ) -> AgentDecision:
        assert evidence, "the acceptance path requires Chroma evidence"
        return AgentDecision(
            score=91,
            recommendation="HIGH_PRIORITY",
            reasons=("月村手毬 target matched",),
            risks=("verify condition before purchase",),
            retrieved_evidence=(evidence[0].document_id,),
            model_name="acceptance-deterministic-evaluator",
            prompt_version="acceptance-v1",
        )


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def send(self, target_session: str, text: str) -> bool:
        self.messages.append((target_session, text))
        return True


@dataclass
class AcceptanceApp:
    repository: Repository
    retriever: MarkdownChromaRetriever
    graph: object
    notifier: RecordingNotifier
    collector: TimeoutThenMockCollector
    crawl_service: CrawlService
    rule: WatchRule


@pytest.fixture
def app(tmp_path: Path) -> AcceptanceApp:
    repository = Repository.open(tmp_path / "acceptance.sqlite3")
    retriever = MarkdownChromaRetriever.build(
        knowledge_dir=Path(__file__).parents[1] / "knowledge",
        persist_dir=tmp_path / "chroma",
        embeddings=DeterministicEmbeddings(),
    )
    notifier = RecordingNotifier()
    graph = build_listing_graph(
        repository,
        retriever,
        EvidenceBackedEvaluator(),
        notifier,
    )
    collector = TimeoutThenMockCollector()

    async def no_sleep(_: float) -> None:
        return None

    rule = WatchRule(
        id="acceptance-temari",
        name="月村手毬 acceptance rule",
        include_keywords=("月村手毬",),
        max_price_jpy=3000,
        interval_seconds=60,
        target_session="aiocqhttp:group:acceptance",
    )
    crawl_service = CrawlService(
        repository,
        collector,
        graph,
        sleep=no_sleep,
        jitter=lambda: 0.0,
    )
    yield AcceptanceApp(
        repository=repository,
        retriever=retriever,
        graph=graph,
        notifier=notifier,
        collector=collector,
        crawl_service=crawl_service,
        rule=rule,
    )
    repository.dispose()


def _table_counts(repository: Repository) -> tuple[int, int, int]:
    with repository._sessions() as session:
        return (
            session.scalar(select(func.count()).select_from(ListingRow)) or 0,
            session.scalar(select(func.count()).select_from(ListingProcessRunRow))
            or 0,
            session.scalar(select(func.count()).select_from(NotificationRow)) or 0,
        )


def test_one_logical_job_retries_then_notifies_exactly_once(app: AcceptanceApp) -> None:
    job_id = asyncio.run(app.crawl_service.run_once(app.rule))

    assert app.repository.get_job(job_id).state is CrawlJobState.SUCCEEDED
    attempts = app.repository.attempts(job_id)
    assert [attempt.state for attempt in attempts] == [
        CrawlAttemptState.FAILED,
        CrawlAttemptState.SUCCEEDED,
    ]
    assert attempts[0].error_type == "timeout"
    assert attempts[1].item_count == 1
    assert _table_counts(app.repository) == (1, 1, 1)

    with app.repository._sessions() as session:
        process_run = session.scalar(select(ListingProcessRunRow))
        notification = session.scalar(select(NotificationRow))
    assert process_run is not None
    assert ListingRunState(process_run.state) is ListingRunState.NOTIFIED
    assert notification is not None
    assert notification.sent_at is not None
    notification_id = notification.id
    sent_at = notification.sent_at

    assert len(app.notifier.messages) == 1
    _, message = app.notifier.messages[0]
    for expected in (
        "月村手毬",
        "1200",
        "Recommendation: HIGH_PRIORITY",
        "Score: 91",
        "https://example.invalid/mock-mercari-001",
    ):
        assert expected in message
    assert "Evidence IDs: " in message
    assert "Evidence IDs: (none)" not in message
    assert any(evidence_id in message for evidence_id in ("aliases", "risk_terms"))

    listing = asyncio.run(MockCollector().collect(app.rule))[0]
    asyncio.run(app.graph.ainvoke({"listing": listing, "watch_rule": app.rule}))

    assert _table_counts(app.repository) == (1, 1, 1)
    assert len(app.notifier.messages) == 1

    repeated_notification = app.repository.get_notification(notification_id)
    assert repeated_notification.id == notification_id
    assert repeated_notification.state is NotificationState.SENT
    assert _table_counts(app.repository) == (1, 1, 1)
    assert repeated_notification.sent_at == sent_at
    assert len(app.notifier.messages) == 1
