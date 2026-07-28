from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import threading

import pytest
from sqlalchemy import func, select

from astrbot_plugin_mercari_agent.domain import (
    AgentDecision,
    Listing,
    ListingRunState,
    WatchRule,
    with_stable_rule_id,
)
from astrbot_plugin_mercari_agent.filters import evaluate_rule, normalize_listing
from astrbot_plugin_mercari_agent.graph import build_listing_graph
from astrbot_plugin_mercari_agent.rag import Evidence, MarkdownChromaRetriever
from astrbot_plugin_mercari_agent.storage import (
    ListingProcessRunRow,
    NotificationRow,
    Repository,
)


class DeterministicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [
            float(len(text)),
            float(sum(ord(character) for character in text) % 997),
            float(text.count("未開封")),
        ]


class FingerprintedEmbeddings:
    def __init__(self, identity: str, dimension: int) -> None:
        self.embedding_identity = identity
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        seed = float((sum(ord(character) for character in text) % 97) + 1)
        return [seed + offset for offset in range(self.dimension)]


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[Evidence]:
        self.queries.append(query)
        return [Evidence(document_id="risk-terms", text="未開封 means unopened")]


class ThreadRecordingRetriever(RecordingRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.thread_ids: list[int] = []

    def retrieve(self, query: str) -> list[Evidence]:
        self.thread_ids.append(threading.get_ident())
        return super().retrieve(query)


class FixedEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[Listing, list[Evidence]]] = []

    async def evaluate(
        self, listing: Listing, evidence: list[Evidence]
    ) -> AgentDecision:
        self.calls.append((listing, evidence))
        return AgentDecision(
            score=88,
            recommendation="HIGH_PRIORITY",
            reasons=("wanted character and item type",),
            risks=("verify condition",),
            retrieved_evidence=tuple(item.document_id for item in evidence),
            model_name="offline-fixed-evaluator",
            prompt_version="decision-v1",
        )


class HallucinatingEvaluator(FixedEvaluator):
    async def evaluate(
        self, listing: Listing, evidence: list[Evidence]
    ) -> AgentDecision:
        decision = await super().evaluate(listing, evidence)
        return decision.model_copy(
            update={"retrieved_evidence": ("not-returned-by-retriever",)}
        )


class RecordingNotifier:
    def __init__(self, succeeds: bool = True) -> None:
        self.succeeds = succeeds
        self.calls: list[tuple[str, str]] = []

    async def send(self, target_session: str, text: str) -> bool:
        self.calls.append((target_session, text))
        return self.succeeds


@pytest.fixture
def repository(tmp_path: Path):
    repo = Repository.open(tmp_path / "mercari.sqlite3")
    yield repo
    repo.dispose()


@pytest.fixture
def listing() -> Listing:
    return Listing(
        marketplace="mercari",
        external_id="m-graph-1",
        title="月村手毬 缶バッジ",
        description="未開封",
        price_jpy=1200,
        url="https://example.invalid/item/m-graph-1",
        discovered_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def rule() -> WatchRule:
    return WatchRule(
        id="rule-1",
        name="手毬 goods",
        include_keywords=("缶バッジ",),
        exclude_keywords=("ジャンク", "欠品"),
        max_price_jpy=2000,
        interval_seconds=60,
        target_session="aiocqhttp:group:123",
        decision_version="decision-v1",
    )


def test_async_graph_runs_synchronous_retrieval_off_event_loop_thread(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    retriever = ThreadRecordingRetriever()
    graph = build_listing_graph(
        repository,
        retriever,
        FixedEvaluator(),
        RecordingNotifier(),
    )

    async def exercise() -> None:
        event_loop_thread = threading.get_ident()
        await graph.ainvoke({"listing": listing, "watch_rule": rule})
        assert retriever.thread_ids
        assert retriever.thread_ids[0] != event_loop_thread

    asyncio.run(exercise())


def _run_count(repository: Repository) -> int:
    with repository._sessions() as session:
        return (
            session.scalar(
                select(func.count()).select_from(ListingProcessRunRow)
            )
            or 0
        )


def _notification_sent_at(repository: Repository):
    with repository._sessions() as session:
        row = session.scalar(select(NotificationRow))
        assert row is not None
        return row.sent_at


def test_exclusion_wins_over_include_keyword(
    listing: Listing, rule: WatchRule
) -> None:
    copied = listing.model_copy(
        update={"title": "月村手毬 缶バッジ ジャンク"}
    )

    result = evaluate_rule(copied, rule)

    assert result.accepted is False
    assert result.reason == "excluded_keyword:ジャンク"


@pytest.mark.parametrize(
    ("rule_update", "listing_update", "expected"),
    [
        ({"include_keywords": (" ", "\t")}, {}, "blank_include_keywords"),
        ({"include_keywords": ("アクスタ",)}, {}, "missing_include_keyword"),
        ({}, {"price_jpy": 2001}, "price_above_maximum"),
        ({}, {}, "accepted"),
    ],
)
def test_filter_outcomes_follow_the_required_order(
    listing: Listing,
    rule: WatchRule,
    rule_update: dict[str, object],
    listing_update: dict[str, object],
    expected: str,
) -> None:
    result = evaluate_rule(
        listing.model_copy(update=listing_update),
        rule.model_copy(update=rule_update),
    )

    assert result.reason == expected
    assert result.accepted is (expected == "accepted")


def test_normalize_listing_applies_nfkc_and_collapses_whitespace_without_mutation(
    listing: Listing,
) -> None:
    raw = listing.model_copy(
        update={
            "title": "  月村手毬　ＡＢＣ\t缶バッジ  ",
            "description": " 未開封\n\n　美品 ",
        }
    )

    normalized = normalize_listing(raw)

    assert normalized.title == "月村手毬 ABC 缶バッジ"
    assert normalized.description == "未開封 美品"
    assert raw.title == "  月村手毬　ＡＢＣ\t缶バッジ  "
    assert raw.description == " 未開封\n\n　美品 "
    assert normalized is not raw


def test_markdown_retriever_returns_stable_source_metadata(
    tmp_path: Path,
) -> None:
    knowledge_dir = Path(__file__).parents[1] / "knowledge"
    retriever = MarkdownChromaRetriever.build(
        knowledge_dir=knowledge_dir,
        persist_dir=tmp_path / "chroma",
        embeddings=DeterministicEmbeddings(),
    )

    evidence = retriever.retrieve("未開封の商品")

    assert evidence
    assert evidence[0].document_id
    assert evidence[0].document_id in {"aliases", "risk_terms"}


def test_markdown_retriever_reopens_without_duplicates_and_isolates_embeddings(
    tmp_path: Path,
) -> None:
    knowledge_dir = Path(__file__).parents[1] / "knowledge"
    persist_dir = tmp_path / "persistent-chroma"

    first = MarkdownChromaRetriever.build(
        knowledge_dir,
        persist_dir,
        FingerprintedEmbeddings("provider-a", 3),
    )
    first_name = first.vector_store._collection.name
    first_ids = sorted(first.vector_store.get()["ids"])
    first_count = first.vector_store._collection.count()

    reopened = MarkdownChromaRetriever.build(
        knowledge_dir,
        persist_dir,
        FingerprintedEmbeddings("provider-a", 3),
    )
    changed_identity = MarkdownChromaRetriever.build(
        knowledge_dir,
        persist_dir,
        FingerprintedEmbeddings("provider-b", 3),
    )
    changed_dimension = MarkdownChromaRetriever.build(
        knowledge_dir,
        persist_dir,
        FingerprintedEmbeddings("provider-a", 4),
    )

    assert reopened.vector_store._collection.name == first_name
    assert reopened.vector_store._collection.count() == first_count
    assert sorted(reopened.vector_store.get()["ids"]) == first_ids
    assert len(first_ids) == len(set(first_ids))
    assert all(len(chunk_id) == 64 for chunk_id in first_ids)
    assert changed_identity.vector_store._collection.name != first_name
    assert changed_dimension.vector_store._collection.name != first_name
    assert (
        changed_dimension.vector_store._collection.name
        != changed_identity.vector_store._collection.name
    )


def test_knowledge_files_define_all_required_terms() -> None:
    knowledge_dir = Path(__file__).parents[1] / "knowledge"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(knowledge_dir.glob("*.md"))
    )

    for term in ("缶バッジ", "アクスタ", "未開封", "ジャンク", "欠品", "バラ売り不可"):
        assert term in text


def test_accepted_listing_reaches_notified_and_builds_auditable_message(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    retriever = RecordingRetriever()
    evaluator = FixedEvaluator()
    notifier = RecordingNotifier()
    graph = build_listing_graph(repository, retriever, evaluator, notifier)

    result = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": rule})
    )

    run = repository.get_listing_run(result["process_run_id"])
    assert run.state is ListingRunState.NOTIFIED
    assert result["agent_decision"].score == 88
    assert len(retriever.queries) == 1
    assert len(evaluator.calls) == 1
    assert len(notifier.calls) == 1
    target_session, message = notifier.calls[0]
    assert target_session == rule.target_session
    for expected in (
        listing.title,
        "JPY 1200",
        listing.url,
        "88",
        "HIGH_PRIORITY",
        "wanted character and item type",
        "verify condition",
        "risk-terms",
    ):
        assert expected in message
    assert _notification_sent_at(repository) is not None


def test_invoking_graph_twice_creates_one_run_notification_and_send(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    retriever = RecordingRetriever()
    evaluator = FixedEvaluator()
    notifier = RecordingNotifier()
    graph = build_listing_graph(repository, retriever, evaluator, notifier)

    first = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": rule})
    )
    second = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": rule})
    )

    assert first["listing_created"] is True
    assert second["listing_created"] is False
    assert first["process_run_created"] is True
    assert second["process_run_created"] is False
    assert repository.count_notifications() == 1
    assert _run_count(repository) == 1
    assert len(notifier.calls) == 1


def test_graph_processes_a_precreated_discovered_run_without_duplication(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    listing_id, _ = repository.save_listing(listing)
    run_id, created = repository.get_or_create_listing_run(
        listing_id,
        rule.id,
    )
    notifier = RecordingNotifier()
    graph = build_listing_graph(
        repository,
        RecordingRetriever(),
        FixedEvaluator(),
        notifier,
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "listing": listing,
                "watch_rule": rule,
                "process_run_id": run_id,
            }
        )
    )

    assert created is True
    assert result["process_run_id"] == run_id
    assert repository.get_listing_run(run_id).state is ListingRunState.NOTIFIED
    assert _run_count(repository) == 1
    assert len(notifier.calls) == 1


def test_same_listing_under_different_rules_creates_distinct_runs_and_notifications(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    retriever = RecordingRetriever()
    evaluator = FixedEvaluator()
    notifier = RecordingNotifier()
    graph = build_listing_graph(repository, retriever, evaluator, notifier)
    second_rule = rule.model_copy(
        update={
            "id": "rule-2",
            "target_session": "aiocqhttp:group:456",
        }
    )

    first = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": rule})
    )
    second = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": second_rule})
    )

    assert first["process_run_id"] != second["process_run_id"]
    assert first["process_run_created"] is True
    assert second["process_run_created"] is True
    assert repository.count_notifications() == 2
    assert _run_count(repository) == 2
    assert len(notifier.calls) == 2
    assert [target for target, _ in notifier.calls] == [
        rule.target_session,
        second_rule.target_session,
    ]


def test_relaxed_material_rule_version_reevaluates_a_rejected_listing(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    low_rule = with_stable_rule_id(
        rule.model_copy(update={"max_price_jpy": listing.price_jpy - 1})
    )
    relaxed_rule = with_stable_rule_id(
        rule.model_copy(update={"max_price_jpy": listing.price_jpy})
    )
    notifier = RecordingNotifier()
    graph = build_listing_graph(
        repository,
        RecordingRetriever(),
        FixedEvaluator(),
        notifier,
    )

    rejected = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": low_rule})
    )
    accepted = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": relaxed_rule})
    )

    assert low_rule.id != relaxed_rule.id
    assert repository.get_listing_run(
        rejected["process_run_id"]
    ).state is ListingRunState.REJECTED
    assert repository.get_listing_run(
        accepted["process_run_id"]
    ).state is ListingRunState.NOTIFIED
    assert _run_count(repository) == 2
    assert repository.count_notifications() == 1
    assert len(notifier.calls) == 1


def test_pre_saved_listing_without_a_run_is_still_processed(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    listing_id, listing_created = repository.save_listing(listing)
    notifier = RecordingNotifier()
    graph = build_listing_graph(
        repository,
        RecordingRetriever(),
        FixedEvaluator(),
        notifier,
    )

    result = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": rule})
    )

    assert listing_created is True
    assert result["listing_id"] == listing_id
    assert result["listing_created"] is False
    assert result["process_run_created"] is True
    assert (
        repository.get_listing_run(result["process_run_id"]).state
        is ListingRunState.NOTIFIED
    )
    assert repository.count_notifications() == 1
    assert _run_count(repository) == 1
    assert len(notifier.calls) == 1


def test_rejected_listing_skips_semantic_and_notification_collaborators(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    retriever = RecordingRetriever()
    evaluator = FixedEvaluator()
    notifier = RecordingNotifier()
    graph = build_listing_graph(repository, retriever, evaluator, notifier)
    rejected = listing.model_copy(
        update={
            "external_id": "m-rejected",
            "title": "月村手毬 缶バッジ ジャンク",
        }
    )

    result = asyncio.run(
        graph.ainvoke({"listing": rejected, "watch_rule": rule})
    )

    run = repository.get_listing_run(result["process_run_id"])
    assert run.state is ListingRunState.REJECTED
    assert result["filter_result"].reason == "excluded_keyword:ジャンク"
    assert retriever.queries == []
    assert evaluator.calls == []
    assert notifier.calls == []
    assert repository.count_notifications() == 0


def test_unsuccessful_notifier_fails_run_and_keeps_notification_unsent(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    notifier = RecordingNotifier(succeeds=False)
    graph = build_listing_graph(
        repository,
        RecordingRetriever(),
        FixedEvaluator(),
        notifier,
    )

    result = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": rule})
    )

    run = repository.get_listing_run(result["process_run_id"])
    assert run.state is ListingRunState.FAILED
    assert repository.count_notifications() == 1
    assert _notification_sent_at(repository) is None
    assert len(notifier.calls) == 1
    assert result["errors"] == ["notification send returned false"]


def test_process_run_error_summaries_are_sanitized(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    listing_id, _ = repository.save_listing(listing)
    run_id = repository.create_listing_run(listing_id, rule.id)

    repository.advance_listing_run(
        run_id,
        ListingRunState.FAILED,
        error_summary="Authorization: Bearer private-token",
    )

    assert (
        repository.get_listing_run(run_id).error_summary
        == "sensitive error detail redacted"
    )


def test_hallucinated_evidence_id_fails_without_queueing_or_dispatch(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    notifier = RecordingNotifier()
    graph = build_listing_graph(
        repository,
        RecordingRetriever(),
        HallucinatingEvaluator(),
        notifier,
    )

    result = asyncio.run(
        graph.ainvoke({"listing": listing, "watch_rule": rule})
    )

    run = repository.get_listing_run(result["process_run_id"])
    assert run.state is ListingRunState.FAILED
    assert run.error_summary == "agent cited unavailable evidence"
    assert repository.count_notifications() == 0
    assert notifier.calls == []
    assert result["errors"] == ["agent cited unavailable evidence"]


def test_decision_version_mismatch_fails_without_notification(
    repository: Repository,
    listing: Listing,
    rule: WatchRule,
) -> None:
    mismatched_rule = rule.model_copy(
        update={"decision_version": "decision-v2"}
    )
    notifier = RecordingNotifier()
    graph = build_listing_graph(
        repository,
        RecordingRetriever(),
        FixedEvaluator(),
        notifier,
    )

    result = asyncio.run(
        graph.ainvoke(
            {"listing": listing, "watch_rule": mismatched_rule}
        )
    )

    run = repository.get_listing_run(result["process_run_id"])
    assert run.state is ListingRunState.FAILED
    assert run.error_summary == "agent decision version does not match rule"
    assert repository.count_notifications() == 0
    assert notifier.calls == []
