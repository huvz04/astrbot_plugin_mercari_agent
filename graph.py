"""LangGraph orchestration for deterministic listing decisions."""

from __future__ import annotations

from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .domain import AgentDecision, Listing, ListingRunState, WatchRule
from .filters import FilterResult, evaluate_rule, normalize_listing
from .rag import Evidence
from .storage import Repository


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[Evidence]: ...


class Evaluator(Protocol):
    async def evaluate(
        self, listing: Listing, evidence: list[Evidence]
    ) -> AgentDecision: ...


class Notifier(Protocol):
    async def send(self, target_session: str, text: str) -> bool: ...


class ListingGraphState(TypedDict, total=False):
    watch_rule: WatchRule
    raw_listing: Listing
    listing: Listing
    listing_id: int
    listing_created: bool
    process_run_id: int
    process_run_created: bool
    filter_result: FilterResult
    retrieved_documents: list[Evidence]
    agent_decision: AgentDecision
    notification_id: int
    notification_created: bool
    errors: list[str]


def _notification_message(
    listing: Listing,
    decision: AgentDecision,
    evidence: list[Evidence],
) -> str:
    evidence_ids = tuple(
        dict.fromkeys(
            [
                *(item.document_id for item in evidence),
                *decision.retrieved_evidence,
            ]
        )
    )
    return "\n".join(
        (
            f"Title: {listing.title}",
            f"Price: JPY {listing.price_jpy}",
            f"URL: {listing.url}",
            f"Score: {decision.score}",
            f"Recommendation: {decision.recommendation}",
            f"Reasons: {', '.join(decision.reasons) or '(none)'}",
            f"Risks: {', '.join(decision.risks) or '(none)'}",
            f"Evidence IDs: {', '.join(evidence_ids) or '(none)'}",
        )
    )


def build_listing_graph(
    repository: Repository,
    retriever: Retriever,
    evaluator: Evaluator,
    notifier: Notifier,
):
    def fail_process_run(
        state: ListingGraphState, error: Exception
    ) -> ListingGraphState:
        repository.advance_listing_run(
            state["process_run_id"],
            ListingRunState.FAILED,
            error_summary=str(error),
        )
        return {"errors": [*state.get("errors", ()), str(error)]}

    def normalize_node(state: ListingGraphState) -> ListingGraphState:
        raw_listing = state.get("raw_listing", state["listing"])
        return {
            "raw_listing": raw_listing,
            "listing": normalize_listing(raw_listing),
            "errors": list(state.get("errors", ())),
        }

    def deduplicate_node(state: ListingGraphState) -> ListingGraphState:
        listing_id, listing_created = repository.save_listing(state["listing"])
        result: ListingGraphState = {
            "listing_id": listing_id,
            "listing_created": listing_created,
        }
        run_id, run_created = repository.get_or_create_listing_run(
            listing_id,
            state["watch_rule"].id,
        )
        result["process_run_id"] = run_id
        result["process_run_created"] = run_created
        if run_created:
            repository.advance_listing_run(
                run_id,
                ListingRunState.NORMALIZED,
            )
            repository.advance_listing_run(
                run_id,
                ListingRunState.DEDUP_CHECKED,
            )
        return result

    def hard_filter_node(state: ListingGraphState) -> ListingGraphState:
        result = evaluate_rule(state["listing"], state["watch_rule"])
        repository.advance_listing_run(
            state["process_run_id"],
            ListingRunState.RULE_EVALUATED,
        )
        if not result.accepted:
            repository.advance_listing_run(
                state["process_run_id"],
                ListingRunState.REJECTED,
            )
        return {"filter_result": result}

    def retrieve_node(state: ListingGraphState) -> ListingGraphState:
        listing = state["listing"]
        try:
            documents = retriever.retrieve(
                f"{listing.title}\n{listing.description}"
            )
        except Exception as error:
            return fail_process_run(state, error)
        repository.advance_listing_run(
            state["process_run_id"],
            ListingRunState.RAG_RETRIEVED,
        )
        return {"retrieved_documents": documents}

    async def evaluate_node(
        state: ListingGraphState,
    ) -> ListingGraphState:
        try:
            decision = await evaluator.evaluate(
                state["listing"],
                state["retrieved_documents"],
            )
        except Exception as error:
            return fail_process_run(state, error)
        repository.advance_listing_run(
            state["process_run_id"],
            ListingRunState.AGENT_EVALUATED,
        )
        return {"agent_decision": decision}

    def queue_notification_node(
        state: ListingGraphState,
    ) -> ListingGraphState:
        message = _notification_message(
            state["listing"],
            state["agent_decision"],
            state["retrieved_documents"],
        )
        notification_id, created = repository.queue_notification(
            state["listing_id"],
            state["watch_rule"].id,
            state["agent_decision"].prompt_version,
            state["watch_rule"].target_session,
            message,
        )
        if created:
            repository.advance_listing_run(
                state["process_run_id"],
                ListingRunState.NOTIFICATION_QUEUED,
            )
        return {
            "notification_id": notification_id,
            "notification_created": created,
        }

    async def notify_node(state: ListingGraphState) -> ListingGraphState:
        try:
            sent = await notifier.send(
                state["watch_rule"].target_session,
                _notification_message(
                    state["listing"],
                    state["agent_decision"],
                    state["retrieved_documents"],
                ),
            )
        except Exception as error:
            return fail_process_run(state, error)
        if not sent:
            error = "notification send returned false"
            repository.advance_listing_run(
                state["process_run_id"],
                ListingRunState.FAILED,
                error_summary=error,
            )
            return {"errors": [*state.get("errors", ()), error]}
        repository.mark_notification_sent(state["notification_id"])
        repository.advance_listing_run(
            state["process_run_id"],
            ListingRunState.NOTIFIED,
        )
        return {}

    builder = StateGraph(ListingGraphState)
    builder.add_node("normalize", normalize_node)
    builder.add_node("deduplicate", deduplicate_node)
    builder.add_node("hard_filter", hard_filter_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("queue_notification", queue_notification_node)
    builder.add_node("notify", notify_node)

    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "deduplicate")
    builder.add_conditional_edges(
        "deduplicate",
        lambda state: (
            "new" if state["process_run_created"] else "duplicate"
        ),
        {"new": "hard_filter", "duplicate": END},
    )
    builder.add_conditional_edges(
        "hard_filter",
        lambda state: (
            "accepted" if state["filter_result"].accepted else "rejected"
        ),
        {"accepted": "retrieve", "rejected": END},
    )
    builder.add_conditional_edges(
        "retrieve",
        lambda state: "failed" if state.get("errors") else "evaluate",
        {"failed": END, "evaluate": "evaluate"},
    )
    builder.add_conditional_edges(
        "evaluate",
        lambda state: "failed" if state.get("errors") else "queue_notification",
        {"failed": END, "queue_notification": "queue_notification"},
    )
    builder.add_conditional_edges(
        "queue_notification",
        lambda state: (
            "notify" if state["notification_created"] else "duplicate"
        ),
        {"notify": "notify", "duplicate": END},
    )
    builder.add_edge("notify", END)
    return builder.compile()
