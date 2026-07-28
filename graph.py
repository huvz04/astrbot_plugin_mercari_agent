"""LangGraph orchestration for deterministic listing decisions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .domain import (
    AgentDecision,
    Listing,
    ListingRunState,
    NotificationState,
    WatchRule,
)
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


async def dispatch_notification(
    repository: Repository,
    notifier: Notifier,
    notification_id: int,
) -> NotificationState:
    """Claim and finalize one persisted notification without blind resends."""
    try:
        notification = repository.claim_notification(notification_id)
    except Exception:
        return repository.get_notification(notification_id).state
    if notification is None:
        return repository.get_notification(notification_id).state

    try:
        accepted = await notifier.send(
            notification.target_session,
            notification.message_text,
        )
    except asyncio.CancelledError:
        try:
            repository.mark_notification_verify_required(
                notification_id,
                "dispatch cancelled with unknown result",
            )
        finally:
            raise
    except Exception as error:
        repository.mark_notification_verify_required(notification_id, str(error))
        return NotificationState.VERIFY_REQUIRED

    if accepted:
        try:
            repository.finalize_notification_sent(notification_id)
            return NotificationState.SENT
        except Exception as error:
            try:
                repository.mark_notification_verify_required(
                    notification_id,
                    str(error),
                )
            except Exception:
                pass
            return repository.get_notification(notification_id).state

    try:
        repository.finalize_notification_known_failure(
            notification_id,
            "notification send returned false",
        )
        return NotificationState.FAILED_KNOWN
    except Exception as error:
        try:
            repository.mark_notification_verify_required(
                notification_id,
                str(error),
            )
        except Exception:
            pass
        return repository.get_notification(notification_id).state


async def recover_notification_outbox(
    repository: Repository,
    notifier: Notifier,
) -> None:
    """Conservatively reconcile uncertain sends, then dispatch safe queued work."""
    repository.reconcile_sending_notifications()
    for notification in repository.queued_notifications():
        await dispatch_notification(repository, notifier, notification.id)


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
        run = repository.get_listing_run(state["process_run_id"])
        if run.state not in {
            ListingRunState.FAILED,
            ListingRunState.REJECTED,
            ListingRunState.NOTIFIED,
        }:
            repository.advance_listing_run(
                run.id,
                ListingRunState.FAILED,
                error_summary=str(error),
            )
        return {"errors": [*state.get("errors", ()), str(error)]}

    def guarded_node(
        node: Callable[[ListingGraphState], ListingGraphState],
    ) -> Callable[[ListingGraphState], ListingGraphState]:
        def wrapped(state: ListingGraphState) -> ListingGraphState:
            try:
                return node(state)
            except Exception as error:
                return fail_process_run(state, error)

        return wrapped

    def guarded_async_node(
        node: Callable[[ListingGraphState], Awaitable[ListingGraphState]],
    ) -> Callable[[ListingGraphState], Awaitable[ListingGraphState]]:
        async def wrapped(state: ListingGraphState) -> ListingGraphState:
            try:
                return await node(state)
            except Exception as error:
                return fail_process_run(state, error)

        return wrapped

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
        supplied_run_id = state.get("process_run_id")
        if supplied_run_id is not None:
            supplied_run = repository.get_listing_run(supplied_run_id)
            if (
                supplied_run.listing_id != listing_id
                or supplied_run.watch_rule_id != state["watch_rule"].id
            ):
                raise ValueError(
                    "precreated listing run does not match listing and rule"
                )
            run_id = supplied_run.id
        else:
            run_id, _ = repository.get_or_create_listing_run(
                listing_id,
                state["watch_rule"].id,
            )
        run_created = repository.claim_discovered_listing_run(run_id)
        result["process_run_id"] = run_id
        result["process_run_created"] = run_created
        if run_created:
            try:
                repository.advance_listing_run(
                    run_id,
                    ListingRunState.DEDUP_CHECKED,
                )
            except Exception as error:
                return {**result, **fail_process_run({**state, **result}, error)}
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

    async def retrieve_node(
        state: ListingGraphState,
    ) -> ListingGraphState:
        listing = state["listing"]
        documents = await asyncio.to_thread(
            retriever.retrieve,
            f"{listing.title}\n{listing.description}",
        )
        repository.advance_listing_run(
            state["process_run_id"],
            ListingRunState.RAG_RETRIEVED,
        )
        return {"retrieved_documents": documents}

    async def evaluate_node(
        state: ListingGraphState,
    ) -> ListingGraphState:
        decision = await evaluator.evaluate(
            state["listing"],
            state["retrieved_documents"],
        )
        available_evidence = {
            item.document_id for item in state["retrieved_documents"]
        }
        if not set(decision.retrieved_evidence) <= available_evidence:
            raise ValueError("agent cited unavailable evidence")
        if (
            decision.prompt_version
            != state["watch_rule"].decision_version
        ):
            raise ValueError("agent decision version does not match rule")
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
        notification_id, created = repository.queue_notification_for_run(
            state["process_run_id"],
            decision_version=state["agent_decision"].prompt_version,
            target_session=state["watch_rule"].target_session,
            message_text=message,
        )
        return {
            "notification_id": notification_id,
            "notification_created": created,
        }

    async def notify_node(state: ListingGraphState) -> ListingGraphState:
        dispatch_state = await dispatch_notification(
            repository,
            notifier,
            state["notification_id"],
        )
        if dispatch_state is not NotificationState.SENT:
            notification = repository.get_notification(
                state["notification_id"]
            )
            error = (
                notification.last_error
                or f"notification dispatch ended in {dispatch_state.value}"
            )
            return {"errors": [*state.get("errors", ()), error]}
        return {}

    builder = StateGraph(ListingGraphState)
    builder.add_node("normalize", normalize_node)
    builder.add_node("deduplicate", deduplicate_node)
    builder.add_node("hard_filter", guarded_node(hard_filter_node))
    builder.add_node("retrieve", guarded_async_node(retrieve_node))
    builder.add_node("evaluate", guarded_async_node(evaluate_node))
    builder.add_node("queue_notification", guarded_node(queue_notification_node))
    builder.add_node("notify", guarded_async_node(notify_node))

    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "deduplicate")
    builder.add_conditional_edges(
        "deduplicate",
        lambda state: (
            "failed"
            if state.get("errors")
            else "new"
            if state["process_run_created"]
            else "duplicate"
        ),
        {"new": "hard_filter", "duplicate": END, "failed": END},
    )
    builder.add_conditional_edges(
        "hard_filter",
        lambda state: (
            "failed"
            if state.get("errors")
            else "accepted"
            if state["filter_result"].accepted
            else "rejected"
        ),
        {"accepted": "retrieve", "rejected": END, "failed": END},
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
            "failed"
            if state.get("errors")
            else "notify"
            if state["notification_created"]
            else "duplicate"
        ),
        {"notify": "notify", "duplicate": END, "failed": END},
    )
    builder.add_edge("notify", END)
    return builder.compile()
