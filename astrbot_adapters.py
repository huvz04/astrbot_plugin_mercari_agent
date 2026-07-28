"""Adapters between AstrBot providers and the local Mercari agent core."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Coroutine
from typing import Any, TypeVar

from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain
from astrbot.api.star import Context
from langchain_core.embeddings import Embeddings

from .domain import AgentDecision, Listing
from .rag import Evidence

_PROMPT_VERSION = "mercari-v1"
_RISK_TERMS = ("ジャンク", "欠品", "破損", "偽物", "risk", "风险")
T = TypeVar("T")


class AstrBotNotifier:
    """Send a single plain AstrBot message and retain no delivery secrets."""

    def __init__(self, context: Context) -> None:
        self.context = context
        self.successful_sends = 0

    async def send(self, target_session: str, text: str) -> bool:
        sent = (
            await self.context.send_message(
                target_session,
                MessageChain([Plain(text)]),
            )
            is True
        )
        if sent:
            self.successful_sends += 1
        return sent


class AstrBotEvaluator:
    """Request and validate a strict AgentDecision from an AstrBot provider."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    async def evaluate(
        self, listing: Listing, evidence: list[Evidence]
    ) -> AgentDecision:
        prompt = self._prompt(listing, evidence)
        response = await self.provider.text_chat(prompt=prompt)
        completion = getattr(response, "completion_text", "")
        decision = AgentDecision.model_validate_json(
            _strip_single_json_fence(completion)
        )
        return decision.model_copy(
            update={
                "model_name": _provider_model_name(self.provider),
                "prompt_version": _PROMPT_VERSION,
            }
        )

    @staticmethod
    def _prompt(listing: Listing, evidence: list[Evidence]) -> str:
        payload = {
            "listing": listing.model_dump(mode="json"),
            "evidence": [
                {"document_id": item.document_id, "text": item.text}
                for item in evidence
            ],
        }
        schema = AgentDecision.model_json_schema()
        return (
            "Evaluate this Mercari listing. Return exactly one JSON object and "
            "no prose. The object must validate against this AgentDecision JSON "
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
        )


def _strip_single_json_fence(value: str) -> str:
    """Remove one complete Markdown JSON fence, without extracting other text."""
    stripped = value.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _provider_model_name(provider: Any) -> str:
    try:
        metadata = provider.meta()
    except Exception:
        metadata = None
    provider_name = (
        getattr(metadata, "id", None)
        or getattr(metadata, "type", None)
        or provider.__class__.__name__
    )
    model = getattr(metadata, "model", None)
    return f"{provider_name}:{model}" if model else str(provider_name)


class AstrBotEmbeddings(Embeddings):
    """Bridge AstrBot async embeddings into LangChain's synchronous API.

    Chroma calls these methods from a worker thread. Provider coroutines are
    scheduled back onto the AstrBot event loop captured during initialization.
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        try:
            self._event_loop: asyncio.AbstractEventLoop | None = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            self._event_loop = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run_provider(self.provider.get_embeddings(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._run_provider(self.provider.get_embedding(text))

    def _run_provider(self, coroutine: Coroutine[Any, Any, T]) -> T:
        loop = self._event_loop
        if loop is not None and loop.is_running():
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is loop:
                coroutine.close()
                raise RuntimeError(
                    "AstrBotEmbeddings must be called from asyncio.to_thread"
                )
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            return future.result()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        coroutine.close()
        raise RuntimeError(
            "AstrBotEmbeddings requires construction on the AstrBot event loop "
            "and invocation through asyncio.to_thread"
        )


class DeterministicEmbeddings(Embeddings):
    """Stable local embeddings used when AstrBot has no embedding provider."""

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


class DeterministicEvaluator:
    """Offline listing/risk heuristic with stable structured output."""

    async def evaluate(
        self, listing: Listing, evidence: list[Evidence]
    ) -> AgentDecision:
        searchable = " ".join(
            (
                listing.title,
                listing.description,
                *(item.text for item in evidence),
            )
        ).lower()
        risks = tuple(term for term in _RISK_TERMS if term.lower() in searchable)
        price_penalty = min(listing.price_jpy // 500, 30)
        score = max(0, 90 - price_penalty - (25 * len(risks)))
        recommendation = (
            "HIGH_PRIORITY"
            if score >= 70 and not risks
            else "REVIEW"
            if score >= 40
            else "SKIP"
        )
        reasons = (
            f"offline price heuristic for JPY {listing.price_jpy}",
            "deterministic local evaluation",
        )
        return AgentDecision(
            score=score,
            recommendation=recommendation,
            reasons=reasons,
            risks=risks,
            retrieved_evidence=tuple(item.document_id for item in evidence),
            model_name="deterministic-fallback",
            prompt_version=_PROMPT_VERSION,
        )
