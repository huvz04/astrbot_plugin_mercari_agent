"""Local Markdown ingestion and Chroma-backed semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Evidence:
    document_id: str
    text: str


class MarkdownChromaRetriever:
    def __init__(self, vector_store: Chroma) -> None:
        self.vector_store = vector_store

    @classmethod
    def build(
        cls,
        knowledge_dir: str | Path,
        persist_dir: str | Path,
        embeddings: Any,
    ) -> "MarkdownChromaRetriever":
        knowledge_path = Path(knowledge_dir)
        documents = [
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.stem},
            )
            for path in sorted(knowledge_path.glob("*.md"))
            if path.is_file()
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        chunk_ids = [
            hashlib.sha256(
                (
                    str(chunk.metadata["source"])
                    + "\0"
                    + chunk.page_content
                ).encode("utf-8")
            ).hexdigest()
            for chunk in chunks
        ]
        corpus_fingerprint = hashlib.sha256(
            "\n".join(sorted(chunk_ids)).encode("utf-8")
        ).hexdigest()
        embedding_identity = _embedding_identity(embeddings)
        embedding_dimension = _embedding_dimension(embeddings)
        collection_fingerprint = hashlib.sha256(
            (
                "mercari-rag-v1\0"
                + corpus_fingerprint
                + "\0"
                + embedding_identity
                + "\0"
                + str(embedding_dimension)
            ).encode("utf-8")
        ).hexdigest()
        vector_store = Chroma(
            collection_name=f"mercari_{collection_fingerprint[:32]}",
            embedding_function=embeddings,
            persist_directory=str(Path(persist_dir)),
        )
        if chunks:
            vector_store.add_documents(chunks, ids=chunk_ids)
        return cls(vector_store)

    def retrieve(self, query: str) -> list[Evidence]:
        return [
            Evidence(
                document_id=str(document.metadata["source"]),
                text=document.page_content,
            )
            for document in self.vector_store.similarity_search(query, k=4)
        ]


def _embedding_identity(embeddings: Any) -> str:
    configured = getattr(embeddings, "embedding_identity", None)
    if callable(configured):
        configured = configured()
    if configured:
        return str(configured)

    provider = getattr(embeddings, "provider", None)
    if provider is not None:
        try:
            metadata = provider.meta()
        except Exception:
            metadata = None
        provider_id = getattr(metadata, "id", None)
        provider_model = getattr(metadata, "model", None)
        if provider_id or provider_model:
            return f"{provider_id or provider.__class__.__name__}:{provider_model or '-'}"

    cls = embeddings.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _embedding_dimension(embeddings: Any) -> int:
    configured = getattr(embeddings, "dimension", None)
    if isinstance(configured, int) and configured > 0:
        return configured
    vector = embeddings.embed_query("mercari-rag-dimension-probe")
    dimension = len(vector)
    if dimension < 1:
        raise ValueError("embedding adapter returned an empty vector")
    return dimension
