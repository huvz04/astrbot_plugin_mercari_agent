"""Local Markdown ingestion and Chroma-backed semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
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
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(Path(persist_dir)),
        )
        return cls(vector_store)

    def retrieve(self, query: str) -> list[Evidence]:
        return [
            Evidence(
                document_id=str(document.metadata["source"]),
                text=document.page_content,
            )
            for document in self.vector_store.similarity_search(query, k=4)
        ]
