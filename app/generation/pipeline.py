from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.config.settings import Settings, get_settings
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer questions using only the provided context excerpts. "
    "If the answer cannot be found in the context, say exactly: "
    "\"I could not find the answer in the provided documents.\""
)

_USER_TEMPLATE = """Context:
{context}

Question:
{query}

Instructions:
- Answer concisely and accurately based solely on the context above.
- After your answer, list the sources you used in the following format:
  Sources:
  - [<doc_name>, page <page_label>]
"""


@dataclass
class Citation:
    doc_name: str
    page_label: int
    chunk_id: str


@dataclass
class RAGResponse:
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    latency_seconds: float
    context_used: str = field(repr=False)


class RAGPipeline:
    """
    End-to-end RAG pipeline: retrieve -> build context -> generate answer.

    The generated answer always includes source citations (document name
    and page number) derived from the retrieved chunks.
    """

    def __init__(
        self,
        retriever: Retriever,
        settings: Optional[Settings] = None,
    ):
        self._retriever = retriever
        self._settings = settings or get_settings()
        self._client = self._build_openai_client()

    def _build_openai_client(self):
        from openai import OpenAI

        if not self._settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set in your environment to use the generation pipeline."
            )
        return OpenAI(api_key=self._settings.openai_api_key)

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        parts: List[str] = []
        total = 0

        for i, chunk in enumerate(chunks, start=1):
            block = (
                f"[{i}] {chunk.doc_name} — page {chunk.page_label}\n"
                f"{chunk.text}"
            )
            if total + len(block) > self._settings.max_context_chars:
                break
            parts.append(block)
            total += len(block)

        return "\n\n---\n\n".join(parts)

    def _generate(self, query: str, context: str) -> str:
        prompt = _USER_TEMPLATE.format(context=context, query=query)
        response = self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._settings.llm_temperature,
        )
        return response.choices[0].message.content.strip()

    def run(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_doc_name: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        Args:
            query: Natural-language question.
            k: Override retrieval top-k.
            score_threshold: Override retrieval distance/similarity threshold.
            filter_doc_name: Restrict retrieval to a single document.

        Returns:
            RAGResponse with answer, citations, chunks, and latency.
        """
        t0 = time.perf_counter()

        chunks = self._retriever.retrieve(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter_doc_name=filter_doc_name,
        )

        context = self._build_context(chunks)
        answer = self._generate(query=query, context=context)

        citations = [
            Citation(
                doc_name=c.doc_name,
                page_label=c.page_label,
                chunk_id=c.chunk_id,
            )
            for c in chunks
        ]

        latency = time.perf_counter() - t0

        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            latency_seconds=round(latency, 3),
            context_used=context,
        )
