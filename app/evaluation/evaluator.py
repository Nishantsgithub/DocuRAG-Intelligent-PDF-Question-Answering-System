from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.generation.pipeline import RAGPipeline, RAGResponse
from app.retrieval.vector_store import RetrievedChunk


@dataclass
class EvaluationResult:
    """Metrics for a single query evaluation run."""

    query: str
    answer: str
    retrieved_chunk_count: int
    latency_seconds: float

    # Retrieval relevance: fraction of retrieved chunks whose text overlaps
    # with the reference answer (lexical proxy, no LLM required).
    retrieval_relevance_score: float

    # Answer faithfulness: fraction of answer sentences that can be grounded
    # in at least one retrieved chunk (lexical proxy).
    faithfulness_score: float

    # Hallucination risk: 1 - faithfulness_score (higher = riskier).
    hallucination_risk_score: float

    # Whether the pipeline returned an "I could not find" response.
    answered: bool

    sources: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregate metrics across multiple evaluation queries."""

    results: List[EvaluationResult]

    @property
    def avg_latency(self) -> float:
        return round(
            sum(r.latency_seconds for r in self.results) / max(len(self.results), 1), 3
        )

    @property
    def avg_retrieval_relevance(self) -> float:
        return round(
            sum(r.retrieval_relevance_score for r in self.results)
            / max(len(self.results), 1),
            3,
        )

    @property
    def avg_faithfulness(self) -> float:
        return round(
            sum(r.faithfulness_score for r in self.results) / max(len(self.results), 1),
            3,
        )

    @property
    def avg_hallucination_risk(self) -> float:
        return round(
            sum(r.hallucination_risk_score for r in self.results)
            / max(len(self.results), 1),
            3,
        )

    @property
    def answer_rate(self) -> float:
        return round(
            sum(1 for r in self.results if r.answered) / max(len(self.results), 1), 3
        )

    def summary(self) -> dict:
        return {
            "num_queries": len(self.results),
            "avg_latency_seconds": self.avg_latency,
            "avg_retrieval_relevance": self.avg_retrieval_relevance,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_hallucination_risk": self.avg_hallucination_risk,
            "answer_rate": self.answer_rate,
        }


def _tokenize(text: str) -> set[str]:
    """Very simple whitespace+punctuation tokeniser for overlap scoring."""
    import re

    return set(re.findall(r"\b\w+\b", text.lower()))


def _retrieval_relevance(
    chunks: List[RetrievedChunk], reference: str
) -> float:
    """
    Fraction of retrieved chunks that share at least one content word with
    the reference text.  Uses token overlap as a cheap, LLM-free proxy.
    """
    if not chunks or not reference:
        return 0.0

    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0

    relevant = sum(
        1 for c in chunks if _tokenize(c.text) & ref_tokens
    )
    return round(relevant / len(chunks), 3)


def _faithfulness(answer: str, chunks: List[RetrievedChunk]) -> float:
    """
    Fraction of non-trivial answer sentences that overlap with at least one
    retrieved chunk.  Sentences beginning with "I could not" are excluded.
    """
    if not answer or not chunks:
        return 0.0

    context_tokens = set()
    for c in chunks:
        context_tokens |= _tokenize(c.text)

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    sentences = [s for s in sentences if not s.lower().startswith("i could not")]

    if not sentences:
        return 1.0  # short / non-answerable responses are not hallucinations

    grounded = sum(
        1 for s in sentences if _tokenize(s) & context_tokens
    )
    return round(grounded / len(sentences), 3)


class Evaluator:
    """
    Run and score a set of queries through the RAG pipeline.

    Metrics are computed using lexical overlap — no external LLM calls are
    needed, making evaluation fast and deterministic.  For production use,
    swap in embedding-based or LLM-judge metrics as needed.
    """

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def evaluate_single(
        self,
        query: str,
        reference_answer: Optional[str] = None,
        k: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Run the pipeline for one query and compute metrics.

        Args:
            query: The test question.
            reference_answer: Optional gold-standard answer for relevance
                scoring.  If omitted, the generated answer is used as proxy.
            k: Override retrieval top-k.

        Returns:
            EvaluationResult with all metric scores.
        """
        response: RAGResponse = self._pipeline.run(query=query, k=k)

        reference = reference_answer or response.answer

        rel_score = _retrieval_relevance(response.retrieved_chunks, reference)
        faith_score = _faithfulness(response.answer, response.retrieved_chunks)
        halluc_risk = round(1.0 - faith_score, 3)

        not_found_phrase = "i could not find"
        answered = not_found_phrase not in response.answer.lower()

        sources = list(
            dict.fromkeys(
                f"{c.doc_name}, page {c.page_label}"
                for c in response.retrieved_chunks
            )
        )

        return EvaluationResult(
            query=query,
            answer=response.answer,
            retrieved_chunk_count=len(response.retrieved_chunks),
            latency_seconds=response.latency_seconds,
            retrieval_relevance_score=rel_score,
            faithfulness_score=faith_score,
            hallucination_risk_score=halluc_risk,
            answered=answered,
            sources=sources,
        )

    def evaluate_batch(
        self,
        queries: List[str],
        reference_answers: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> EvaluationReport:
        """
        Evaluate a list of queries and return an aggregate report.

        Args:
            queries: List of test questions.
            reference_answers: Optional list of gold answers (same length as
                queries).  Pass None to use generated answers as proxies.
            k: Override retrieval top-k for all queries.

        Returns:
            EvaluationReport containing per-query results and summary stats.
        """
        references = reference_answers or [None] * len(queries)
        if len(references) != len(queries):
            raise ValueError(
                "reference_answers must have the same length as queries."
            )

        results: List[EvaluationResult] = []
        for query, ref in zip(queries, references):
            result = self.evaluate_single(query=query, reference_answer=ref, k=k)
            results.append(result)

        return EvaluationReport(results=results)
