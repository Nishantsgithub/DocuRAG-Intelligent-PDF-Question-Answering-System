"""Tests for the generation pipeline and evaluation module."""
from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.evaluator import (
    EvaluationReport,
    EvaluationResult,
    Evaluator,
    _faithfulness,
    _retrieval_relevance,
    _tokenize,
)
from app.generation.pipeline import Citation, RAGPipeline, RAGResponse
from app.retrieval.vector_store import RetrievedChunk


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_chunk(text: str = "The transformer model uses self-attention.", page_label: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        doc_name="paper",
        page_label=page_label,
        page_number=page_label - 1,
        chunk_id="abc123",
        upload_timestamp="2024-01-01T00:00:00+00:00",
        source_path="paper.pdf",
        score=0.5,
    )


def _make_pipeline(answer: str = "Self-attention maps query-key-value pairs.") -> RAGPipeline:
    """Return a RAGPipeline with mocked retriever and OpenAI client."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_chunk("The transformer model uses self-attention."),
        _make_chunk("Attention maps queries to keys and values.", page_label=2),
    ]

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._retriever = mock_retriever
    pipeline._settings = MagicMock()
    pipeline._settings.openai_api_key = "sk-test"
    pipeline._settings.openai_model = "gpt-4o-mini"
    pipeline._settings.llm_temperature = 0.2
    pipeline._settings.max_context_chars = 6000
    pipeline._settings.retrieval_top_k = 5
    pipeline._settings.retrieval_score_threshold = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = answer
    pipeline._client = mock_client

    return pipeline


# ── RAGPipeline ────────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_run_returns_rag_response(self):
        pipeline = _make_pipeline("Self-attention maps queries to keys.")
        response = pipeline.run(query="What is self-attention?")

        assert isinstance(response, RAGResponse)
        assert response.query == "What is self-attention?"
        assert "attention" in response.answer.lower()

    def test_run_includes_citations(self):
        pipeline = _make_pipeline()
        response = pipeline.run(query="What is attention?")

        assert len(response.citations) == 2
        assert all(isinstance(c, Citation) for c in response.citations)
        assert response.citations[0].doc_name == "paper"

    def test_run_measures_latency(self):
        pipeline = _make_pipeline()
        response = pipeline.run(query="What is attention?")
        assert response.latency_seconds >= 0

    def test_build_context_truncates_at_max_chars(self):
        """Context builder should stop adding chunks once max_context_chars is hit."""
        pipeline = _make_pipeline()
        pipeline._settings.max_context_chars = 50  # very small

        chunks = [_make_chunk("A" * 100), _make_chunk("B" * 100)]
        context = pipeline._build_context(chunks)

        assert len(context) <= 150  # only first chunk fits


# ── Evaluation helpers ─────────────────────────────────────────────────────────

class TestEvaluationHelpers:
    def test_tokenize_basic(self):
        tokens = _tokenize("The quick brown fox")
        assert "quick" in tokens
        assert "fox" in tokens

    def test_tokenize_case_insensitive(self):
        assert _tokenize("Hello") == _tokenize("hello")

    def test_retrieval_relevance_full_overlap(self):
        chunk = _make_chunk("attention transformer self-attention")
        score = _retrieval_relevance([chunk], "attention transformer self-attention")
        assert score == 1.0

    def test_retrieval_relevance_no_overlap(self):
        chunk = _make_chunk("the cat sat on the mat")
        score = _retrieval_relevance([chunk], "quantum chromodynamics")
        assert score == 0.0

    def test_retrieval_relevance_empty_inputs(self):
        assert _retrieval_relevance([], "anything") == 0.0
        assert _retrieval_relevance([_make_chunk()], "") == 0.0

    def test_faithfulness_full_grounding(self):
        chunk = _make_chunk("attention transformer query key value output")
        score = _faithfulness(
            "The attention mechanism maps query key value to output.",
            [chunk],
        )
        assert score > 0.5

    def test_faithfulness_no_grounding(self):
        chunk = _make_chunk("the cat sat on the mat")
        score = _faithfulness("Quantum entanglement is mysterious.", [chunk])
        assert score < 0.5

    def test_faithfulness_not_found_response(self):
        """'I could not find' answers should not be penalised."""
        chunk = _make_chunk()
        score = _faithfulness(
            "I could not find the answer in the provided documents.", [chunk]
        )
        assert score == 1.0


# ── Evaluator ──────────────────────────────────────────────────────────────────

class TestEvaluator:
    def _make_evaluator(self, answer: str = "Attention maps queries to keys.") -> Evaluator:
        pipeline = _make_pipeline(answer=answer)
        return Evaluator(pipeline=pipeline)

    def test_evaluate_single_returns_result(self):
        evaluator = self._make_evaluator()
        result = evaluator.evaluate_single("What is attention?")

        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.retrieval_relevance_score <= 1.0
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert 0.0 <= result.hallucination_risk_score <= 1.0

    def test_evaluate_single_scores_sum_to_one(self):
        evaluator = self._make_evaluator()
        result = evaluator.evaluate_single("What is attention?")
        total = round(result.faithfulness_score + result.hallucination_risk_score, 5)
        assert total == 1.0

    def test_evaluate_batch_report(self):
        evaluator = self._make_evaluator()
        queries = ["What is attention?", "What are transformers?"]
        report = evaluator.evaluate_batch(queries)

        assert isinstance(report, EvaluationReport)
        assert len(report.results) == 2
        assert report.avg_latency >= 0

    def test_evaluate_batch_mismatched_refs_raises(self):
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate_batch(
                queries=["q1", "q2"],
                reference_answers=["only one ref"],
            )
