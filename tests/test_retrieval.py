"""Tests for the retrieval module (embeddings, vector store, retriever)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ingestion.chunker import DocumentChunk
from app.retrieval.vector_store import RetrievedChunk, VectorStoreManager


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_chunk(
    text: str = "Sample text about transformers.",
    doc_name: str = "paper",
    page_number: int = 0,
    page_label: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    import uuid

    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        doc_name=doc_name,
        page_number=page_number,
        page_label=page_label,
        total_pages=10,
        upload_timestamp="2024-01-01T00:00:00+00:00",
        source_path="paper.pdf",
        chunk_index=chunk_index,
    )


# ── EmbeddingManager ───────────────────────────────────────────────────────────

class TestEmbeddingManager:
    def test_encode_returns_correct_shape(self):
        from app.config.settings import EmbeddingBackend, Settings
        from app.retrieval.embeddings import EmbeddingManager

        settings = Settings(
            embedding_backend=EmbeddingBackend.SENTENCE_TRANSFORMERS,
            sentence_transformer_model="all-MiniLM-L6-v2",
        )
        manager = EmbeddingManager(settings=settings)

        texts = ["Hello world", "Transformers are great", "RAG pipelines rock"]
        vectors = manager.encode(texts)

        assert vectors.shape == (3, manager.dimension)
        assert vectors.dtype == np.float32

    def test_encode_raises_on_empty_input(self):
        from app.config.settings import EmbeddingBackend, Settings
        from app.retrieval.embeddings import EmbeddingManager

        settings = Settings(embedding_backend=EmbeddingBackend.SENTENCE_TRANSFORMERS)
        manager = EmbeddingManager(settings=settings)

        with pytest.raises(ValueError):
            manager.encode([])


# ── VectorStoreManager (ChromaDB) ──────────────────────────────────────────────

class TestVectorStoreChroma:
    @pytest.fixture
    def store(self, tmp_path):
        from app.config.settings import Settings, VectorStoreBackend

        settings = Settings(
            vector_store_backend=VectorStoreBackend.CHROMA,
            chroma_persist_dir=str(tmp_path / "chroma"),
            chroma_collection_name="test_collection",
        )
        return VectorStoreManager(settings=settings)

    def test_empty_store_count(self, store):
        assert store.collection_count() == 0

    def test_add_and_count(self, store):
        chunks = [_make_chunk(text=f"chunk {i}") for i in range(5)]
        embeddings = np.random.rand(5, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        assert store.collection_count() == 5

    def test_search_returns_results(self, store):
        chunks = [_make_chunk(text="The attention mechanism is key to transformers.")]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)

        query_vec = np.random.rand(384).astype(np.float32)
        results = store.search(query_vec, k=1)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)

    def test_filter_by_doc_name(self, store):
        chunks_a = [_make_chunk(text=f"Doc A chunk {i}", doc_name="docA") for i in range(3)]
        chunks_b = [_make_chunk(text=f"Doc B chunk {i}", doc_name="docB") for i in range(3)]
        embeddings = np.random.rand(6, 384).astype(np.float32)
        store.add_chunks(chunks_a + chunks_b, embeddings)

        query_vec = np.random.rand(384).astype(np.float32)
        results = store.search(query_vec, k=10, filter_doc_name="docA")

        assert all(r.doc_name == "docA" for r in results)

    def test_reset_collection(self, store):
        chunks = [_make_chunk()]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        store.reset_collection()
        assert store.collection_count() == 0

    def test_add_chunks_raises_on_mismatch(self, store):
        chunks = [_make_chunk() for _ in range(3)]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        with pytest.raises(ValueError):
            store.add_chunks(chunks, embeddings)


# ── VectorStoreManager (FAISS) ─────────────────────────────────────────────────

class TestVectorStoreFAISS:
    @pytest.fixture
    def store(self, tmp_path):
        pytest.importorskip("faiss")
        from app.config.settings import Settings, VectorStoreBackend

        settings = Settings(
            vector_store_backend=VectorStoreBackend.FAISS,
            faiss_index_path=str(tmp_path / "faiss.index"),
        )
        return VectorStoreManager(settings=settings)

    def test_add_and_search(self, store):
        chunks = [_make_chunk(text=f"FAISS chunk {i}") for i in range(4)]
        embeddings = np.random.rand(4, 128).astype(np.float32)
        store.add_chunks(chunks, embeddings)

        query_vec = np.random.rand(128).astype(np.float32)
        results = store.search(query_vec, k=2)

        assert len(results) == 2
        assert isinstance(results[0], RetrievedChunk)
