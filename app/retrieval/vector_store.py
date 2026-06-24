from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.config.settings import Settings, VectorStoreBackend, get_settings
from app.ingestion.chunker import DocumentChunk


@dataclass
class RetrievedChunk:
    """A chunk returned by the vector store, with score and provenance."""

    text: str
    doc_name: str
    page_label: int       # 1-indexed
    page_number: int      # 0-indexed
    chunk_id: str
    upload_timestamp: str
    source_path: str
    score: float          # lower = more similar for Chroma; higher for FAISS cosine


class VectorStoreManager:
    """
    Manages document storage and retrieval.

    Supports ChromaDB (default) and FAISS (optional).  Both backends share
    the same public interface: ``add_chunks`` and ``search``.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._backend = self._settings.vector_store_backend

        # ChromaDB state
        self._chroma_client = None
        self._chroma_collection = None

        # FAISS state
        self._faiss_index = None
        self._faiss_metadata: List[Dict[str, Any]] = []
        self._faiss_texts: List[str] = []

        self._init_backend()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        if self._backend == VectorStoreBackend.CHROMA:
            self._init_chroma()
        else:
            self._init_faiss()

    def _init_chroma(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(self._settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(
            path=self._settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=self._settings.chroma_collection_name
        )

    def _verify_chroma_dimension(self, embedding_dim: int) -> None:
        """Delete and recreate the collection if its dimension doesn't match."""
        meta = self._chroma_collection.metadata or {}
        stored_dim = meta.get("embedding_dim")
        if stored_dim is None:
            # First add — record the dimension in collection metadata
            self._chroma_client.delete_collection(self._settings.chroma_collection_name)
            self._chroma_collection = self._chroma_client.create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"embedding_dim": embedding_dim},
            )
        elif stored_dim != embedding_dim:
            # Stale collection with wrong dimension — wipe and recreate
            self._chroma_client.delete_collection(self._settings.chroma_collection_name)
            self._chroma_collection = self._chroma_client.create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"embedding_dim": embedding_dim},
            )

    def _init_faiss(self) -> None:
        import faiss

        index_path = Path(self._settings.faiss_index_path)
        meta_path = index_path.with_suffix(".meta.pkl")

        if index_path.exists() and meta_path.exists():
            self._faiss_index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                saved = pickle.load(f)
            self._faiss_metadata = saved["metadata"]
            self._faiss_texts = saved["texts"]
        # Index created lazily on first add_chunks call (need embedding dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self, chunks: List[DocumentChunk], embeddings: np.ndarray
    ) -> None:
        """Persist chunks and their embeddings in the active vector store."""
        if not chunks or embeddings is None or len(embeddings) == 0:
            raise ValueError("chunks and embeddings must be non-empty.")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        if self._backend == VectorStoreBackend.CHROMA:
            self._add_to_chroma(chunks, embeddings)
        else:
            self._add_to_faiss(chunks, embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_doc_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Return the top-k most similar chunks.

        Args:
            query_embedding: 1-D float32 array of shape (dim,).
            k: Number of results to return.
            filter_doc_name: Optional — restrict results to a single document.

        Returns:
            List of RetrievedChunk sorted by relevance (best first).
        """
        if self._backend == VectorStoreBackend.CHROMA:
            return self._search_chroma(query_embedding, k, filter_doc_name)
        return self._search_faiss(query_embedding, k, filter_doc_name)

    def collection_count(self) -> int:
        """Return number of stored vectors."""
        if self._backend == VectorStoreBackend.CHROMA:
            return self._chroma_collection.count()
        return len(self._faiss_metadata)

    def reset_collection(self) -> None:
        """Delete all stored chunks (useful for re-ingestion)."""
        if self._backend == VectorStoreBackend.CHROMA:
            self._chroma_client.delete_collection(
                self._settings.chroma_collection_name
            )
            self._init_chroma()
        else:
            self._faiss_index = None
            self._faiss_metadata = []
            self._faiss_texts = []
            index_path = Path(self._settings.faiss_index_path)
            meta_path = index_path.with_suffix(".meta.pkl")
            index_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # ChromaDB internals
    # ------------------------------------------------------------------

    def _add_to_chroma(
        self, chunks: List[DocumentChunk], embeddings: np.ndarray
    ) -> None:
        self._verify_chroma_dimension(embeddings.shape[1])
        self._chroma_collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
            embeddings=embeddings.tolist(),
        )

    def _search_chroma(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_doc_name: Optional[str],
    ) -> List[RetrievedChunk]:
        where = {"doc_name": filter_doc_name} if filter_doc_name else None
        results = self._chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, distances):
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    doc_name=meta.get("doc_name", ""),
                    page_label=meta.get("page_label", 1),
                    page_number=meta.get("page_number", 0),
                    chunk_id=meta.get("chunk_id", ""),
                    upload_timestamp=meta.get("upload_timestamp", ""),
                    source_path=meta.get("source_path", ""),
                    score=dist,
                )
            )
        return retrieved

    # ------------------------------------------------------------------
    # FAISS internals
    # ------------------------------------------------------------------

    def _ensure_faiss_index(self, dim: int) -> None:
        if self._faiss_index is None:
            import faiss

            inner = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIDMap(inner)

    def _add_to_faiss(
        self, chunks: List[DocumentChunk], embeddings: np.ndarray
    ) -> None:
        import faiss

        dim = embeddings.shape[1]
        self._ensure_faiss_index(dim)

        # Normalise for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (embeddings / norms).astype(np.float32)

        start_id = len(self._faiss_metadata)
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)

        self._faiss_index.add_with_ids(normed, ids)
        self._faiss_metadata.extend([c.to_metadata() for c in chunks])
        self._faiss_texts.extend([c.text for c in chunks])

        # Persist
        index_path = Path(self._settings.faiss_index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(index_path))
        meta_path = index_path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"metadata": self._faiss_metadata, "texts": self._faiss_texts}, f
            )

    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_doc_name: Optional[str],
    ) -> List[RetrievedChunk]:
        if self._faiss_index is None or len(self._faiss_metadata) == 0:
            return []

        vec = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        actual_k = min(k * 3 if filter_doc_name else k, len(self._faiss_metadata))
        scores, indices = self._faiss_index.search(vec, actual_k)

        retrieved: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._faiss_metadata[idx]
            if filter_doc_name and meta.get("doc_name") != filter_doc_name:
                continue
            retrieved.append(
                RetrievedChunk(
                    text=self._faiss_texts[idx],
                    doc_name=meta.get("doc_name", ""),
                    page_label=meta.get("page_label", 1),
                    page_number=meta.get("page_number", 0),
                    chunk_id=meta.get("chunk_id", ""),
                    upload_timestamp=meta.get("upload_timestamp", ""),
                    source_path=meta.get("source_path", ""),
                    score=float(score),
                )
            )
            if len(retrieved) == k:
                break

        return retrieved
