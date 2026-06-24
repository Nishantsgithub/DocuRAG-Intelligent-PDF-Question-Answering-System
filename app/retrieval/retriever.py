from __future__ import annotations

from typing import List, Optional

from app.config.settings import Settings, get_settings
from app.retrieval.embeddings import EmbeddingManager
from app.retrieval.vector_store import RetrievedChunk, VectorStoreManager


class Retriever:
    """
    Encode a query and fetch the top-k matching chunks from the vector store.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStoreManager,
        settings: Optional[Settings] = None,
    ):
        self._embedder = embedding_manager
        self._store = vector_store
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_doc_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural-language question.
            k: Number of chunks to return (defaults to settings.retrieval_top_k).
            score_threshold: For Chroma, maximum distance; for FAISS, minimum
                cosine similarity.  Pass None to skip filtering.
            filter_doc_name: Restrict search to a single document by name.

        Returns:
            List of RetrievedChunk sorted best-first.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        k = k or self._settings.retrieval_top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self._settings.retrieval_score_threshold
        )

        query_vec = self._embedder.encode([query])[0]
        results = self._store.search(
            query_embedding=query_vec, k=k, filter_doc_name=filter_doc_name
        )

        if threshold is not None:
            from app.config.settings import VectorStoreBackend

            if self._settings.vector_store_backend == VectorStoreBackend.CHROMA:
                results = [r for r in results if r.score <= threshold]
            else:
                results = [r for r in results if r.score >= threshold]

        return results
