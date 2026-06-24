from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.config.settings import EmbeddingBackend, Settings, get_settings


class EmbeddingManager:
    """Encode text into dense vectors using either sentence-transformers or OpenAI."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._model = None
        self._openai_client = None
        self._load()

    def _load(self) -> None:
        backend = self._settings.embedding_backend

        if backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._settings.sentence_transformer_model
            )
            self._dim = self._model.get_sentence_embedding_dimension()

        elif backend == EmbeddingBackend.OPENAI:
            from openai import OpenAI

            if not self._settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be set to use the OpenAI embedding backend."
                )
            self._openai_client = OpenAI(api_key=self._settings.openai_api_key)
            # dimension is known per model; resolve lazily on first encode
            self._dim = None

        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    @property
    def dimension(self) -> Optional[int]:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into a 2-D numpy array of shape (N, dim).

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            Float32 numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            raise ValueError("texts must be non-empty.")

        backend = self._settings.embedding_backend

        if backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
            vectors = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            return vectors

        # OpenAI path
        response = self._openai_client.embeddings.create(
            input=texts,
            model=self._settings.openai_embedding_model,
        )
        vectors = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )
        if self._dim is None:
            self._dim = vectors.shape[1]
        return vectors


def build_embedding_manager(settings: Optional[Settings] = None) -> EmbeddingManager:
    return EmbeddingManager(settings=settings)
