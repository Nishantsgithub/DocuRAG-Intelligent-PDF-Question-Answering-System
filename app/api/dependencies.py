"""FastAPI dependency injection: singletons for the embedding/store/pipeline stack."""
from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.evaluation.evaluator import Evaluator
from app.generation.pipeline import RAGPipeline
from app.retrieval.embeddings import build_embedding_manager
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStoreManager


@lru_cache(maxsize=1)
def get_embedding_manager():
    return build_embedding_manager()


@lru_cache(maxsize=1)
def get_vector_store():
    return VectorStoreManager()


@lru_cache(maxsize=1)
def get_retriever():
    return Retriever(
        embedding_manager=get_embedding_manager(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_pipeline():
    return RAGPipeline(retriever=get_retriever())


@lru_cache(maxsize=1)
def get_evaluator():
    return Evaluator(pipeline=get_pipeline())
