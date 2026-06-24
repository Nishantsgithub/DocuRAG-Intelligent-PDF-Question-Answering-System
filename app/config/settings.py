from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingBackend(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,  # allow both field name and alias in constructors
    )

    # --- OpenAI ---
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # --- Embeddings ---
    embedding_backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.SENTENCE_TRANSFORMERS, alias="EMBEDDING_BACKEND"
    )
    sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2", alias="SENTENCE_TRANSFORMER_MODEL"
    )

    # --- Vector store ---
    vector_store_backend: VectorStoreBackend = Field(
        default=VectorStoreBackend.CHROMA, alias="VECTOR_STORE_BACKEND"
    )
    chroma_persist_dir: str = Field(
        default="data/vector_store/chroma", alias="CHROMA_PERSIST_DIR"
    )
    faiss_index_path: str = Field(
        default="data/vector_store/faiss.index", alias="FAISS_INDEX_PATH"
    )
    chroma_collection_name: str = Field(
        default="docurag_collection", alias="CHROMA_COLLECTION_NAME"
    )

    # --- Chunking ---
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_score_threshold: Optional[float] = Field(
        default=None, alias="RETRIEVAL_SCORE_THRESHOLD"
    )

    # --- Generation ---
    max_context_chars: int = Field(default=6000, alias="MAX_CONTEXT_CHARS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    # --- Paths ---
    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")
    sample_data_dir: str = Field(default="sample_data", alias="SAMPLE_DATA_DIR")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    def ensure_dirs(self) -> None:
        for path in (
            self.upload_dir,
            self.chroma_persist_dir,
            Path(self.faiss_index_path).parent,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
