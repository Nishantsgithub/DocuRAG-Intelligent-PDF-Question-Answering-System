from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class UploadResponse(BaseModel):
    message: str
    doc_name: str
    pages_loaded: int
    chunks_created: int


class BatchUploadResponse(BaseModel):
    message: str
    total_files: int
    successful: int
    failed: int
    results: List["UploadResponse"]
    errors: List[str]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    k: Optional[int] = Field(default=None, ge=1, le=20, description="Top-k chunks to retrieve")
    score_threshold: Optional[float] = Field(
        default=None, description="Distance threshold for Chroma / similarity threshold for FAISS"
    )
    filter_doc_name: Optional[str] = Field(
        default=None, description="Restrict retrieval to a specific document"
    )


class CitationModel(BaseModel):
    doc_name: str
    page_label: int
    chunk_id: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    latency_seconds: float
    retrieved_chunk_count: int


class EvaluationRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1)
    reference_answers: Optional[List[str]] = None
    k: Optional[int] = Field(default=None, ge=1, le=20)


class EvaluationSummary(BaseModel):
    num_queries: int
    avg_latency_seconds: float
    avg_retrieval_relevance: float
    avg_faithfulness: float
    avg_hallucination_risk: float
    answer_rate: float
