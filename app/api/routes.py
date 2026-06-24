from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from typing import List

from app.api.dependencies import (
    get_embedding_manager,
    get_evaluator,
    get_pipeline,
    get_vector_store,
)
from app.api.schemas import (
    BatchUploadResponse,
    CitationModel,
    EvaluationRequest,
    EvaluationSummary,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.config.settings import get_settings
from app.ingestion.chunker import Chunker
from app.ingestion.loader import PDFLoader

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Liveness probe — returns 200 when the service is ready."""
    return HealthResponse()


async def _ingest_one(file: UploadFile, embedding_manager, vector_store) -> UploadResponse:
    """Ingest a single UploadFile — shared by single and batch upload routes."""
    settings = get_settings()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise ValueError(f"'{file.filename}' is not a PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        loader = PDFLoader()
        pages = loader.load_file(tmp_path)

        if not pages:
            raise ValueError(f"'{file.filename}' is empty or has no extractable text.")

        doc_name = Path(file.filename).stem
        for page in pages:
            page.doc_name = doc_name
            page.source_path = file.filename

        chunker = Chunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chunks = chunker.chunk_pages(pages)
        embeddings = embedding_manager.encode([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(
        message="Document ingested successfully.",
        doc_name=doc_name,
        pages_loaded=len(pages),
        chunks_created=len(chunks),
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document to ingest"),
    embedding_manager=Depends(get_embedding_manager),
    vector_store=Depends(get_vector_store),
):
    """Upload a single PDF, chunk it, embed it, and store it in the vector database."""
    try:
        return await _ingest_one(file, embedding_manager, vector_store)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def upload_pdf_batch(
    files: List[UploadFile] = File(..., description="PDF documents to ingest (select multiple)"),
    embedding_manager=Depends(get_embedding_manager),
    vector_store=Depends(get_vector_store),
):
    """
    Upload multiple PDFs in a single request.

    Each file is processed independently — a failure on one does not stop the others.
    Returns a summary with per-file results and any errors.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No files provided.")

    results: List[UploadResponse] = []
    errors: List[str] = []

    for file in files:
        try:
            result = await _ingest_one(file, embedding_manager, vector_store)
            results.append(result)
        except Exception as e:
            errors.append(f"{file.filename}: {e}")

    return BatchUploadResponse(
        message=f"Batch complete: {len(results)} succeeded, {len(errors)} failed.",
        total_files=len(files),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )


@router.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def query_documents(
    request: QueryRequest,
    pipeline=Depends(get_pipeline),
):
    """
    Ask a question against the ingested document corpus.

    Returns the generated answer, cited sources (document + page), and
    response latency.
    """
    vector_store = get_vector_store()
    if vector_store.collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been ingested yet. Please upload a PDF first.",
        )

    response = pipeline.run(
        query=request.query,
        k=request.k,
        score_threshold=request.score_threshold,
        filter_doc_name=request.filter_doc_name,
    )

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        citations=[
            CitationModel(
                doc_name=c.doc_name,
                page_label=c.page_label,
                chunk_id=c.chunk_id,
            )
            for c in response.citations
        ],
        latency_seconds=response.latency_seconds,
        retrieved_chunk_count=len(response.retrieved_chunks),
    )


@router.post("/evaluate", response_model=EvaluationSummary, tags=["Evaluation"])
def evaluate(
    request: EvaluationRequest,
    evaluator=Depends(get_evaluator),
):
    """
    Run a batch of queries through the pipeline and return aggregate metrics.

    Metrics (retrieval relevance, faithfulness, hallucination risk, latency)
    are computed using lexical overlap — no additional LLM calls required.
    """
    report = evaluator.evaluate_batch(
        queries=request.queries,
        reference_answers=request.reference_answers,
        k=request.k,
    )
    return EvaluationSummary(**report.summary())
