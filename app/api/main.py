from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    # Pre-warm singletons so first request is fast
    from app.api.dependencies import get_embedding_manager, get_vector_store

    get_embedding_manager()
    get_vector_store()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="DocuRAG API",
        description=(
            "Production-grade Retrieval-Augmented Generation for PDF documents. "
            "Upload PDFs, ask questions, and receive cited answers."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
