"""FastAPI application factory for GridTrace."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import chat, health, knowledge
from gridtrace.pipeline import GridTracePipeline
from gridtrace.utils.config import get_settings
from gridtrace.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

DEFAULT_FAQ_PATH = Path(__file__).resolve().parents[1] / "data" / "faq" / "OpsWarden_FAQ.md"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    logger.info("Starting GridTrace API (storage=%s)", settings.storage_backend)
    pipeline = GridTracePipeline.from_config(settings)
    app.state.pipeline = pipeline

    # auto-load FAQ if KB is empty
    if pipeline.storage.stats().total_entries == 0 and DEFAULT_FAQ_PATH.exists():
        try:
            from gridtrace.loaders import load_entries

            entries = load_entries(DEFAULT_FAQ_PATH)
            for e in entries:
                pipeline.ingest(
                    question=e.question,
                    solution=e.solution,
                    category=e.category,
                    doc_id=e.doc_id,
                    page_index=e.page_index,
                )
            logger.info("Auto-ingested %d entries from %s", len(entries), DEFAULT_FAQ_PATH.name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to auto-ingest FAQ: %s", exc)

    yield
    try:
        pipeline.close()
    except Exception:  # pragma: no cover
        pass
    logger.info("GridTrace API stopped.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="GridTrace",
        version="0.1.0",
        description="Quantized Unlearning-Aware Retrieval & Knowledge",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(knowledge.router)
    app.include_router(health.router)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "Internal Server Error", "data": None},
        )

    return app


app = create_app()
