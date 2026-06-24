"""GET /api/health — service health check."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_pipeline
from gridtrace.pipeline import GridTracePipeline

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(pipeline: GridTracePipeline = Depends(get_pipeline)) -> dict:
    try:
        stats = pipeline.stats()
        storage_ok = True
    except Exception as exc:  # pragma: no cover
        stats = None
        storage_ok = False
    return {
        "status": "ok" if storage_ok else "degraded",
        "version": "0.1.0",
        "storage_ok": storage_ok,
        "embedder": type(pipeline.embedder).__name__,
        "generator": type(pipeline.generator).__name__,
        "kb": (
            {
                "anchors": stats.total_anchors,
                "entries": stats.total_entries,
                "docs": stats.unique_docs,
                "avg_match_score": round(stats.avg_match_score, 4),
            }
            if stats
            else None
        ),
    }
