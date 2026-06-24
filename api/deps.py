"""FastAPI dependencies (DI)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from gridtrace.pipeline import GridTracePipeline
from gridtrace.utils.config import GridTraceConfig, get_settings


@lru_cache(maxsize=1)
def _pipeline_singleton() -> GridTracePipeline:
    """Build a single shared pipeline from settings."""
    cfg = get_settings()
    return GridTracePipeline.from_config(cfg)


def get_settings_dep() -> GridTraceConfig:
    return get_settings()


def get_pipeline(request: Request) -> GridTracePipeline:
    """Resolve pipeline from app.state (preferred) or build a singleton."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        return pipeline
    return _pipeline_singleton()
