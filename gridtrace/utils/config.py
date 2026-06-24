"""Pydantic-Settings configuration for GridTrace.

The ``GridTraceConfig`` object centralizes *all* tunable parameters and is
injected into every component. Use ``get_settings()`` to obtain a cached
singleton from environment variables / .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GridTraceConfig(BaseSettings):
    """Top-level GridTrace configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Storage ────────────────────────────────────────────
    storage_backend: Literal["postgres", "sqlite", "memory"] = Field(
        default="postgres",
        alias="GRIDTRACE_STORAGE_BACKEND",
    )
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/gridtrace",
        alias="DATABASE_URL",
    )
    sqlite_path: str = Field(
        default="./data/gridtrace.db",
        alias="GRIDTRACE_SQLITE_PATH",
    )

    # ── Quantization ───────────────────────────────────────
    epsilon: float = Field(default=0.02, alias="ANCHOR_QUANT_EPSILON", ge=1e-6)
    embedding_dim: int = Field(default=512, alias="EMBEDDING_DIM", ge=16)

    # ── Retrieval ──────────────────────────────────────────
    anchor_top_k: int = Field(default=8, alias="RAG_ANCHOR_TOP_K", ge=1)
    score_threshold: float = Field(
        default=0.65, alias="RAG_SCORE_THRESHOLD", ge=-1.0, le=1.0
    )
    top_k: int = Field(default=3, alias="RAG_TOP_K", ge=1)
    fetch_k: int = Field(default=20, alias="RAG_FETCH_K", ge=1)

    # ── Embedding ──────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_local_path: str | None = Field(
        default=None, alias="EMBEDDING_LOCAL_PATH"
    )

    # ── LLM ────────────────────────────────────────────────
    llm_provider: Literal["echo", "deepseek", "openai", "ollama"] = Field(
        default="echo", alias="LLM_PROVIDER"
    )
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1", alias="LLM_BASE_URL"
    )
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS", ge=1)
    llm_ollama_url: str = Field(
        default="http://localhost:11434", alias="LLM_OLLAMA_URL"
    )

    # ── API ────────────────────────────────────────────────
    api_cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="API_CORS_ORIGINS",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> GridTraceConfig:
    """Cached settings singleton (env-driven)."""
    return GridTraceConfig()  # type: ignore[call-arg]
