"""Shared pytest fixtures."""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import pytest

# Ensure tests don't accidentally hit a real DB
os.environ.setdefault("GRIDTRACE_STORAGE_BACKEND", "memory")
os.environ.setdefault("EMBEDDING_DIM", "64")

from gridtrace.embedder import HashingEmbedder  # noqa: E402
from gridtrace.pipeline import GridTracePipeline  # noqa: E402
from gridtrace.storage import MemoryBackend  # noqa: E402
from gridtrace.utils.config import GridTraceConfig  # noqa: E402


@pytest.fixture
def dim() -> int:
    return 64


@pytest.fixture
def embedder(dim: int) -> HashingEmbedder:
    return HashingEmbedder(dim=dim)


@pytest.fixture
def backend(dim: int) -> Iterator[MemoryBackend]:
    be = MemoryBackend(dim=dim)
    yield be
    be.close()


@pytest.fixture
def config(dim: int) -> GridTraceConfig:
    return GridTraceConfig(
        storage_backend="memory",
        epsilon=0.05,
        anchor_top_k=4,
        score_threshold=0.0,
        top_k=3,
        embedding_dim=dim,
    )


@pytest.fixture
def pipeline(embedder: HashingEmbedder, config: GridTraceConfig) -> Iterator[GridTracePipeline]:
    from gridtrace.storage.memory import MemoryBackend as MB
    from gridtrace.generator import EchoGenerator

    p = GridTracePipeline(MB(dim=config.embedding_dim), embedder, EchoGenerator(), config)
    yield p
    p.close()


@pytest.fixture
def random_unit_vec(dim: int) -> np.ndarray:
    v = np.random.default_rng(0).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)
