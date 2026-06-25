"""GridTrace — Quantized Unlearning-Aware Retrieval & Knowledge.

Top-level package re-exports the public API so consumers can write
``from gridtrace import GridTracePipeline, GridTraceConfig`` etc.
"""

from __future__ import annotations

from gridtrace.embedder import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from gridtrace.generator import DeepSeekGenerator, EchoGenerator, Generator, OpenAIGenerator
from gridtrace.pipeline import GridTracePipeline
from gridtrace.quantizer import quantize_vector, quant_key_from_vector
from gridtrace.retriever import Retriever
from gridtrace.storage import (
    AnchorHit,
    Entry,
    IngestResult,
    MemoryBackend,
    SearchHit,
    SqliteBackend,
    StorageBackend,
    UnlearnResult,
)
from gridtrace.utils.config import GridTraceConfig, get_settings

__version__ = "0.1.0"

__all__ = [
    "AnchorHit",
    "DeepSeekGenerator",
    "EchoGenerator",
    "Embedder",
    "Entry",
    "Generator",
    "HashingEmbedder",
    "IngestResult",
    "MemoryBackend",
    "OpenAIGenerator",
    "GridTraceConfig",
    "GridTracePipeline",
    "Retriever",
    "SearchHit",
    "SentenceTransformerEmbedder",
    "SqliteBackend",
    "StorageBackend",
    "UnlearnResult",
    "get_settings",
    "quant_key_from_vector",
    "quantize_vector",
]
