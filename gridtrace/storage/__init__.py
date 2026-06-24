"""Pluggable storage backends for GridTrace.

Every backend implements the :class:`StorageBackend` ABC. The pipeline
only depends on this interface, so swapping PostgreSQL ↔ SQLite ↔
In-memory is a one-line config change.
"""

from __future__ import annotations

from gridtrace.storage._types import (
    AnchorHit,
    Entry,
    IngestResult,
    KBStats,
    SearchHit,
    StorageBackend,
    UnlearnResult,
)
from gridtrace.storage.memory import MemoryBackend
from gridtrace.storage.sqlite import SqliteBackend

__all__ = [
    "AnchorHit",
    "Entry",
    "IngestResult",
    "KBStats",
    "MemoryBackend",
    "SearchHit",
    "SqliteBackend",
    "StorageBackend",
    "UnlearnResult",
]
