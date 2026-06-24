"""Dataclasses and the storage ABC shared across all storage backends.

This module is intentionally separate from ``gridtrace/storage/__init__.py``
to break the circular import: ``memory.py`` / ``sqlite.py`` /
``postgres.py`` need these types and the ABC but must not trigger a
re-import of the storage package while it is still being initialized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = [
    "AnchorHit",
    "Entry",
    "IngestResult",
    "KBStats",
    "SearchHit",
    "StorageBackend",
    "UnlearnResult",
]


# ── Data classes ──────────────────────────────────────────


@dataclass(slots=True)
class AnchorHit:
    """A row hit by the L1 anchor search."""

    anchor_id: int
    quant_key: str
    anchor_vec: list[float]
    score: float


@dataclass(slots=True)
class Entry:
    """A full-precision entry (KB record)."""

    entry_id: int
    anchor_id: int
    question: str
    solution: str
    category: str | None
    doc_id: str
    page_index: int
    match_score: float
    embedding: list[float]


@dataclass(slots=True)
class SearchHit:
    """A search result returned to the user."""

    entry_id: int
    anchor_id: int
    question: str
    solution: str
    category: str | None
    doc_id: str
    page_index: int
    match_score: float
    score: float          # L2 cosine


@dataclass(slots=True)
class IngestResult:
    """Returned by :meth:`StorageBackend.ingest`."""

    entry_id: int
    anchor_id: int
    quant_key: str
    match_score: float
    reused_anchor: bool


@dataclass(slots=True)
class UnlearnResult:
    """Returned by :meth:`StorageBackend.delete_by_doc`."""

    doc_id: str
    entries_deleted: int
    anchors_pruned: int


@dataclass(slots=True)
class KBStats:
    """Lightweight KB statistics."""

    total_anchors: int = 0
    total_entries: int = 0
    unique_docs: int = 0
    avg_match_score: float = 0.0
    orphan_anchors: int = 0
    extra: dict[str, int] = field(default_factory=dict)


# ── Abstract interface ────────────────────────────────────


class StorageBackend(ABC):
    """Abstract storage interface.

    All methods are synchronous. The ABC defines the *minimum* set of
    operations needed by the GridTrace pipeline. Implementations may extend
    with backend-specific helpers (see ``postgres.py``).
    """

    # ── Anchor management ────────────────────────────────
    @abstractmethod
    def upsert_anchor(self, quant_key: str, anchor_vec: list[float]) -> int:
        """Get-or-create anchor by quant_key. Returns anchor_id."""

    @abstractmethod
    def search_anchors(self, qvec: list[float], top_k: int) -> list[AnchorHit]:
        """L1: cosine-similarity search over the anchor table."""

    @abstractmethod
    def fetch_entries_by_anchors(self, anchor_ids: list[int]) -> list[Entry]:
        """L2 pool: fetch all entries whose anchor is in ``anchor_ids``."""

    @abstractmethod
    def prune_orphan_anchors(self) -> int:
        """Remove anchors with no referencing entries. Returns count."""

    # ── Entry management ─────────────────────────────────
    @abstractmethod
    def ingest_entry(
        self,
        embedding: list[float],
        anchor_id: int,
        question: str,
        solution: str,
        category: str | None,
        doc_id: str,
        page_index: int,
        match_score: float,
    ) -> IngestResult: ...

    @abstractmethod
    def fetch_entry(self, entry_id: int) -> Entry | None: ...

    @abstractmethod
    def list_entries(self, limit: int = 50, offset: int = 0) -> list[Entry]: ...

    @abstractmethod
    def update_entry(
        self,
        entry_id: int,
        *,
        question: str | None = None,
        solution: str | None = None,
        category: str | None = None,
        embedding: list[float] | None = None,
        match_score: float | None = None,
    ) -> Entry | None: ...

    # ── Unlearning ───────────────────────────────────────
    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> UnlearnResult: ...

    # ── Stats / lifecycle ────────────────────────────────
    @abstractmethod
    def stats(self) -> KBStats: ...

    @abstractmethod
    def reset(self) -> None:
        """DESTRUCTIVE: drop all KB data."""

    def close(self) -> None:        # optional, not abstract
        return None
