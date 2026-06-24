"""Pure in-memory storage backend.

Used by unit tests, the evaluation engine, and the ``memory`` config
profile. The implementation mirrors the SQL backend's behavior exactly
(including ref_count semantics) so a test passing here will pass
against PostgreSQL.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Sequence

import numpy as np

from gridtrace.embedder import cosine_similarity
from gridtrace.storage._types import (
    AnchorHit,
    Entry,
    IngestResult,
    KBStats,
    SearchHit,
    StorageBackend,
    UnlearnResult,
)
from gridtrace.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryBackend(StorageBackend):
    """Thread-safe in-memory storage."""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim
        self._lock = threading.RLock()

        self._next_anchor_id: int = 1
        self._next_entry_id: int = 1

        # quant_key -> anchor_id
        self._anchor_by_key: dict[str, int] = {}
        # anchor_id -> {"vec": np.ndarray, "key": str, "ref_count": int}
        self._anchors: dict[int, dict] = {}
        # entry_id -> Entry
        self._entries: dict[int, Entry] = {}
        # doc_id -> list[entry_id]
        self._entries_by_doc: dict[str, list[int]] = defaultdict(list)
        # anchor_id -> list[entry_id]
        self._entries_by_anchor: dict[int, list[int]] = defaultdict(list)
        # Pre-built anchor matrix for fast L1 search
        self._anchor_matrix_dirty: bool = True
        self._anchor_matrix_ids: list[int] = []
        self._anchor_matrix_vecs: np.ndarray | None = None

    # ── helpers ───────────────────────────────────────────
    def _invalidate_anchor_cache(self) -> None:
        self._anchor_matrix_dirty = True

    def _rebuild_anchor_cache(self) -> None:
        ids = sorted(self._anchors.keys())
        if ids:
            mat = np.stack([self._anchors[i]["vec"] for i in ids]).astype(np.float32)
        else:
            mat = np.zeros((0, self._dim), dtype=np.float32)
        self._anchor_matrix_ids = ids
        self._anchor_matrix_vecs = mat
        self._anchor_matrix_dirty = False

    def _to_list(self, arr: np.ndarray) -> list[float]:
        return arr.astype(float).tolist()

    # ── StorageBackend API ───────────────────────────────
    def upsert_anchor(self, quant_key: str, anchor_vec: list[float]) -> int:
        with self._lock:
            existing = self._anchor_by_key.get(quant_key)
            if existing is not None:
                return existing
            anchor_id = self._next_anchor_id
            self._next_anchor_id += 1
            arr = np.asarray(anchor_vec, dtype=np.float32)
            self._anchors[anchor_id] = {
                "vec": arr,
                "key": quant_key,
                "ref_count": 0,
            }
            self._anchor_by_key[quant_key] = anchor_id
            self._invalidate_anchor_cache()
            logger.debug("upsert_anchor: created id=%s key=%s…", anchor_id, quant_key[:8])
            return anchor_id

    def search_anchors(self, qvec: list[float], top_k: int) -> list[AnchorHit]:
        with self._lock:
            if self._anchor_matrix_dirty:
                self._rebuild_anchor_cache()
            assert self._anchor_matrix_vecs is not None
            if self._anchor_matrix_vecs.shape[0] == 0:
                return []
            q = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
            sims = cosine_similarity(q, self._anchor_matrix_vecs)[0]
            k = min(top_k, sims.shape[0])
            # argpartition is O(N); full sort is fine for small N
            top_idx = np.argsort(-sims)[:k]
            hits: list[AnchorHit] = []
            for idx in top_idx:
                anchor_id = self._anchor_matrix_ids[int(idx)]
                meta = self._anchors[anchor_id]
                hits.append(
                    AnchorHit(
                        anchor_id=anchor_id,
                        quant_key=meta["key"],
                        anchor_vec=self._to_list(meta["vec"]),
                        score=float(sims[int(idx)]),
                    )
                )
            return hits

    def fetch_entries_by_anchors(self, anchor_ids: list[int]) -> list[Entry]:
        with self._lock:
            ids_set = set(anchor_ids)
            out: list[Entry] = []
            for eid, entry in self._entries.items():
                if entry.anchor_id in ids_set:
                    out.append(entry)
            return out

    def prune_orphan_anchors(self) -> int:
        with self._lock:
            orphans = [
                aid
                for aid, meta in self._anchors.items()
                if meta["ref_count"] <= 0
                and not self._entries_by_anchor.get(aid)
            ]
            for aid in orphans:
                meta = self._anchors.pop(aid, None)
                if meta is None:
                    continue
                self._anchor_by_key.pop(meta["key"], None)
                self._entries_by_anchor.pop(aid, None)
            if orphans:
                self._invalidate_anchor_cache()
            return len(orphans)

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
    ) -> IngestResult:
        with self._lock:
            if anchor_id not in self._anchors:
                raise KeyError(f"anchor_id={anchor_id} does not exist")
            entry_id = self._next_entry_id
            self._next_entry_id += 1
            entry = Entry(
                entry_id=entry_id,
                anchor_id=anchor_id,
                question=question,
                solution=solution,
                category=category,
                doc_id=doc_id,
                page_index=page_index,
                match_score=match_score,
                embedding=list(embedding),
            )
            self._entries[entry_id] = entry
            self._entries_by_doc[doc_id].append(entry_id)
            self._entries_by_anchor[anchor_id].append(entry_id)
            self._anchors[anchor_id]["ref_count"] += 1
            return IngestResult(
                entry_id=entry_id,
                anchor_id=anchor_id,
                quant_key=self._anchors[anchor_id]["key"],
                match_score=match_score,
                reused_anchor=self._anchors[anchor_id]["ref_count"] > 1,
            )

    def fetch_entry(self, entry_id: int) -> Entry | None:
        with self._lock:
            return self._entries.get(entry_id)

    def list_entries(self, limit: int = 50, offset: int = 0) -> list[Entry]:
        with self._lock:
            ids = sorted(self._entries.keys())
            sliced = ids[offset : offset + limit]
            return [self._entries[i] for i in sliced]

    def update_entry(
        self,
        entry_id: int,
        *,
        question: str | None = None,
        solution: str | None = None,
        category: str | None = None,
        embedding: list[float] | None = None,
        match_score: float | None = None,
    ) -> Entry | None:
        with self._lock:
            e = self._entries.get(entry_id)
            if e is None:
                return None
            if question is not None:
                e.question = question
            if solution is not None:
                e.solution = solution
            if category is not None:
                e.category = category
            if embedding is not None:
                e.embedding = list(embedding)
            if match_score is not None:
                e.match_score = match_score
            return e

    def delete_by_doc(self, doc_id: str) -> UnlearnResult:
        with self._lock:
            entry_ids = list(self._entries_by_doc.get(doc_id, []))
            if not entry_ids:
                return UnlearnResult(doc_id=doc_id, entries_deleted=0, anchors_pruned=0)
            affected_anchors: set[int] = set()
            for eid in entry_ids:
                e = self._entries.pop(eid, None)
                if e is None:
                    continue
                affected_anchors.add(e.anchor_id)
                if eid in self._entries_by_anchor.get(e.anchor_id, []):
                    self._entries_by_anchor[e.anchor_id].remove(eid)
                if self._anchors.get(e.anchor_id):
                    self._anchors[e.anchor_id]["ref_count"] = max(
                        0, self._anchors[e.anchor_id]["ref_count"] - 1
                    )
            self._entries_by_doc.pop(doc_id, None)
            pruned = self.prune_orphan_anchors()
            return UnlearnResult(
                doc_id=doc_id,
                entries_deleted=len(entry_ids),
                anchors_pruned=pruned,
            )

    def stats(self) -> KBStats:
        with self._lock:
            total = len(self._entries)
            anchors = len(self._anchors)
            docs = len(self._entries_by_doc)
            ms = (
                sum(e.match_score for e in self._entries.values()) / total
                if total
                else 0.0
            )
            orphans = sum(
                1 for a in self._anchors.values() if a["ref_count"] <= 0
            )
            return KBStats(
                total_anchors=anchors,
                total_entries=total,
                unique_docs=docs,
                avg_match_score=ms,
                orphan_anchors=orphans,
            )

    def reset(self) -> None:
        with self._lock:
            self._anchors.clear()
            self._anchor_by_key.clear()
            self._entries.clear()
            self._entries_by_doc.clear()
            self._entries_by_anchor.clear()
            self._next_anchor_id = 1
            self._next_entry_id = 1
            self._invalidate_anchor_cache()
