"""PostgreSQL + pgvector storage backend.

This is the production-recommended backend. ``pgvector`` provides native
cosine-distance operators (``<=>``) and HNSW/IVFFlat indexes (we use
cosine without index for now; index can be added with a migration when
the anchor table grows large).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from gridtrace.embedder import cosine_similarity
from gridtrace.storage._types import (
    AnchorHit,
    Entry,
    IngestResult,
    KBStats,
    StorageBackend,
    UnlearnResult,
)
from gridtrace.utils.logging import get_logger

logger = get_logger(__name__)


class PostgresBackend(StorageBackend):
    """Postgres + pgvector backend. Requires the schema in ``docker/init.sql``."""

    def __init__(self, database_url: str, dim: int = 512) -> None:
        from sqlalchemy import create_engine, text  # type: ignore

        self._engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self._dim = dim
        self._lock = threading.RLock()
        self._text = text  # keep import binding

    def close(self) -> None:
        with self._lock:
            self._engine.dispose()

    # ── helpers ───────────────────────────────────────────
    def _vec_literal(self, v: list[float]) -> str:
        """Format a Python list[float] as a pgvector literal: '[1,2,3]'."""
        return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

    def _row_to_entry(self, row: Any) -> Entry:
        embedding_raw = row.embedding
        if isinstance(embedding_raw, str):
            # pgvector returned as text '[a,b,c]'
            inner = embedding_raw.strip("[]")
            embedding = [float(x) for x in inner.split(",")] if inner else []
        else:
            embedding = list(embedding_raw)
        return Entry(
            entry_id=int(row.id),
            anchor_id=int(row.anchor_id),
            question=row.question,
            solution=row.solution,
            category=row.category,
            doc_id=row.doc_id,
            page_index=int(row.page_index),
            match_score=float(row.match_score),
            embedding=embedding,
        )

    # ── StorageBackend API ────────────────────────────────
    def upsert_anchor(self, quant_key: str, anchor_vec: list[float]) -> int:
        with self._lock, self._engine.begin() as conn:
            row = conn.execute(
                self._text("SELECT id FROM kb_anchors WHERE quant_key = :k"),
                {"k": quant_key},
            ).fetchone()
            if row is not None:
                return int(row[0])
            res = conn.execute(
                self._text(
                    "INSERT INTO kb_anchors (quant_key, anchor_vec) "
                    "VALUES (:k, CAST(:v AS vector)) RETURNING id"
                ),
                {"k": quant_key, "v": self._vec_literal(anchor_vec)},
            )
            return int(res.scalar_one())

    def search_anchors(self, qvec: list[float], top_k: int) -> list[AnchorHit]:
        with self._lock, self._engine.connect() as conn:
            # pgvector cosine distance operator: <=>
            # Convert distance → similarity: 1 - distance
            rows = conn.execute(
                self._text(
                    "SELECT id, quant_key, anchor_vec, "
                    "  1 - (anchor_vec <=> CAST(:q AS vector)) AS score "
                    "FROM kb_anchors "
                    "ORDER BY anchor_vec <=> CAST(:q AS vector) "
                    "LIMIT :k"
                ),
                {"q": self._vec_literal(qvec), "k": top_k},
            ).fetchall()
            hits: list[AnchorHit] = []
            for r in rows:
                vec_raw = r.anchor_vec
                if isinstance(vec_raw, str):
                    inner = vec_raw.strip("[]")
                    vec = [float(x) for x in inner.split(",")] if inner else []
                else:
                    vec = list(vec_raw)
                hits.append(
                    AnchorHit(
                        anchor_id=int(r.id),
                        quant_key=r.quant_key,
                        anchor_vec=vec,
                        score=float(r.score),
                    )
                )
            return hits

    def fetch_entries_by_anchors(self, anchor_ids: list[int]) -> list[Entry]:
        if not anchor_ids:
            return []
        with self._lock, self._engine.connect() as conn:
            rows = conn.execute(
                self._text(
                    "SELECT id, anchor_id, question, solution, category, "
                    "       doc_id, page_index, match_score, embedding "
                    "FROM kb_entries WHERE anchor_id = ANY(:a)"
                ),
                {"a": anchor_ids},
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def prune_orphan_anchors(self) -> int:
        with self._lock, self._engine.begin() as conn:
            res = conn.execute(
                self._text(
                    "DELETE FROM kb_anchors a "
                    "WHERE NOT EXISTS (SELECT 1 FROM kb_entries e WHERE e.anchor_id = a.id)"
                )
            )
            return int(res.rowcount or 0)

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
        with self._lock, self._engine.begin() as conn:
            row = conn.execute(
                self._text(
                    "SELECT quant_key, ref_count FROM kb_anchors WHERE id = :a FOR UPDATE"
                ),
                {"a": anchor_id},
            ).fetchone()
            if row is None:
                raise KeyError(f"anchor_id={anchor_id} does not exist")
            quant_key, ref_count = row[0], int(row[1])
            res = conn.execute(
                self._text(
                    "INSERT INTO kb_entries "
                    "(anchor_id, question, solution, category, doc_id, "
                    " page_index, match_score, embedding) "
                    "VALUES (:a, :q, :s, :c, :d, :p, :m, CAST(:e AS vector)) "
                    "RETURNING id"
                ),
                {
                    "a": anchor_id,
                    "q": question,
                    "s": solution,
                    "c": category,
                    "d": doc_id,
                    "p": page_index,
                    "m": match_score,
                    "e": self._vec_literal(embedding),
                },
            )
            entry_id = int(res.scalar_one())
            return IngestResult(
                entry_id=entry_id,
                anchor_id=anchor_id,
                quant_key=quant_key,
                match_score=match_score,
                reused_anchor=ref_count > 0,
            )

    def fetch_entry(self, entry_id: int) -> Entry | None:
        with self._lock, self._engine.connect() as conn:
            row = conn.execute(
                self._text(
                    "SELECT id, anchor_id, question, solution, category, "
                    "       doc_id, page_index, match_score, embedding "
                    "FROM kb_entries WHERE id = :i"
                ),
                {"i": entry_id},
            ).fetchone()
            return self._row_to_entry(row) if row else None

    def list_entries(self, limit: int = 50, offset: int = 0) -> list[Entry]:
        with self._lock, self._engine.connect() as conn:
            rows = conn.execute(
                self._text(
                    "SELECT id, anchor_id, question, solution, category, "
                    "       doc_id, page_index, match_score, embedding "
                    "FROM kb_entries ORDER BY id LIMIT :l OFFSET :o"
                ),
                {"l": limit, "o": offset},
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

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
        sets: list[str] = []
        params: dict[str, Any] = {"i": entry_id}
        if question is not None:
            sets.append("question = :q"); params["q"] = question
        if solution is not None:
            sets.append("solution = :s"); params["s"] = solution
        if category is not None:
            sets.append("category = :c"); params["c"] = category
        if match_score is not None:
            sets.append("match_score = :m"); params["m"] = match_score
        if embedding is not None:
            sets.append("embedding = CAST(:e AS vector)")
            params["e"] = self._vec_literal(embedding)
        if not sets:
            return self.fetch_entry(entry_id)
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                self._text(
                    f"UPDATE kb_entries SET {', '.join(sets)} WHERE id = :i"
                ),
                params,
            )
        return self.fetch_entry(entry_id)

    def delete_by_doc(self, doc_id: str) -> UnlearnResult:
        with self._lock, self._engine.begin() as conn:
            count_row = conn.execute(
                self._text("SELECT COUNT(*) FROM kb_entries WHERE doc_id = :d"),
                {"d": doc_id},
            ).fetchone()
            entries_deleted = int(count_row[0]) if count_row else 0
            if entries_deleted == 0:
                return UnlearnResult(doc_id=doc_id, entries_deleted=0, anchors_pruned=0)
            conn.execute(
                self._text("DELETE FROM kb_entries WHERE doc_id = :d"), {"d": doc_id}
            )
            # ref_count trigger already decrements; prune orphans now
            res = conn.execute(
                self._text(
                    "DELETE FROM kb_anchors a "
                    "WHERE NOT EXISTS (SELECT 1 FROM kb_entries e WHERE e.anchor_id = a.id)"
                )
            )
            pruned = int(res.rowcount or 0)
            conn.execute(
                self._text(
                    "INSERT INTO kb_unlearn_log (doc_id, entries_deleted, anchors_pruned) "
                    "VALUES (:d, :e, :p)"
                ),
                {"d": doc_id, "e": entries_deleted, "p": pruned},
            )
            return UnlearnResult(
                doc_id=doc_id,
                entries_deleted=entries_deleted,
                anchors_pruned=pruned,
            )

    def stats(self) -> KBStats:
        with self._lock, self._engine.connect() as conn:
            row = conn.execute(
                self._text(
                    "SELECT "
                    "  (SELECT COUNT(*) FROM kb_anchors) AS total_anchors, "
                    "  (SELECT COUNT(*) FROM kb_entries) AS total_entries, "
                    "  (SELECT COUNT(DISTINCT doc_id) FROM kb_entries) AS unique_docs, "
                    "  COALESCE((SELECT AVG(match_score) FROM kb_entries), 0) AS avg_ms, "
                    "  (SELECT COUNT(*) FROM kb_anchors WHERE ref_count = 0) AS orphans"
                )
            ).fetchone()
            return KBStats(
                total_anchors=int(row.total_anchors),
                total_entries=int(row.total_entries),
                unique_docs=int(row.unique_docs),
                avg_match_score=float(row.avg_ms),
                orphan_anchors=int(row.orphans),
            )

    def reset(self) -> None:
        with self._lock, self._engine.begin() as conn:
            conn.execute(self._text("TRUNCATE kb_entries, kb_anchors RESTART IDENTITY CASCADE"))
