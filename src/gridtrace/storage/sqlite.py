"""SQLite-based storage backend (no external dependencies).

Stores vectors as pickled BLOBs in a single table. Intended for
development and small-scale demos. For production use, switch to the
``PostgresBackend`` which uses ``pgvector`` for native cosine search.
"""

from __future__ import annotations

import pickle
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_anchors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    quant_key  TEXT    NOT NULL UNIQUE,
    vec        BLOB    NOT NULL,
    ref_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kb_entries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_id    INTEGER NOT NULL REFERENCES kb_anchors(id) ON DELETE CASCADE,
    question     TEXT    NOT NULL,
    solution     TEXT    NOT NULL,
    category     TEXT,
    doc_id       TEXT    NOT NULL,
    page_index   INTEGER NOT NULL DEFAULT 0,
    match_score  REAL    NOT NULL DEFAULT 0.0,
    embedding    BLOB    NOT NULL,
    UNIQUE (doc_id, page_index, question)
);

CREATE INDEX IF NOT EXISTS idx_kb_entries_anchor_id ON kb_entries (anchor_id);
CREATE INDEX IF NOT EXISTS idx_kb_entries_doc_id    ON kb_entries (doc_id);
"""


def _vec_to_blob(v: Iterable[float]) -> bytes:
    return np.asarray(list(v), dtype=np.float32).tobytes()


def _blob_to_vec(b: bytes) -> list[float]:
    return np.frombuffer(b, dtype=np.float32).astype(float).tolist()


class SqliteBackend(StorageBackend):
    """File-backed SQLite storage with NumPy vector search."""

    def __init__(self, path: str = "./data/gridtrace.db", dim: int = 512) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._dim = dim
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def _row_to_entry(self, row: sqlite3.Row) -> Entry:
        return Entry(
            entry_id=row["id"],
            anchor_id=row["anchor_id"],
            question=row["question"],
            solution=row["solution"],
            category=row["category"],
            doc_id=row["doc_id"],
            page_index=row["page_index"],
            match_score=row["match_score"],
            embedding=_blob_to_vec(row["embedding"]),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── StorageBackend API ────────────────────────────────
    def upsert_anchor(self, quant_key: str, anchor_vec: list[float]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM kb_anchors WHERE quant_key = ?", (quant_key,)
            )
            row = cur.fetchone()
            if row is not None:
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO kb_anchors (quant_key, vec) VALUES (?, ?)",
                (quant_key, _vec_to_blob(anchor_vec)),
            )
            return int(cur.lastrowid)

    def search_anchors(self, qvec: list[float], top_k: int) -> list[AnchorHit]:
        with self._lock:
            cur = self._conn.execute("SELECT id, quant_key, vec FROM kb_anchors")
            rows = cur.fetchall()
            if not rows:
                return []
            ids: list[int] = []
            keys: list[str] = []
            mat = np.zeros((len(rows), self._dim), dtype=np.float32)
            for i, (aid, key, blob) in enumerate(rows):
                ids.append(int(aid))
                keys.append(key)
                mat[i] = np.frombuffer(blob, dtype=np.float32)
            q = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
            sims = cosine_similarity(q, mat)[0]
            k = min(top_k, sims.shape[0])
            top_idx = np.argsort(-sims)[:k]
            return [
                AnchorHit(
                    anchor_id=ids[int(idx)],
                    quant_key=keys[int(idx)],
                    anchor_vec=mat[int(idx)].astype(float).tolist(),
                    score=float(sims[int(idx)]),
                )
                for idx in top_idx
            ]

    def fetch_entries_by_anchors(self, anchor_ids: list[int]) -> list[Entry]:
        if not anchor_ids:
            return []
        with self._lock:
            placeholders = ",".join("?" * len(anchor_ids))
            cur = self._conn.execute(
                f"SELECT * FROM kb_entries WHERE anchor_id IN ({placeholders})",
                anchor_ids,
            )
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def prune_orphan_anchors(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT a.id FROM kb_anchors a "
                "LEFT JOIN kb_entries e ON e.anchor_id = a.id "
                "WHERE e.id IS NULL"
            )
            ids = [int(r[0]) for r in cur.fetchall()]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM kb_anchors WHERE id IN ({placeholders})", ids
            )
            return len(ids)

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
            cur = self._conn.execute(
                "SELECT quant_key, ref_count FROM kb_anchors WHERE id = ?",
                (anchor_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"anchor_id={anchor_id} does not exist")
            quant_key, ref_count = row[0], int(row[1])
            cur = self._conn.execute(
                "INSERT INTO kb_entries "
                "(anchor_id, question, solution, category, doc_id, page_index, match_score, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    anchor_id,
                    question,
                    solution,
                    category,
                    doc_id,
                    page_index,
                    match_score,
                    _vec_to_blob(embedding),
                ),
            )
            entry_id = int(cur.lastrowid)
            self._conn.execute(
                "UPDATE kb_anchors SET ref_count = ref_count + 1 WHERE id = ?",
                (anchor_id,),
            )
            return IngestResult(
                entry_id=entry_id,
                anchor_id=anchor_id,
                quant_key=quant_key,
                match_score=match_score,
                reused_anchor=ref_count > 0,
            )

    def fetch_entry(self, entry_id: int) -> Entry | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM kb_entries WHERE id = ?", (entry_id,))
            row = cur.fetchone()
            return self._row_to_entry(row) if row else None

    def list_entries(self, limit: int = 50, offset: int = 0) -> list[Entry]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM kb_entries ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
            )
            return [self._row_to_entry(r) for r in cur.fetchall()]

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
            cur = self._conn.execute("SELECT * FROM kb_entries WHERE id = ?", (entry_id,))
            row = cur.fetchone()
            if row is None:
                return None
            fields: list[str] = []
            values: list = []
            if question is not None:
                fields.append("question = ?"); values.append(question)
            if solution is not None:
                fields.append("solution = ?"); values.append(solution)
            if category is not None:
                fields.append("category = ?"); values.append(category)
            if embedding is not None:
                fields.append("embedding = ?"); values.append(_vec_to_blob(embedding))
            if match_score is not None:
                fields.append("match_score = ?"); values.append(match_score)
            if not fields:
                return self._row_to_entry(row)
            values.append(entry_id)
            self._conn.execute(
                f"UPDATE kb_entries SET {', '.join(fields)} WHERE id = ?", values
            )
            cur = self._conn.execute("SELECT * FROM kb_entries WHERE id = ?", (entry_id,))
            return self._row_to_entry(cur.fetchone())

    def delete_by_doc(self, doc_id: str) -> UnlearnResult:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, anchor_id FROM kb_entries WHERE doc_id = ?", (doc_id,)
            )
            rows = cur.fetchall()
            if not rows:
                return UnlearnResult(doc_id=doc_id, entries_deleted=0, anchors_pruned=0)
            entry_ids = [int(r[0]) for r in rows]
            anchor_ids = sorted({int(r[1]) for r in rows})
            self._conn.execute("DELETE FROM kb_entries WHERE doc_id = ?", (doc_id,))
            for aid in anchor_ids:
                self._conn.execute(
                    "UPDATE kb_anchors SET ref_count = MAX(0, ref_count - "
                    "(SELECT COUNT(*) FROM kb_entries WHERE anchor_id = ? AND doc_id = ?)) "
                    "WHERE id = ?",
                    (aid, doc_id, aid),
                )
            pruned = self.prune_orphan_anchors()
            return UnlearnResult(
                doc_id=doc_id,
                entries_deleted=len(entry_ids),
                anchors_pruned=pruned,
            )

    def stats(self) -> KBStats:
        with self._lock:
            total_anchors = self._conn.execute(
                "SELECT COUNT(*) FROM kb_anchors"
            ).fetchone()[0]
            total_entries = self._conn.execute(
                "SELECT COUNT(*) FROM kb_entries"
            ).fetchone()[0]
            unique_docs = self._conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM kb_entries"
            ).fetchone()[0]
            avg = self._conn.execute(
                "SELECT COALESCE(AVG(match_score), 0) FROM kb_entries"
            ).fetchone()[0]
            orphans = self._conn.execute(
                "SELECT COUNT(*) FROM kb_anchors WHERE ref_count = 0"
            ).fetchone()[0]
            return KBStats(
                total_anchors=int(total_anchors),
                total_entries=int(total_entries),
                unique_docs=int(unique_docs),
                avg_match_score=float(avg),
                orphan_anchors=int(orphans),
            )

    def reset(self) -> None:
        with self._lock:
            self._conn.executescript(
                "DELETE FROM kb_entries; DELETE FROM kb_anchors; "
                "DELETE FROM sqlite_sequence WHERE name IN ('kb_entries', 'kb_anchors');"
            )
