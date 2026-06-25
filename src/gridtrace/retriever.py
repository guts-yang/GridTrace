"""Two-phase retriever.

Phase 1 (L1) — cosine over the *anchor* table.  The query vector is not
quantized; we keep the full-precision query and rank anchors with the
same cosine similarity that pgvector / numpy / our own helpers use.

Phase 2 (L2) — full-precision cosine over candidate entries gathered
from the Top-K anchors. Threshold filtering happens here. We then
return the top ``RAG_TOP_K`` results.
"""

from __future__ import annotations

import numpy as np

from gridtrace.embedder import Embedder, cosine_similarity
from gridtrace.storage._types import Entry, SearchHit, StorageBackend
from gridtrace.utils.config import GridTraceConfig
from gridtrace.utils.logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """L1 anchor routing → L2 entry rerank."""

    def __init__(
        self,
        storage: StorageBackend,
        embedder: Embedder,
        config: GridTraceConfig | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._config = config or GridTraceConfig()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        anchor_top_k: int | None = None,
        score_threshold: float | None = None,
        fetch_k: int | None = None,
    ) -> list[SearchHit]:
        qvec = self._embedder.encode([query], kind="query")[0]
        return self.search_with_vector(
            qvec,
            top_k=top_k,
            anchor_top_k=anchor_top_k,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
        )

    def search_with_vector(
        self,
        qvec: np.ndarray,
        *,
        top_k: int | None = None,
        anchor_top_k: int | None = None,
        score_threshold: float | None = None,
        fetch_k: int | None = None,
    ) -> list[SearchHit]:
        k_final = top_k or self._config.top_k
        k_anchor = anchor_top_k or self._config.anchor_top_k
        thr = (
            score_threshold
            if score_threshold is not None
            else self._config.score_threshold
        )
        fk = fetch_k or self._config.fetch_k

        # ── L1: anchor routing ─────────────────────────────────
        anchor_hits = self._storage.search_anchors(qvec.astype(float).tolist(), k_anchor)
        if not anchor_hits:
            return []
        anchor_ids = [h.anchor_id for h in anchor_hits]
        logger.debug("L1 routed %d anchors (top score=%.4f)", len(anchor_hits), anchor_hits[0].score)

        # ── L2: full-precision rerank ──────────────────────────
        candidates: list[Entry] = self._storage.fetch_entries_by_anchors(anchor_ids)
        if not candidates:
            return []
        cand_mat = np.stack([np.asarray(e.embedding, dtype=np.float32) for e in candidates])
        q = qvec.astype(np.float32).reshape(1, -1)
        sims = cosine_similarity(q, cand_mat)[0]

        order = np.argsort(-sims)
        hits: list[SearchHit] = []
        for idx in order:
            score = float(sims[int(idx)])
            if score < thr:
                break
            e = candidates[int(idx)]
            hits.append(
                SearchHit(
                    entry_id=e.entry_id,
                    anchor_id=e.anchor_id,
                    question=e.question,
                    solution=e.solution,
                    category=e.category,
                    doc_id=e.doc_id,
                    page_index=e.page_index,
                    match_score=e.match_score,
                    score=score,
                )
            )
            if len(hits) >= fk:
                break
        return hits[:k_final]
