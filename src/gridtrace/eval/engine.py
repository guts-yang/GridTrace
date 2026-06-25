"""Evaluation engine — re-implements the two-phase retrieval in-memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridtrace.embedder import Embedder, cosine_similarity
from gridtrace.quantizer import quantize_vector
from gridtrace.storage import IngestResult
from gridtrace.storage.memory import MemoryBackend
from gridtrace.utils.config import GridTraceConfig
from gridtrace.utils.hashing import quant_key_from_vector
from gridtrace.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EvalPrediction:
    """One prediction in an eval run."""

    query: str
    expected_doc_id: str
    expected_page_index: int
    predicted_doc_ids: list[str]
    predicted_page_indexes: list[int]
    l1_anchor_doc_ids: list[str]        # doc_ids hit by L1 anchors (debug)


class EvalEngine:
    """Drop-in eval engine using an in-memory backend."""

    def __init__(
        self,
        embedder: Embedder,
        config: GridTraceConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._config = config or GridTraceConfig()
        self._storage = MemoryBackend(dim=embedder.dim)

    @property
    def storage(self) -> MemoryBackend:
        return self._storage

    def ingest(
        self,
        question: str,
        solution: str,
        *,
        doc_id: str,
        page_index: int,
        category: str | None = None,
    ) -> IngestResult:
        qvec, svec = self._embedder.encode([question, solution], kind="document")
        from gridtrace.embedder import cosine, l2_normalize

        joint = l2_normalize(((qvec + svec) / 2.0).astype(np.float32))
        match_score = float(cosine(qvec, svec))
        anchor_vec = quantize_vector(joint, epsilon=self._config.epsilon)
        quant_key = quant_key_from_vector(anchor_vec)
        anchor_id = self._storage.upsert_anchor(quant_key, anchor_vec.tolist())
        return self._storage.ingest_entry(
            embedding=joint.tolist(),
            anchor_id=anchor_id,
            question=question,
            solution=solution,
            category=category,
            doc_id=doc_id,
            page_index=page_index,
            match_score=match_score,
        )

    def search_l1(
        self,
        query: str,
        anchor_top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        k = anchor_top_k or self._config.anchor_top_k
        qvec = self._embedder.encode([query], kind="query")[0]
        hits = self._storage.search_anchors(qvec.astype(float).tolist(), k)
        return [(h.anchor_id, h.score) for h in hits]

    def predict(
        self,
        query: str,
        expected_doc_id: str,
        expected_page_index: int,
        *,
        top_k: int | None = None,
        anchor_top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> EvalPrediction:
        from gridtrace.retriever import Retriever

        r = Retriever(self._storage, self._embedder, self._config)
        hits = r.search(
            query,
            top_k=top_k,
            anchor_top_k=anchor_top_k,
            score_threshold=score_threshold,
        )
        l1 = self.search_l1(query, anchor_top_k)
        l1_anchor_ids = {a for a, _ in l1}
        l1_entries = self._storage.fetch_entries_by_anchors(list(l1_anchor_ids))
        l1_doc_ids = sorted({e.doc_id for e in l1_entries})
        return EvalPrediction(
            query=query,
            expected_doc_id=expected_doc_id,
            expected_page_index=expected_page_index,
            predicted_doc_ids=[h.doc_id for h in hits],
            predicted_page_indexes=[h.page_index for h in hits],
            l1_anchor_doc_ids=l1_doc_ids,
        )
