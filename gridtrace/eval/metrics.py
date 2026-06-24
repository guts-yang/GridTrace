"""Evaluation metrics for GridTrace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .engine import EvalPrediction


@dataclass(slots=True)
class EvalMetrics:
    """Aggregate metrics over a batch of predictions."""

    total: int = 0
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    mrr_sum: float = 0.0
    l1_hit: int = 0
    avg_score: float = 0.0
    score_sum: float = 0.0
    score_count: int = 0
    per_query: list[dict] = field(default_factory=list)

    def merge(self, other: "EvalMetrics") -> "EvalMetrics":
        self.total += other.total
        self.hit_at_1 += other.hit_at_1
        self.hit_at_3 += other.hit_at_3
        self.hit_at_5 += other.hit_at_5
        self.mrr_sum += other.mrr_sum
        self.l1_hit += other.l1_hit
        self.score_sum += other.score_sum
        self.score_count += other.score_count
        self.per_query.extend(other.per_query)
        return self

    def finalize(self) -> "EvalMetrics":
        if self.total == 0:
            return self
        self.avg_score = self.score_sum / self.score_count if self.score_count else 0.0
        return self

    def as_dict(self) -> dict:
        if self.total == 0:
            return {"total": 0}
        return {
            "total": self.total,
            "hit@1": round(self.hit_at_1 / self.total, 4),
            "hit@3": round(self.hit_at_3 / self.total, 4),
            "hit@5": round(self.hit_at_5 / self.total, 4),
            "mrr": round(self.mrr_sum / self.total, 4),
            "l1_anchor_hit_rate": round(self.l1_hit / self.total, 4),
            "avg_match_score": round(self.avg_score, 4),
        }


def _is_correct(pred: EvalPrediction, rank: int) -> bool:
    if rank >= len(pred.predicted_doc_ids):
        return False
    return pred.predicted_doc_ids[rank] == pred.expected_doc_id


def _reciprocal_rank(pred: EvalPrediction) -> float:
    for i, doc in enumerate(pred.predicted_doc_ids):
        if doc == pred.expected_doc_id:
            return 1.0 / (i + 1)
    return 0.0


def _l1_anchor_hit(pred: EvalPrediction) -> bool:
    return pred.expected_doc_id in pred.l1_anchor_doc_ids


def compute_metrics(
    predictions: Sequence[EvalPrediction],
    *,
    scores: Sequence[float] | None = None,
) -> EvalMetrics:
    """Compute aggregate metrics from a list of predictions."""
    m = EvalMetrics(total=len(predictions))
    if scores is None:
        scores = [0.0] * len(predictions)
    for pred, score in zip(predictions, scores, strict=True):
        if _is_correct(pred, 0):
            m.hit_at_1 += 1
        if _is_correct(pred, 2):
            m.hit_at_3 += 1
        if _is_correct(pred, 4):
            m.hit_at_5 += 1
        m.mrr_sum += _reciprocal_rank(pred)
        if _l1_anchor_hit(pred):
            m.l1_hit += 1
        m.score_sum += float(score)
        m.score_count += 1
        m.per_query.append(
            {
                "query": pred.query,
                "expected": pred.expected_doc_id,
                "predicted": pred.predicted_doc_ids,
                "l1_doc_ids": pred.l1_anchor_doc_ids,
            }
        )
    return m.finalize()
