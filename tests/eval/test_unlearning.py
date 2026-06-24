"""Exact unlearning regression test."""

from __future__ import annotations

from gridtrace.embedder import HashingEmbedder
from gridtrace.eval import EvalEngine, compute_metrics
from gridtrace.loaders import load_entries
from gridtrace.utils.config import GridTraceConfig


def test_unlearn_removes_target_doc() -> None:
    cfg = GridTraceConfig(
        storage_backend="memory", epsilon=0.05, anchor_top_k=4,
        score_threshold=0.0, top_k=3, embedding_dim=128,
    )
    emb = HashingEmbedder(dim=128)
    engine = EvalEngine(emb, cfg)

    # ingest three docs
    engine.ingest("q1", "a1", doc_id="d1", page_index=1, category="x")
    engine.ingest("q2", "a2", doc_id="d2", page_index=1, category="y")
    engine.ingest("q3", "a3", doc_id="d3", page_index=1, category="z")
    assert engine.storage.stats().total_anchors >= 1

    # unlearn d2
    result = engine.storage.delete_by_doc("d2")
    assert result.entries_deleted == 1

    # remaining entries must NOT include d2
    for e in engine.storage.list_entries(limit=100):
        assert e.doc_id != "d2"

    # orphan anchor (if d2 had a unique anchor) should be pruned
    stats = engine.storage.stats()
    assert stats.total_entries == 2
    assert stats.unique_docs == 2


def test_unlearn_keeps_shared_anchor() -> None:
    """If two docs share an anchor, deleting one doc keeps the anchor alive."""
    import numpy as np

    from gridtrace.quantizer import quantize_vector
    from gridtrace.utils.hashing import quant_key_from_vector

    cfg = GridTraceConfig(
        storage_backend="memory", epsilon=1.0,   # very coarse → many collapses
        anchor_top_k=4, score_threshold=0.0, top_k=3, embedding_dim=64,
    )
    emb = HashingEmbedder(dim=64)
    engine = EvalEngine(emb, cfg)

    # Manually create two entries sharing the SAME anchor, bypassing embedder noise
    joint = quantize_vector([0.1] * 64, epsilon=0.05)
    key = quant_key_from_vector(joint)
    aid = engine.storage.upsert_anchor(key, joint.tolist())
    engine.storage.ingest_entry(
        embedding=[0.1] * 64, anchor_id=aid,
        question="q1", solution="a1", category=None,
        doc_id="d1", page_index=1, match_score=0.5,
    )
    engine.storage.ingest_entry(
        embedding=[0.1] * 64, anchor_id=aid,
        question="q2", solution="a2", category=None,
        doc_id="d2", page_index=1, match_score=0.5,
    )
    pre = engine.storage.stats()
    assert pre.total_anchors == 1, "two identical entries should share anchor"

    res = engine.storage.delete_by_doc("d1")
    assert res.anchors_pruned == 0  # anchor still referenced by d2
    assert engine.storage.stats().total_anchors == 1


def test_eval_pipeline_after_unlearn() -> None:
    """Sanity: eval engine still works after an unlearn operation."""
    cfg = GridTraceConfig(
        storage_backend="memory", epsilon=0.05, anchor_top_k=4,
        score_threshold=0.0, top_k=3, embedding_dim=64,
    )
    emb = HashingEmbedder(dim=64)
    engine = EvalEngine(emb, cfg)
    engine.ingest("q1", "a1", doc_id="d1", page_index=1)
    engine.ingest("q2", "a2", doc_id="d2", page_index=1)
    engine.unlearn_storage("d1") if hasattr(engine, "unlearn_storage") else engine.storage.delete_by_doc("d1")

    pred = engine.predict("q1", "d2", 1, top_k=2, anchor_top_k=4, score_threshold=0.0)
    metrics = compute_metrics([pred])
    assert metrics.total == 1
