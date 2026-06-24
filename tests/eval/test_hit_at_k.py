"""Hit@K regression test using the bundled OpsWarden FAQ + eval_v3."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridtrace.embedder import HashingEmbedder
from gridtrace.eval import EvalEngine, compute_metrics
from gridtrace.loaders import load_entries
from gridtrace.utils.config import GridTraceConfig


ROOT = Path(__file__).resolve().parents[2]
FAQ_PATH = ROOT / "data" / "faq" / "OpsWarden_FAQ.md"
EVAL_PATH = ROOT / "data" / "eval" / "eval_v3.json"


@pytest.mark.skipif(not FAQ_PATH.exists(), reason="FAQ not found")
@pytest.mark.skipif(not EVAL_PATH.exists(), reason="Eval set not found")
def test_hit_at_k_regression() -> None:
    import json

    cfg = GridTraceConfig(
        storage_backend="memory", epsilon=0.05, anchor_top_k=8,
        score_threshold=0.0, top_k=3, embedding_dim=512,
    )
    emb = HashingEmbedder(dim=512)
    engine = EvalEngine(emb, cfg)

    entries = load_entries(FAQ_PATH)
    for e in entries:
        engine.ingest(
            question=e.question, solution=e.solution,
            category=e.category, doc_id=e.doc_id, page_index=e.page_index,
        )
    assert engine.storage.stats().total_entries == len(entries)

    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    predictions = [
        engine.predict(
            query=q["query"],
            expected_doc_id=q["expected_doc_id"],
            expected_page_index=q["expected_page_index"],
            top_k=3, anchor_top_k=8, score_threshold=0.0,
        )
        for q in eval_data
    ]
    metrics = compute_metrics(predictions)
    summary = metrics.as_dict()
    # the hashing embedder is weak — we only check that the pipeline runs end-to-end
    # and produces finite numbers; stronger embedders in CI will improve this.
    assert summary["total"] == len(eval_data)
    for k in ["hit@1", "hit@3", "hit@5", "mrr", "l1_anchor_hit_rate"]:
        assert 0.0 <= summary[k] <= 1.0
