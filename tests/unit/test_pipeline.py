"""End-to-end pipeline tests."""

from __future__ import annotations

import asyncio

import pytest

from gridtrace.pipeline import GridTracePipeline


@pytest.mark.asyncio
async def test_ingest_then_search_then_generate(pipeline: GridTracePipeline) -> None:
    pipeline.ingest(
        "How to unlock a frozen account?",
        "Run UPDATE users SET status='active' WHERE id=:uid;",
        category="account", doc_id="d1", page_index=1,
    )
    pipeline.ingest(
        "What causes container OOMKilled?",
        "Memory limit exceeded; inspect describe pod and heap dump.",
        category="k8s", doc_id="d2", page_index=1,
    )

    hits = pipeline.search("frozen account unlock", top_k=2)
    assert hits
    assert hits[0].doc_id == "d1"

    answer, refs = await pipeline.generate("frozen account unlock", top_k=2)
    assert "UPDATE users" in answer
    assert len(refs) >= 1


@pytest.mark.asyncio
async def test_generate_empty_kb_returns_fallback(pipeline: GridTracePipeline) -> None:
    answer, hits = await pipeline.generate("anything")
    assert hits == []
    assert "暂无" in answer  # EchoGenerator fallback


def test_unlearn_cascade(pipeline: GridTracePipeline) -> None:
    pipeline.ingest("q1", "a1", doc_id="d1", page_index=1)
    pipeline.ingest("q2", "a2", doc_id="d1", page_index=2)
    pipeline.ingest("q3", "a3", doc_id="d2", page_index=1)

    result = pipeline.unlearn("d1")
    assert result.entries_deleted == 2
    stats = pipeline.stats()
    assert stats.total_entries == 1
    # All d1 anchors should be pruned
    assert stats.total_anchors == 1


def test_stats_reflect_ingest(pipeline: GridTracePipeline) -> None:
    pipeline.ingest("q1", "a1", doc_id="d1", page_index=1)
    pipeline.ingest("q2", "a2", doc_id="d2", page_index=1)
    s = pipeline.stats()
    assert s.total_entries == 2
    assert s.unique_docs == 2
    assert 0.0 <= s.avg_match_score <= 1.0
