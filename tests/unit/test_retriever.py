"""Tests for the two-phase retriever."""

from __future__ import annotations

import numpy as np
import pytest

from gridtrace.embedder import HashingEmbedder
from gridtrace.pipeline import GridTracePipeline
from gridtrace.storage import MemoryBackend
from gridtrace.utils.config import GridTraceConfig


@pytest.fixture
def indexed_pipeline() -> GridTracePipeline:
    cfg = GridTraceConfig(
        storage_backend="memory",
        epsilon=0.05,
        anchor_top_k=4,
        score_threshold=0.0,
        top_k=3,
        embedding_dim=64,
    )
    emb = HashingEmbedder(dim=64)
    from gridtrace.generator import EchoGenerator
    p = GridTracePipeline(MemoryBackend(dim=64), emb, EchoGenerator(), cfg)
    p.ingest("How to reset a frozen user account?",
             "Run: UPDATE users SET status='active' WHERE user_id=:uid;",
             category="account", doc_id="kb1", page_index=1)
    p.ingest("Container OOM killed, what to do?",
             "Check describe pod, then adjust resources.limits.memory.",
             category="k8s", doc_id="kb2", page_index=1)
    p.ingest("PostgreSQL replication lag diagnosis",
             "Use pg_last_xact_replay_timestamp() to check lag seconds.",
             category="db", doc_id="kb3", page_index=1)
    p.ingest("Nginx 502 troubleshooting",
             "Inspect upstream health, check error.log for connection refused.",
             category="nginx", doc_id="kb4", page_index=1)
    p.ingest("Redis connection pool exhaustion",
             "Check INFO clients, raise max-connections if high concurrency.",
             category="cache", doc_id="kb5", page_index=1)
    return p


class TestSearch:
    def test_returns_at_most_top_k(self, indexed_pipeline: GridTracePipeline) -> None:
        hits = indexed_pipeline.search("frozen account", top_k=2)
        assert len(hits) <= 2

    def test_threshold_filters(self) -> None:
        cfg = GridTraceConfig(
            storage_backend="memory", epsilon=0.05, anchor_top_k=4,
            score_threshold=0.99, top_k=5, embedding_dim=64,
        )
        emb = HashingEmbedder(dim=64)
        from gridtrace.generator import EchoGenerator
        p = GridTracePipeline(MemoryBackend(dim=64), emb, EchoGenerator(), cfg)
        p.ingest("a", "b", doc_id="d", page_index=1)
        hits = p.search("zzz completely unrelated")
        assert hits == []

    def test_search_with_vector(self, indexed_pipeline: GridTracePipeline) -> None:
        emb = indexed_pipeline.embedder
        qvec = emb.encode(["reset frozen user account"], kind="query")[0]
        hits = indexed_pipeline.retriever.search_with_vector(qvec, top_k=3)
        assert 0 < len(hits) <= 3
        for h in hits:
            assert 0.0 <= h.score <= 1.0

    def test_hit_returns_solution_text(self, indexed_pipeline: GridTracePipeline) -> None:
        # query shares many n-grams with the frozen-account FAQ entry
        hits = indexed_pipeline.search(
            "frozen user account how to reset update users status", top_k=3
        )
        assert hits
        # ensure solution field is present and non-empty
        assert all(h.solution for h in hits)

    def test_empty_kb_returns_empty(self) -> None:
        cfg = GridTraceConfig(
            storage_backend="memory", epsilon=0.05, anchor_top_k=4,
            score_threshold=0.0, top_k=3, embedding_dim=64,
        )
        emb = HashingEmbedder(dim=64)
        from gridtrace.generator import EchoGenerator
        p = GridTracePipeline(MemoryBackend(dim=64), emb, EchoGenerator(), cfg)
        assert p.search("anything") == []


class TestUnlearn:
    def test_unlearn_removes_entries(self, indexed_pipeline: GridTracePipeline) -> None:
        before = indexed_pipeline.stats()
        result = indexed_pipeline.unlearn("kb2")
        assert result.entries_deleted == 1
        after = indexed_pipeline.stats()
        assert after.total_entries == before.total_entries - 1

    def test_unlearn_prunes_orphan_anchor(self, indexed_pipeline: GridTracePipeline) -> None:
        result = indexed_pipeline.unlearn("kb1")
        # kb1 was a unique-doc anchor, so it should be pruned
        assert result.anchors_pruned == 1
