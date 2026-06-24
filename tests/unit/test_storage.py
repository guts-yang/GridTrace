"""Tests for MemoryBackend — the reference behavior all other backends must match."""

from __future__ import annotations

import numpy as np
import pytest

from gridtrace.storage import MemoryBackend


def _vec(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    return (arr / max(np.linalg.norm(arr), 1e-9)).tolist()


class TestAnchorOps:
    def test_upsert_returns_id(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        assert isinstance(aid, int) and aid > 0

    def test_upsert_is_idempotent(self, backend: MemoryBackend) -> None:
        a = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        b = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        assert a == b

    def test_search_returns_top_k(self, backend: MemoryBackend) -> None:
        backend.upsert_anchor("a", _vec([1.0, 0.0]))
        backend.upsert_anchor("b", _vec([0.0, 1.0]))
        backend.upsert_anchor("c", _vec([-1.0, 0.0]))
        hits = backend.search_anchors(_vec([0.9, 0.1]), top_k=2)
        assert len(hits) == 2
        assert hits[0].quant_key == "a"
        assert hits[0].score > hits[1].score

    def test_search_empty_table(self, backend: MemoryBackend) -> None:
        assert backend.search_anchors([0.0, 0.0], top_k=5) == []


class TestEntryOps:
    def test_ingest_creates_entry(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        res = backend.ingest_entry(
            embedding=[0.5, 0.5],
            anchor_id=aid,
            question="Q?",
            solution="A.",
            category=None,
            doc_id="d1",
            page_index=1,
            match_score=0.8,
        )
        assert res.entry_id > 0
        assert res.anchor_id == aid

    def test_ingest_increments_ref_count(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        backend.ingest_entry(_vec([1.0, 0.0]), aid, "q1", "a1", None, "d1", 1, 0.5)
        backend.ingest_entry(_vec([1.0, 0.0]), aid, "q2", "a2", None, "d1", 2, 0.5)
        stats = backend.stats()
        assert stats.total_entries == 2

    def test_ingest_unknown_anchor_raises(self, backend: MemoryBackend) -> None:
        with pytest.raises(KeyError):
            backend.ingest_entry(
                embedding=[0.0, 0.0], anchor_id=9999,
                question="q", solution="a", category=None,
                doc_id="d", page_index=1, match_score=0.0,
            )

    def test_fetch_entry(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        res = backend.ingest_entry(_vec([1.0, 0.0]), aid, "q", "a", "cat", "d", 1, 0.7)
        e = backend.fetch_entry(res.entry_id)
        assert e is not None
        assert e.question == "q"
        assert e.match_score == pytest.approx(0.7)

    def test_update_entry(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        res = backend.ingest_entry(_vec([1.0, 0.0]), aid, "q", "a", "c", "d", 1, 0.5)
        updated = backend.update_entry(res.entry_id, question="q2", match_score=0.9)
        assert updated is not None
        assert updated.question == "q2"
        assert updated.match_score == pytest.approx(0.9)

    def test_list_entries(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k", _vec([1.0, 0.0]))
        for i in range(5):
            backend.ingest_entry(_vec([1.0, 0.0]), aid, f"q{i}", f"a{i}", None, "d", i, 0.5)
        page = backend.list_entries(limit=3, offset=0)
        assert len(page) == 3
        assert page[0].entry_id == 1


class TestUnlearn:
    def test_delete_by_doc(self, backend: MemoryBackend) -> None:
        # Two distinct anchors (different quant_keys) so d1's anchor can be pruned
        a1 = backend.upsert_anchor("k1", _vec([1.0, 0.0]))
        a2 = backend.upsert_anchor("k2", _vec([0.0, 1.0]))
        backend.ingest_entry(_vec([1.0, 0.0]), a1, "q1", "a1", None, "d1", 1, 0.5)
        backend.ingest_entry(_vec([1.0, 0.0]), a1, "q2", "a2", None, "d1", 2, 0.5)
        backend.ingest_entry(_vec([0.0, 1.0]), a2, "q3", "a3", None, "d2", 1, 0.5)

        result = backend.delete_by_doc("d1")
        assert result.entries_deleted == 2
        assert result.anchors_pruned == 1    # d1's anchor becomes orphan

        stats = backend.stats()
        assert stats.total_entries == 1
        assert stats.total_anchors == 1
        # The remaining entry belongs to d2
        remaining = backend.list_entries(limit=10)
        assert all(e.doc_id == "d2" for e in remaining)

    def test_delete_unknown_doc(self, backend: MemoryBackend) -> None:
        result = backend.delete_by_doc("nonexistent")
        assert result.entries_deleted == 0
        assert result.anchors_pruned == 0

    def test_delete_keeps_shared_anchor(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k", _vec([1.0, 0.0]))
        backend.ingest_entry(_vec([1.0, 0.0]), aid, "q1", "a1", None, "d1", 1, 0.5)
        backend.ingest_entry(_vec([1.0, 0.0]), aid, "q2", "a2", None, "d2", 1, 0.5)
        result = backend.delete_by_doc("d1")
        assert result.entries_deleted == 1
        assert result.anchors_pruned == 0    # still referenced by d2
        assert backend.stats().total_anchors == 1


class TestPrune:
    def test_prune_orphan_anchors(self, backend: MemoryBackend) -> None:
        backend.upsert_anchor("k1", _vec([1.0, 0.0]))     # no entries → orphan
        aid = backend.upsert_anchor("k2", _vec([0.0, 1.0]))
        backend.ingest_entry(_vec([0.0, 1.0]), aid, "q", "a", None, "d", 1, 0.5)
        pruned = backend.prune_orphan_anchors()
        assert pruned == 1
        assert backend.stats().total_anchors == 1


class TestStats:
    def test_empty_stats(self, backend: MemoryBackend) -> None:
        s = backend.stats()
        assert s.total_anchors == 0
        assert s.total_entries == 0
        assert s.unique_docs == 0
        assert s.avg_match_score == 0.0

    def test_reset(self, backend: MemoryBackend) -> None:
        aid = backend.upsert_anchor("k", _vec([1.0, 0.0]))
        backend.ingest_entry(_vec([1.0, 0.0]), aid, "q", "a", None, "d", 1, 0.5)
        backend.reset()
        s = backend.stats()
        assert s.total_anchors == 0
        assert s.total_entries == 0


class TestThreadSafety:
    def test_concurrent_ingest(self, backend: MemoryBackend) -> None:
        """Smoke test: 20 concurrent ingests should not corrupt the store."""
        import threading

        aid = backend.upsert_anchor("k", _vec([1.0, 0.0]))
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                backend.ingest_entry(
                    embedding=_vec([1.0, 0.0]),
                    anchor_id=aid,
                    question=f"q{i}", solution=f"a{i}", category=None,
                    doc_id=f"d{i}", page_index=i, match_score=0.5,
                )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert backend.stats().total_entries == 20
