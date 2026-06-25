"""End-to-end GridTrace pipeline.

Composes the embedder, quantizer, retriever, generator, and storage
backend into a single object exposing the four high-level operations:

* :meth:`ingest` — encode + quantize + upsert
* :meth:`search` — L1 + L2 retrieval
* :meth:`generate` — search + LLM answer
* :meth:`unlearn` — delete by doc_id, prune orphans

The pipeline is **stateless** between calls; all state lives in the
storage backend. This makes the pipeline trivially safe to share
across threads / processes.
"""

from __future__ import annotations

import numpy as np

from gridtrace.embedder import Embedder, cosine, l2_normalize
from gridtrace.generator import (
    DeepSeekGenerator,
    EchoGenerator,
    Generator,
    OpenAIGenerator,
)
from gridtrace.quantizer import quantize_vector
from gridtrace.retriever import Retriever
from gridtrace.storage._types import (
    Entry,
    IngestResult,
    SearchHit,
    StorageBackend,
    UnlearnResult,
)
from gridtrace.utils.config import GridTraceConfig
from gridtrace.utils.hashing import quant_key_from_vector
from gridtrace.utils.logging import get_logger

logger = get_logger(__name__)


def _make_generator(config: GridTraceConfig) -> Generator:
    provider = config.llm_provider
    if provider == "echo":
        return EchoGenerator()
    if provider == "deepseek":
        return DeepSeekGenerator(config)
    if provider == "openai":
        return OpenAIGenerator(config)
    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def _make_backend(config: GridTraceConfig) -> StorageBackend:
    backend = config.storage_backend
    dim = config.embedding_dim
    if backend == "memory":
        from gridtrace.storage.memory import MemoryBackend

        return MemoryBackend(dim=dim)
    if backend == "sqlite":
        from gridtrace.storage.sqlite import SqliteBackend

        return SqliteBackend(path=config.sqlite_path, dim=dim)
    if backend == "postgres":
        from gridtrace.storage.postgres import PostgresBackend

        return PostgresBackend(database_url=config.database_url, dim=dim)
    raise ValueError(f"Unsupported storage backend: {backend!r}")


class GridTracePipeline:
    """The end-to-end pipeline. Composable; safe to share across threads."""

    def __init__(
        self,
        storage: StorageBackend,
        embedder: Embedder,
        generator: Generator,
        config: GridTraceConfig | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._generator = generator
        self._config = config or GridTraceConfig()
        self._retriever = Retriever(storage, embedder, self._config)

    @classmethod
    def from_config(
        cls,
        config: GridTraceConfig | None = None,
        *,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
    ) -> "GridTracePipeline":
        cfg = config or GridTraceConfig()
        be = _make_backend(cfg)
        emb = embedder or _default_embedder(cfg)
        gen = generator or _make_generator(cfg)
        return cls(be, emb, gen, cfg)

    @property
    def storage(self) -> StorageBackend:
        return self._storage

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def generator(self) -> Generator:
        return self._generator

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    # ── ingest ────────────────────────────────────────────
    def ingest(
        self,
        question: str,
        solution: str,
        *,
        category: str | None = None,
        doc_id: str = "default",
        page_index: int = 0,
    ) -> IngestResult:
        """Encode q + s jointly, quantize, upsert anchor + entry.

        The joint vector is ``normalize((q_vec + s_vec) / 2)``. The
        self-check ``match_score`` is ``cosine(q_vec, s_vec)``.
        """
        qvec, svec = self._embedder.encode([question, solution], kind="document")
        joint = l2_normalize(((qvec + svec) / 2.0).astype(np.float32))
        match_score = float(cosine(qvec, svec))
        anchor_vec = quantize_vector(joint, epsilon=self._config.epsilon)
        quant_key = quant_key_from_vector(anchor_vec)
        anchor_id = self._storage.upsert_anchor(quant_key, anchor_vec.tolist())
        result = self._storage.ingest_entry(
            embedding=joint.tolist(),
            anchor_id=anchor_id,
            question=question,
            solution=solution,
            category=category,
            doc_id=doc_id,
            page_index=page_index,
            match_score=match_score,
        )
        logger.info(
            "ingest doc=%s#%d entry=%d anchor=%d score=%.3f reused=%s",
            doc_id, page_index, result.entry_id, result.anchor_id,
            match_score, result.reused_anchor,
        )
        return result

    # ── search ────────────────────────────────────────────
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        anchor_top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        return self._retriever.search(
            query,
            top_k=top_k,
            anchor_top_k=anchor_top_k,
            score_threshold=score_threshold,
        )

    # ── generate ──────────────────────────────────────────
    async def generate(self, query: str, *, top_k: int | None = None) -> tuple[str, list[SearchHit]]:
        hits = self.search(query, top_k=top_k)
        answer = await self._generator.generate(query, hits)
        return answer, hits

    # ── unlearn ───────────────────────────────────────────
    def unlearn(self, doc_id: str) -> UnlearnResult:
        result = self._storage.delete_by_doc(doc_id)
        logger.info(
            "unlearn doc=%s entries=%d anchors_pruned=%d",
            doc_id, result.entries_deleted, result.anchors_pruned,
        )
        return result

    # ── stats ─────────────────────────────────────────────
    def stats(self):
        return self._storage.stats()

    def close(self) -> None:
        self._storage.close()


def _default_embedder(cfg: GridTraceConfig) -> Embedder:
    """Best-effort default embedder: try sentence-transformers, else hashing.

    The hashing fallback guarantees the pipeline is importable and
    usable in environments where model download isn't possible (CI,
    offline dev, etc.).
    """
    # Probe for sentence-transformers up-front so missing packages
    # don't surface during the first encode() call inside a request.
    try:
        import sentence_transformers  # noqa: F401

        from gridtrace.embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=cfg.embedding_model,
            device=cfg.embedding_device,
            local_path=cfg.embedding_local_path,
        )
    except Exception as exc:  # pragma: no cover - depends on env
        logger.warning(
            "Falling back to HashingEmbedder (sentence-transformers unavailable: %s)",
            exc,
        )
        from gridtrace.embedder import HashingEmbedder

        return HashingEmbedder(dim=cfg.embedding_dim)
