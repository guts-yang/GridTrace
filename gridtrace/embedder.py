"""Semantic encoder for GridTrace.

The default implementation wraps a ``sentence-transformers`` model. We
intentionally keep the embedder interface small so it can be swapped for
a remote embedding API (OpenAI, Cohere, BGE-M3 HTTP, etc.) without
touching the rest of the system.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

__all__ = ["Embedder", "SentenceTransformerEmbedder", "HashingEmbedder"]


class Embedder(ABC):
    """Abstract embedder. Returns L2-normalized float32 vectors."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        """Encode a batch of texts.

        Parameters
        ----------
        texts:
            Input strings.
        kind:
            ``"query"`` for user search queries (uses BGE query
            instruction prefix); ``"document"`` for KB content.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), dim)``, dtype float32, L2-normalized.
        """
        ...


class SentenceTransformerEmbedder(Embedder):
    """Wraps a sentence-transformers model.

    The model is loaded lazily on the first call to :meth:`encode`. This
    keeps unit tests fast (they can use :class:`HashingEmbedder`).
    """

    _QUERY_INSTRUCT_ZH = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        local_path: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._local_path = local_path
        self._model = None  # lazy
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore

        path = self._local_path or self._model_name
        self._model = SentenceTransformer(path, device=self._device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        if not texts:
            dim = self._dim or 1
            return np.zeros((0, dim), dtype=np.float32)
        prefix = self._QUERY_INSTRUCT_ZH if kind == "query" else None
        if prefix:
            texts = [prefix + t for t in texts]  # type: ignore[assignment]
        vecs = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free embedder for tests.

    Uses the signed-hash trick from ``scikit-learn``'s HashingVectorizer
    applied at the character n-gram level. Quality is low but it is
    fast, deterministic, and has no external dependencies.
    """

    def __init__(self, dim: int = 256, n_features: int = 2**18) -> None:
        self._dim = dim
        self._n_features = n_features

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            for tok in tokens:
                h = hash(tok) % self._n_features
                sign = 1.0 if (h & 1) else -1.0
                out[i, h % self._dim] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out

    @staticmethod
    def _tokenize(text: str, n: int = 3) -> list[str]:
        text = f" {text.strip()} "
        if len(text) < n:
            return [text]
        return [text[i : i + n] for i in range(len(text) - n + 1)]


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize each row of ``vecs`` (zero-rows are left as-is)."""
    if vecs.ndim == 1:
        norm = float(np.linalg.norm(vecs))
        return vecs / norm if norm > 0 else vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (vecs / safe).astype(vecs.dtype, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of ``a`` and ``b``."""
    a_n = l2_normalize(a.astype(np.float32, copy=False))
    b_n = l2_normalize(b.astype(np.float32, copy=False))
    return a_n @ b_n.T


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors. Returns 0 if either is zero."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
