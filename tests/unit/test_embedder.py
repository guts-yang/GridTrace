"""Tests for the embedder module."""

from __future__ import annotations

import numpy as np

from gridtrace.embedder import HashingEmbedder, cosine, cosine_similarity, l2_normalize


class TestHashingEmbedder:
    def test_dim(self) -> None:
        e = HashingEmbedder(dim=128)
        assert e.dim == 128

    def test_encode_shape(self) -> None:
        e = HashingEmbedder(dim=64)
        v = e.encode(["hello", "world"])
        assert v.shape == (2, 64)
        assert v.dtype == np.float32

    def test_encode_normalized(self) -> None:
        e = HashingEmbedder(dim=64)
        v = e.encode(["a longer text with several tokens 1234", "another text"])
        norms = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    def test_empty_input(self) -> None:
        e = HashingEmbedder(dim=32)
        v = e.encode([])
        assert v.shape == (0, 32)


class TestVectorMath:
    def test_l2_normalize_2d(self) -> None:
        v = np.array([[3.0, 4.0], [0.0, 0.0]])
        out = l2_normalize(v)
        np.testing.assert_allclose(out[0], [0.6, 0.8])
        np.testing.assert_array_equal(out[1], [0.0, 0.0])

    def test_l2_normalize_1d(self) -> None:
        v = np.array([3.0, 4.0])
        out = l2_normalize(v)
        np.testing.assert_allclose(out, [0.6, 0.8])

    def test_cosine_identical(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        assert cosine(v, v) == pytest_approx(1.0)

    def test_cosine_orthogonal(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine(a, b) == pytest_approx(0.0)

    def test_cosine_zero(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine(a, b) == 0.0

    def test_cosine_similarity_batch(self) -> None:
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [1.0, 0.0]])
        out = cosine_similarity(a, b)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(out[1, 0], 0.0, atol=1e-5)


def pytest_approx(x: float):
    import pytest
    return pytest.approx(x, abs=1e-5)
