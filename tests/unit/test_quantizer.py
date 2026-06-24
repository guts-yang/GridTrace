"""Tests for gridtrace.quantizer."""

from __future__ import annotations

import numpy as np
import pytest

from gridtrace.quantizer import quantize_vector, quantize_batch
from gridtrace.utils.hashing import quant_key_from_vector


class TestQuantizeVector:
    def test_snap_to_grid(self) -> None:
        v = np.array([0.001, 0.019, 0.020, 0.031, -0.011], dtype=np.float64)
        out = quantize_vector(v, epsilon=0.02)
        np.testing.assert_allclose(out, [0.0, 0.02, 0.02, 0.04, -0.02])

    def test_default_epsilon(self) -> None:
        v = np.array([0.005, 0.015, 0.025])
        out = quantize_vector(v)  # default 0.02
        np.testing.assert_allclose(out, [0.0, 0.02, 0.02])

    def test_epsilon_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            quantize_vector([0.1, 0.2], epsilon=0)
        with pytest.raises(ValueError):
            quantize_vector([0.1, 0.2], epsilon=-0.01)

    def test_one_dimensional_only(self) -> None:
        with pytest.raises(ValueError):
            quantize_vector(np.zeros((3, 4)))

    def test_idempotence(self) -> None:
        """Re-quantizing a quantized vector returns the same vector."""
        v = np.random.default_rng(42).normal(size=32)
        q1 = quantize_vector(v, epsilon=0.05)
        q2 = quantize_vector(q1, epsilon=0.05)
        np.testing.assert_array_equal(q1, q2)

    def test_same_input_same_output(self) -> None:
        v = np.array([0.123, 0.456, 0.789])
        a = quantize_vector(v, epsilon=0.01)
        b = quantize_vector(v, epsilon=0.01)
        np.testing.assert_array_equal(a, b)

    def test_handles_lists(self) -> None:
        out = quantize_vector([0.001, 0.5, -0.5], epsilon=0.1)
        np.testing.assert_allclose(out, [0.0, 0.5, -0.5])

    def test_zero_vector(self) -> None:
        out = quantize_vector([0.0, 0.0, 0.0], epsilon=0.02)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])


class TestQuantizeBatch:
    def test_shape(self) -> None:
        vecs = np.random.default_rng(0).normal(size=(5, 16))
        out = quantize_batch(vecs, epsilon=0.05)
        assert out.shape == (5, 16)

    def test_2d_only(self) -> None:
        with pytest.raises(ValueError):
            quantize_batch(np.zeros((4,)))

    def test_matches_per_vector(self) -> None:
        rng = np.random.default_rng(7)
        vecs = rng.normal(size=(3, 8))
        batch = quantize_batch(vecs, epsilon=0.03)
        for i in range(3):
            per = quantize_vector(vecs[i], epsilon=0.03)
            np.testing.assert_array_equal(per, batch[i])


class TestQuantKey:
    def test_stable(self) -> None:
        v = np.array([0.12, 0.34, 0.56])
        k1 = quant_key_from_vector(v)
        k2 = quant_key_from_vector(v)
        assert k1 == k2
        assert len(k1) == 64

    def test_epsilon_groups_collapse(self) -> None:
        """Vectors in the same grid cell must share a key."""
        v1 = np.array([0.121, 0.339, 0.561])
        v2 = np.array([0.124, 0.342, 0.558])  # same cell with eps=0.02
        k1 = quant_key_from_vector(quantize_vector(v1, 0.02))
        k2 = quant_key_from_vector(quantize_vector(v2, 0.02))
        assert k1 == k2

    def test_different_cells_differ(self) -> None:
        v1 = np.array([0.01, 0.01])
        v2 = np.array([0.05, 0.05])
        k1 = quant_key_from_vector(quantize_vector(v1, 0.02))
        k2 = quant_key_from_vector(quantize_vector(v2, 0.02))
        assert k1 != k2
