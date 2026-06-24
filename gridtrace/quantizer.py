"""Grid quantization — the core of GridTrace.

The algorithm is intentionally trivial:

    anchor_vec = round(v / ε) × ε
    quant_key  = SHA256(anchor_vec)

Properties we rely on:

* **Determinism** — the same ``v`` always maps to the same key on any
  machine. This lets distributed ingest processes converge to the same
  anchor IDs.
* **No training** — unlike K-means or PQ there is no fit() step, no
  codebook, no random init.
* **O(d) per vector** — just a multiply, a round, and a multiply.
* **Tunable granularity** — smaller ε means finer grid, more anchors,
  tighter routing precision at the cost of storage.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from gridtrace.utils.hashing import quant_key_from_vector

__all__ = ["quantize_vector", "quant_key_from_vector", "QUANT_EPSILON_DEFAULT"]

QUANT_EPSILON_DEFAULT = 0.02


def quantize_vector(
    vec: Iterable[float] | np.ndarray,
    epsilon: float = QUANT_EPSILON_DEFAULT,
) -> np.ndarray:
    """Snap each component of ``vec`` to the nearest multiple of ``epsilon``.

    Parameters
    ----------
    vec:
        1-D iterable of floats (length ``d``) or a numpy array of shape
        ``(d,)``.
    epsilon:
        Grid granularity. Must be > 0. Smaller values mean a finer grid.

    Returns
    -------
    np.ndarray
        Float64 vector of the same shape with values in ``{…, -2ε, -ε, 0,
        ε, 2ε, …}``.

    Raises
    ------
    ValueError
        If ``epsilon <= 0`` or ``vec`` is not 1-D.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"vec must be 1-D, got shape {arr.shape}")
    return np.round(arr / epsilon) * epsilon


def quantize_batch(
    vecs: Sequence[Sequence[float]],
    epsilon: float = QUANT_EPSILON_DEFAULT,
) -> np.ndarray:
    """Vectorized quantization for a stack of vectors. Shape: (N, d)."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    arr = np.asarray(vecs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"vecs must be 2-D (N, d), got shape {arr.shape}")
    return np.round(arr / epsilon) * epsilon
