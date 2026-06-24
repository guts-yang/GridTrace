"""Stable SHA256 hashing for quantized vectors.

The quant_key is the **identity** of a quantized anchor. Two vectors that
snap to the same grid cell MUST produce identical keys; identical keys
across machines are essential so that the same anchor can be discovered
deterministically when ingesting the same content twice.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def quant_key_from_vector(vec: Iterable[float], *, precision: int = 6) -> str:
    """Compute a stable SHA256 hash for a (quantized) vector.

    Parameters
    ----------
    vec:
        Any iterable of floats. The values are expected to be *quantized*
        (i.e. multiples of `epsilon`), but the function only needs stable
        representation, so it rounds to `precision` decimal places to
        remove floating-point noise.
    precision:
        Decimal places kept before hashing. Default 6 is plenty for
        `epsilon ≥ 1e-4`.

    Returns
    -------
    str
        A 64-character lowercase hex digest.
    """
    arr = np.asarray(list(vec), dtype=np.float64).round(precision)
    payload = arr.tobytes()
    return hashlib.sha256(payload).hexdigest()


def short_key(quant_key: str, *, length: int = 12) -> str:
    """Return a short human-readable prefix of a quant_key."""
    if length < 4:
        raise ValueError("length must be >= 4")
    return quant_key[:length]
