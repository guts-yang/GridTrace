"""GridTrace evaluation engine — offline two-phase retrieval replica.

The eval engine mirrors the production retriever's behavior exactly so
that offline metrics (Hit@K, MRR, etc.) faithfully predict online
performance. Implementation uses a fresh :class:`MemoryBackend` per
eval run to isolate state.
"""

from __future__ import annotations

from .engine import EvalEngine
from .metrics import EvalMetrics, compute_metrics

__all__ = ["EvalEngine", "EvalMetrics", "compute_metrics"]
