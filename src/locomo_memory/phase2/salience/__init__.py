"""Phase 2 Salience Scorer (Milestone 4).

Computes and manages salience scores for MemoryUnit objects, driving lifecycle
decisions (compression, forgetting) when the memory store approaches capacity.
"""

from __future__ import annotations

from locomo_memory.phase2.salience.scorer import (
    SalienceResult,
    SalienceScorer,
    SalienceWeights,
)

__all__ = ["SalienceResult", "SalienceScorer", "SalienceWeights"]
