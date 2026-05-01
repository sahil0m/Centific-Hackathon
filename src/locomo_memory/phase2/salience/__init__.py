"""Phase 2 Salience Scorer (Milestone 4).

Computes salience scores for MemoryUnit objects using the Ebbinghaus
forgetting curve, driving lifecycle eviction decisions when the active
memory store approaches capacity.
"""

from __future__ import annotations

from locomo_memory.phase2.salience.scorer import (
    SalienceResult,
    SalienceScorer,
)

__all__ = ["SalienceResult", "SalienceScorer"]
