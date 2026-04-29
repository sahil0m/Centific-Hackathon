"""Phase 2 Contradiction Resolver with Provenance (Milestone 7).

Classifies relationships between MemoryUnit claims (same fact, update,
temporal change, contradiction, related, unrelated) and creates EdgeRecords
for provenance tracking. Rule-based — no LLM call required.
"""

from __future__ import annotations

from locomo_memory.phase2.contradiction.resolver import (
    ComparisonResult,
    ContradictionResolver,
    RelationshipType,
    ResolutionAction,
    ResolutionResult,
)

__all__ = [
    "ComparisonResult",
    "ContradictionResolver",
    "RelationshipType",
    "ResolutionAction",
    "ResolutionResult",
]
