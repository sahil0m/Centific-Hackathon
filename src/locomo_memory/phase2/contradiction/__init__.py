"""Phase 2 Contradiction Resolver with Provenance (Milestone 7).

Classifies relationships between MemoryUnit claims (same fact, update,
temporal change, contradiction, related, unrelated) using NLI as the primary
signal and creates EdgeRecords for provenance tracking.

Research basis: cross-encoder/nli-deberta-v3-large (He et al. 2021) fine-tuned
on SNLI + MultiNLI corpora for state-of-the-art zero-shot contradiction detection.
"""

from __future__ import annotations

from locomo_memory.phase2.contradiction.nli_classifier import (
    FakeNLIClassifier,
    NLIContradictionClassifier,
    NLIScores,
)
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
    "FakeNLIClassifier",
    "NLIContradictionClassifier",
    "NLIScores",
    "RelationshipType",
    "ResolutionAction",
    "ResolutionResult",
]
