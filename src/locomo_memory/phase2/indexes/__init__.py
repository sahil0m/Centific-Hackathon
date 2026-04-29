"""Phase 2 Memory Indexes — Milestones 8 and 8B.

FAISS dense index for MemoryUnit claims, CompressedLabel FAISS index,
plus the existing NetworkX graph index.
"""

from __future__ import annotations

from locomo_memory.phase2.indexes.faiss_index import (
    EmbedFn,
    FAISSSearchResult,
    MemoryFAISSIndex,
)
from locomo_memory.phase2.indexes.label_index import (
    CompressedLabelFAISSIndex,
    LabelSearchResult,
)
from locomo_memory.phase2.indexes.source_evidence_index import (
    SourceEvidenceEntry,
    SourceEvidenceHit,
    SourceEvidenceIndex,
)
from locomo_memory.phase2.store.graph_index import MemoryGraphIndex

__all__ = [
    "CompressedLabelFAISSIndex",
    "EmbedFn",
    "FAISSSearchResult",
    "LabelSearchResult",
    "MemoryFAISSIndex",
    "MemoryGraphIndex",
    "SourceEvidenceEntry",
    "SourceEvidenceHit",
    "SourceEvidenceIndex",
]
