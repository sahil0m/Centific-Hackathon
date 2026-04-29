"""Phase 2 ingestion pipeline components.

Pipeline order (see PHASE2_METHODOLOGY.md §5):

    [1] Trivial Filter          (rule-based, removes greetings/laughs)
    [2] Semantic Chunking       (group consecutive turns by topic)        ← this module
    [3] Candidate Detector      (cheap pre-LLM filter)                     ← this module
    [4] Agentic Chunking        (LLM extracts atomic facts)
    [5] Embedding Generation
    [6] Salience Scoring
    [7] Contradiction Detection
    [8] Graph Linking
    [9] Memory Store Write
"""

from __future__ import annotations

from locomo_memory.phase2.ingestion.candidate_detector import (
    CandidateDetector,
    CandidateScore,
    CandidateWeights,
)
from locomo_memory.phase2.ingestion.fact_extractor import (
    ExtractionResult,
    FactExtractor,
)
from locomo_memory.phase2.ingestion.importance import TopicImportanceEstimator
from locomo_memory.phase2.ingestion.semantic_chunker import (
    SemanticChunker,
    TurnEmbedder,
)

__all__ = [
    "CandidateDetector",
    "CandidateScore",
    "CandidateWeights",
    "ExtractionResult",
    "FactExtractor",
    "SemanticChunker",
    "TopicImportanceEstimator",
    "TurnEmbedder",
]
