"""Phase 2 runner configuration — Milestone 10.

All experiment settings live in a single Pydantic model tree so configs can
be loaded from YAML, validated, serialised to JSON for reproducibility, and
swept for ablation studies by toggling individual flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class Phase2DatasetConfig(BaseModel):
    """LoCoMo dataset location and slice controls."""

    path: str = "data/raw/locomo10.json"
    max_conversations: int | None = None
    """Cap the number of conversations processed (useful for quick tests)."""
    max_qa_per_conversation: int | None = None
    """Cap QA items per conversation."""


class Phase2IngestionConfig(BaseModel):
    """How conversation turns become MemoryUnits."""

    use_turn_as_claim: bool = True
    """If True, each dialogue turn is one MU (no LLM extraction).
    Set to False to enable agentic fact extraction (not yet implemented)."""
    claim_format: str = "speaker_text"
    """'speaker_text' → '{speaker}: {text}'; 'text_only' → '{text}' only."""
    skip_summary_turns: bool = True
    """Skip turns with speaker='summary' (session summaries)."""
    min_turn_length: int = 3
    """Turns with fewer characters than this are skipped."""


class Phase2EmbeddingConfig(BaseModel):
    """Embedding model settings."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    normalize: bool = True
    batch_size: int = 64
    cache_dir: str = "data/processed/embedding_cache"


class Phase2RetrievalConfig(BaseModel):
    """Hybrid retrieval pipeline settings."""

    top_k: int = 5
    rrf_k: int = 60
    dense_candidates: int = 20
    bm25_candidates: int = 20
    label_candidates: int = 10
    enable_bm25: bool = True
    enable_label_search: bool = False
    """Requires compressed labels; off by default for the basic runner."""
    enable_graph_traversal: bool = False
    enable_forgotten_fallback: bool = False
    # Source evidence lane
    enable_source_evidence_lane: bool = False
    source_context_window: int = 2
    source_bm25_top_n: int = 20
    source_dense_top_n: int = 0
    source_lane_rrf_weight: float = 1.0
    # Cross-encoder reranking
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    cross_encoder_weight: float = 3.0
    cross_encoder_batch_size: int = 32
    cross_encoder_max_length: int = 512
    cross_encoder_pool_size: int = 50
    ce_superseded_penalty: float = 0.10
    ce_diversity_max_same_dia: int = 2


class Phase2ContextConfig(BaseModel):
    """Context builder settings."""

    max_entries: int = 10
    """Maximum number of evidence entries in the built context."""


class Phase2GenerationConfig(BaseModel):
    """Answer LLM settings."""

    enabled: bool = False
    """Retrieval-only mode when False — no LLM calls made."""
    provider: str = "anthropic"
    model_name: str = "claude-3-5-sonnet-latest"
    temperature: float = 0.0
    max_output_tokens: int = 120
    cache_dir: str = "data/processed/llm_cache"


class Phase2GuardConfig(BaseModel):
    """ResponseGuard settings."""

    min_grounding_score: float = 0.0
    require_uncertainty_for_conflicts: bool = True


class Phase2EvaluationConfig(BaseModel):
    """Evaluation metric controls."""

    compute_f1: bool = True
    compute_exact_match: bool = True
    compute_evidence_recall: bool = True
    compute_grounding: bool = True


class Phase2OutputConfig(BaseModel):
    """Result file locations."""

    dir: str = "results/phase2"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class Phase2RunnerConfig(BaseModel):
    """Root configuration for one Phase 2 experiment run."""

    experiment_name: str = "phase2_retrieval_only"
    seed: int = 42
    db_dir: str = "data/processed/phase2_db"
    """Directory for per-conversation SQLite stores."""

    dataset: Phase2DatasetConfig = Field(default_factory=Phase2DatasetConfig)
    ingestion: Phase2IngestionConfig = Field(default_factory=Phase2IngestionConfig)
    embedding: Phase2EmbeddingConfig = Field(default_factory=Phase2EmbeddingConfig)
    retrieval: Phase2RetrievalConfig = Field(default_factory=Phase2RetrievalConfig)
    context: Phase2ContextConfig = Field(default_factory=Phase2ContextConfig)
    generation: Phase2GenerationConfig = Field(default_factory=Phase2GenerationConfig)
    guard: Phase2GuardConfig = Field(default_factory=Phase2GuardConfig)
    evaluation: Phase2EvaluationConfig = Field(default_factory=Phase2EvaluationConfig)
    output: Phase2OutputConfig = Field(default_factory=Phase2OutputConfig)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Phase2RunnerConfig:
        """Load from a YAML file.  Nested keys are merged with defaults."""
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(_flatten_yaml(raw))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# YAML normaliser
# ---------------------------------------------------------------------------

def _flatten_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept both flat and nested YAML shapes.

    Supports:
      experiment_name: foo            (flat)
      experiment:
        name: foo                     (nested with 'name' key)
    """
    out: dict[str, Any] = dict(raw)
    # experiment block
    if "experiment" in raw and isinstance(raw["experiment"], dict):
        exp = raw.pop("experiment")
        out.pop("experiment", None)
        if "name" in exp:
            out["experiment_name"] = exp["name"]
        if "seed" in exp:
            out["seed"] = exp["seed"]
    return out


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Phase2ContextConfig",
    "Phase2DatasetConfig",
    "Phase2EmbeddingConfig",
    "Phase2EvaluationConfig",
    "Phase2GenerationConfig",
    "Phase2GuardConfig",
    "Phase2IngestionConfig",
    "Phase2OutputConfig",
    "Phase2RetrievalConfig",
    "Phase2RunnerConfig",
]
