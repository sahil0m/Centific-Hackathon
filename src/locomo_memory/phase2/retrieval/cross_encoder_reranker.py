"""Cross-encoder reranker — Phase 2 Milestone 16.

Provides:
  CrossEncoderRerankerProtocol      — structural typing interface
  SentenceTransformersCrossEncoderReranker — real implementation (lazy load)
  FakeCrossEncoderReranker          — deterministic token-overlap fake for tests
  build_candidate_text              — rich candidate text builder
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from locomo_memory.phase2.indexes.source_evidence_index import SourceEvidenceIndex


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CrossEncoderRerankerProtocol(Protocol):
    """Any object with score_pairs() satisfies this interface."""

    def score_pairs(self, query: str, candidate_texts: list[str]) -> list[float]:
        """Return one score per candidate text, same order as input."""
        ...


# ---------------------------------------------------------------------------
# Real implementation (lazy model load)
# ---------------------------------------------------------------------------


class SentenceTransformersCrossEncoderReranker:
    """Cross-encoder using sentence-transformers CrossEncoder.

    The model is loaded lazily on the first call to :meth:`score_pairs` so
    importing this class never triggers a download.

    Args:
        model_name: HuggingFace model ID.  Defaults to BAAI/bge-reranker-base.
            Use BAAI/bge-reranker-large for higher quality at higher cost.
        batch_size: pairs per forward pass.
        max_length: token truncation limit passed to the tokenizer.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None  # lazy

    def score_pairs(self, query: str, candidate_texts: list[str]) -> list[float]:
        """Score (query, candidate) pairs; higher = more relevant."""
        if not candidate_texts:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import
            self._model = CrossEncoder(self._model_name, max_length=self._max_length)
        pairs = [[query, t] for t in candidate_texts]
        raw = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in raw]


# ---------------------------------------------------------------------------
# Fake implementation for tests (no model download)
# ---------------------------------------------------------------------------


class FakeCrossEncoderReranker:
    """Deterministic token-overlap scorer — zero dependencies, zero downloads.

    Score = number of shared word tokens between query and candidate text.
    This is sufficient to test that the correct candidate rises to rank 1 when
    it contains more query keywords.
    """

    def score_pairs(self, query: str, candidate_texts: list[str]) -> list[float]:
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        scores: list[float] = []
        for text in candidate_texts:
            t_tokens = set(re.findall(r"\b\w+\b", text.lower()))
            scores.append(float(len(q_tokens & t_tokens)))
        return scores


# ---------------------------------------------------------------------------
# Rich candidate text builder
# ---------------------------------------------------------------------------

_MAX_CANDIDATE_CHARS = 450


def build_candidate_text(
    hit,  # HybridHit — duck-typed to avoid circular import
    source_evidence_index: "SourceEvidenceIndex | None" = None,
    *,
    max_chars: int = _MAX_CANDIDATE_CHARS,
) -> str:
    """Build a rich text passage for the cross-encoder.

    Combines claim, original_text, compressed label summary, session/speaker
    metadata header, and optionally ±2 context turns from the source evidence
    index.  Truncated to *max_chars* to stay within cross-encoder token limits.
    """
    mu = hit.mu
    parts: list[str] = []

    # Metadata header — gives the model temporal/speaker grounding
    meta: list[str] = []
    if mu.session_id:
        meta.append(f"Session:{mu.session_id}")
    if mu.source_speaker:
        meta.append(f"Speaker:{mu.source_speaker}")
    if mu.timestamp:
        ts = str(mu.timestamp)
        meta.append(f"Time:{ts[:10]}")
    if meta:
        parts.append("[" + " | ".join(meta) + "]")

    claim = (mu.claim or "").strip()
    original = (mu.original_text or "").strip()
    label_sum = (getattr(hit, "label_summary", None) or "").strip()

    if claim:
        parts.append(claim)
    if original and original not in claim:
        parts.append(original)
    if label_sum and label_sum not in claim and label_sum not in original:
        parts.append(label_sum)

    # Append ±2 context turns when available — most helpful for SE hits
    if source_evidence_index is not None and mu.source_dia_ids:
        for dia_id in mu.source_dia_ids[:1]:
            ctx = source_evidence_index.get_context_text(dia_id, window=2)
            if ctx:
                ctx = ctx.strip()
                if ctx not in claim and ctx not in original:
                    budget = max_chars - sum(len(p) + 1 for p in parts)
                    if budget > 40:
                        parts.append(ctx[:budget])

    return " ".join(parts)[:max_chars]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "CrossEncoderRerankerProtocol",
    "FakeCrossEncoderReranker",
    "SentenceTransformersCrossEncoderReranker",
    "build_candidate_text",
]
