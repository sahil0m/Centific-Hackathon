"""Salience Scorer — Phase 2.

Research basis
--------------
Ebbinghaus forgetting curve applied to LLM memory
(MemoryBank, Zhong et al. 2023):

    retrievability = e^(-t / S)

where:
    t = days since last access (or creation if never accessed)
    S = memory stability = base_stability × 2^min(retrieval_count, 10)

Each retrieval doubles the stability, so a frequently-retrieved fact decays
much more slowly than one that was never needed.

Combined with topic importance following the Generative Agents structure
(Park et al. 2023):

    salience = 0.60 × ebbinghaus + 0.40 × importance − graph_penalty

Graph penalty is supplied by the caller (lifecycle engine) using edges
already written by the ContradictionResolver:
    SUPERSEDED_BY edge on this MU  →  −0.30  (fact is outdated)
    CONFLICTS_WITH edge on this MU →  −0.10  (fact is contested)
    penalty is capped at 0.40 so salience never goes below 0.

The lifecycle engine sorts candidates by salience (ascending) to decide
which MUs to evict first when the active store exceeds capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from locomo_memory.phase2.schemas import MemoryUnit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIEVAL_CAP: int = 10
_EBBINGHAUS_WEIGHT: float = 0.60
_IMPORTANCE_WEIGHT: float = 0.40
_DEFAULT_BASE_STABILITY: float = 2.0
_MAX_GRAPH_PENALTY: float = 0.40


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SalienceResult:
    """Full breakdown of a single salience computation.

    Attributes:
        mu_id: ID of the scored MemoryUnit.
        salience: final score in [0, 1].
        graph_penalty: penalty applied from contradiction-resolver edges.
        sub_scores: {"ebbinghaus": float, "importance": float}
    """

    mu_id: str
    salience: float
    graph_penalty: float
    sub_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class SalienceScorer:
    """Compute salience scores using the Ebbinghaus forgetting curve.

    Args:
        base_stability: memory stability S for a never-retrieved fact.
            Each retrieval doubles S (up to 2^10 cap), so a fact retrieved
            N times has S = base_stability × 2^min(N, 10).
            Default 2.0 gives a half-life of ~1.4 days for a new fact and
            ~44 days after 5 retrievals.
    """

    def __init__(self, *, base_stability: float = _DEFAULT_BASE_STABILITY) -> None:
        if base_stability <= 0.0:
            raise ValueError(f"base_stability must be > 0, got {base_stability}")
        self.base_stability = base_stability

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        mu: MemoryUnit,
        *,
        now: datetime | None = None,
        graph_penalty: float = 0.0,
    ) -> float:
        """Return salience score in [0, 1].

        Args:
            mu: the MemoryUnit to score.
            now: reference timestamp (defaults to UTC now).
            graph_penalty: pre-computed penalty from contradiction-resolver
                edges (0.30 for SUPERSEDED_BY, 0.10 for CONFLICTS_WITH).
                Capped internally at 0.40.
        """
        return self._compute(mu, now=_resolve_now(now), graph_penalty=graph_penalty).salience

    def score_and_update(
        self,
        mu: MemoryUnit,
        *,
        now: datetime | None = None,
        graph_penalty: float = 0.0,
    ) -> float:
        """Compute salience and write the result back to ``mu.salience_score``."""
        s = self.score(mu, now=now, graph_penalty=graph_penalty)
        mu.salience_score = s
        return s

    def detail(
        self,
        mu: MemoryUnit,
        *,
        now: datetime | None = None,
        graph_penalty: float = 0.0,
    ) -> SalienceResult:
        """Return a full :class:`SalienceResult` with sub-scores."""
        return self._compute(mu, now=_resolve_now(now), graph_penalty=graph_penalty)

    def rank(
        self,
        mus: list[MemoryUnit],
        *,
        now: datetime | None = None,
        penalties: dict[str, float] | None = None,
    ) -> list[MemoryUnit]:
        """Return MUs sorted highest-first by salience.

        Args:
            penalties: optional mapping of mu_id → graph_penalty pre-computed
                from contradiction-resolver edges.
        """
        resolved = _resolve_now(now)
        p = penalties or {}
        return sorted(
            mus,
            key=lambda mu: self._compute(mu, now=resolved, graph_penalty=p.get(mu.mu_id, 0.0)).salience,
            reverse=True,
        )

    def candidates_for_compression(
        self,
        mus: list[MemoryUnit],
        *,
        threshold: float = 0.4,
        now: datetime | None = None,
        penalties: dict[str, float] | None = None,
    ) -> list[MemoryUnit]:
        """Return MUs below ``threshold``, sorted ascending (evict first).

        Pinned MUs are always excluded regardless of score.

        Args:
            threshold: salience ceiling for inclusion (exclusive upper bound).
            penalties: optional mapping of mu_id → graph_penalty.

        Raises:
            ValueError: if threshold is outside [0, 1].
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        resolved = _resolve_now(now)
        p = penalties or {}
        candidates: list[tuple[float, MemoryUnit]] = []
        for mu in mus:
            if mu.user_pinned:
                continue
            s = self._compute(mu, now=resolved, graph_penalty=p.get(mu.mu_id, 0.0)).salience
            if s < threshold:
                candidates.append((s, mu))
        return [mu for _, mu in sorted(candidates, key=lambda t: t[0])]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute(
        self,
        mu: MemoryUnit,
        *,
        now: datetime,
        graph_penalty: float,
    ) -> SalienceResult:
        # --- Ebbinghaus: retrievability = e^(-t / S) ---
        # Defensive: both timestamps are normally set by the schema's
        # ``default_factory=utcnow``, but a manually-constructed MU or a
        # corrupt DB row could have them as None.  Rather than crashing
        # with AttributeError on ``ref.tzinfo``, fall back to ``now`` which
        # gives ebbinghaus = 1.0 (treated as freshly-created).  Same idea
        # for clock skew: if ``ref`` is in the future, ``max(0.0, …)``
        # already clamps t to 0 below, so we don't need a separate guard.
        ref = mu.last_accessed or mu.created_at or now
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        t = max(0.0, (now - ref).total_seconds() / 86_400.0)
        # Retrieval count must be a non-negative integer; clamp defensively
        # against negative values that would invert the stability curve.
        rc = max(0, int(mu.retrieval_count or 0))
        S = self.base_stability * (2.0 ** min(rc, _MAX_RETRIEVAL_CAP))
        ebbinghaus = math.exp(-t / S)

        # --- Topic importance (set by TopicImportanceEstimator at ingestion) ---
        importance = max(0.0, min(1.0, mu.importance))

        # --- Graph penalty (contradiction resolver edges) ---
        penalty = max(0.0, min(_MAX_GRAPH_PENALTY, graph_penalty))

        # --- Combine ---
        raw = _EBBINGHAUS_WEIGHT * ebbinghaus + _IMPORTANCE_WEIGHT * importance
        salience = round(max(0.0, min(1.0, raw - penalty)), 4)

        return SalienceResult(
            mu_id=mu.mu_id,
            salience=salience,
            graph_penalty=penalty,
            sub_scores={
                "ebbinghaus": round(ebbinghaus, 4),
                "importance": round(importance, 4),
            },
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _resolve_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = ["SalienceResult", "SalienceScorer"]
