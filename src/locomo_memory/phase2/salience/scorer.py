"""Salience Scorer — Phase 2 Milestone 4.

Computes a salience score for MemoryUnit objects using a weighted combination
of sub-scores derived from MU fields. Also exposes a utility metric
(salience / storage_cost) that drives compression and forgetting decisions
when memory approaches capacity.

Sub-score dimensions
--------------------
- **importance**: mu.importance — rule-based topic importance set by
  :class:`~locomo_memory.phase2.ingestion.importance.TopicImportanceEstimator`
  at ingestion time (not a constant).
- **confidence**: mu.confidence (LLM-assigned or heuristic).
- **recency**: exponential decay from last_accessed (or created_at).
- **retrieval_frequency**: normalised mu.retrieval_count.
- **user_pinned**: binary 1/0 bonus.
- **uniqueness**: mu.uniqueness (set during ingestion).

Formula
-------
    weighted_sum = Σ weight_i * sub_score_i
    salience     = clip(weighted_sum / Σ weight_i, 0, 1)
    utility      = salience / storage_cost_factor(mu)

Weights do not need to sum to 1 — the scorer normalises them internally.

Ranking and candidate helpers
------------------------------
:meth:`rank`, :meth:`candidates_for_compression`, and :meth:`utility` are
*scoring helpers only*. They do not own or implement the capacity trigger.

The active-memory capacity trigger (fire at 90 % fill, target 75 %) belongs
to the :class:`~locomo_memory.phase2.lifecycle.engine.LifecycleEngine`, which
calls these helpers to decide which MUs to transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

from locomo_memory.phase2.schemas import MemoryUnit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HALF_LIFE_DAYS_DEFAULT: Final[float] = 30.0
_RETRIEVAL_NORM_DEFAULT: Final[int] = 10
_STORAGE_BASELINE_CHARS: Final[int] = 100  # 100-char claim → cost factor 1.0


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalienceWeights:
    """Relative per-dimension weights for salience scoring.

    Values are normalised internally so they do not need to sum to 1.
    Every weight must be ≥ 0 and at least one must be > 0.
    """

    importance: float = 0.30
    confidence: float = 0.15
    recency: float = 0.20
    retrieval_frequency: float = 0.15
    user_pinned: float = 0.10
    uniqueness: float = 0.10

    def __post_init__(self) -> None:
        values = self._values()
        if any(v < 0.0 for v in values):
            raise ValueError(
                f"All SalienceWeights must be >= 0; got {dict(zip(self._names(), values))}"
            )
        if sum(values) == 0.0:
            raise ValueError("At least one SalienceWeight must be > 0")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _names(self) -> tuple[str, ...]:
        return (
            "importance", "confidence", "recency",
            "retrieval_frequency", "user_pinned", "uniqueness",
        )

    def _values(self) -> tuple[float, ...]:
        return (
            self.importance, self.confidence, self.recency,
            self.retrieval_frequency, self.user_pinned, self.uniqueness,
        )

    @property
    def total(self) -> float:
        return sum(self._values())


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SalienceResult:
    """Full breakdown of a single salience computation."""

    mu_id: str
    salience: float
    utility: float
    storage_cost_factor: float
    sub_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class SalienceScorer:
    """Compute salience scores for MemoryUnit objects.

    Args:
        weights: per-dimension weights (normalised internally). Pass
            ``None`` to use the balanced defaults in :class:`SalienceWeights`.
        half_life_days: recency decay time constant. A MU last accessed
            ``half_life_days`` ago gets recency = 0.5.
        retrieval_normalization: retrieval count at which the retrieval
            sub-score reaches 0.5. Acts as a saturation knee.
    """

    def __init__(
        self,
        weights: SalienceWeights | None = None,
        *,
        half_life_days: float = _HALF_LIFE_DAYS_DEFAULT,
        retrieval_normalization: int = _RETRIEVAL_NORM_DEFAULT,
    ) -> None:
        if half_life_days <= 0.0:
            raise ValueError(
                f"half_life_days must be > 0, got {half_life_days}"
            )
        if retrieval_normalization < 1:
            raise ValueError(
                f"retrieval_normalization must be >= 1, got {retrieval_normalization}"
            )
        self.weights = weights or SalienceWeights()
        self.half_life_days = half_life_days
        self.retrieval_normalization = retrieval_normalization

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        mu: MemoryUnit,
        *,
        now: datetime | None = None,
    ) -> float:
        """Return salience score in [0, 1]."""
        return self._compute(mu, now=_resolve_now(now)).salience

    def score_and_update(
        self,
        mu: MemoryUnit,
        *,
        now: datetime | None = None,
    ) -> float:
        """Compute salience score and write it back to ``mu.salience_score``."""
        s = self.score(mu, now=now)
        mu.salience_score = s
        return s

    def detail(
        self,
        mu: MemoryUnit,
        *,
        now: datetime | None = None,
    ) -> SalienceResult:
        """Return a full :class:`SalienceResult` with all sub-scores."""
        return self._compute(mu, now=_resolve_now(now))

    def utility(
        self,
        mu: MemoryUnit,
        *,
        now: datetime | None = None,
    ) -> float:
        """Return utility = salience / storage_cost_factor.

        Lower utility ⟹ first candidate for compression or forgetting.
        """
        return self._compute(mu, now=_resolve_now(now)).utility

    def rank(
        self,
        mus: list[MemoryUnit],
        *,
        now: datetime | None = None,
        by_utility: bool = False,
    ) -> list[MemoryUnit]:
        """Return MUs sorted highest-first by salience (or utility)."""
        resolved = _resolve_now(now)
        key = (
            (lambda mu: self._compute(mu, now=resolved).utility)
            if by_utility
            else (lambda mu: self._compute(mu, now=resolved).salience)
        )
        return sorted(mus, key=key, reverse=True)

    def candidates_for_compression(
        self,
        mus: list[MemoryUnit],
        *,
        threshold: float = 0.4,
        now: datetime | None = None,
        by_utility: bool = False,
    ) -> list[MemoryUnit]:
        """Return MUs below ``threshold``, sorted ascending (compress first).

        This is a *scoring helper only* — it performs no capacity check and
        executes no state transitions. The Lifecycle Engine owns the 90%
        capacity trigger and calls this method to identify candidates.

        Pinned MUs are always excluded regardless of score.

        Args:
            mus: candidate pool.
            threshold: score/utility ceiling for inclusion (exclusive upper bound).
            now: reference timestamp for recency; defaults to UTC now.
            by_utility: if True, compare utility instead of salience.

        Raises:
            ValueError: if threshold is outside [0, 1].
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}"
            )
        resolved = _resolve_now(now)
        candidates: list[tuple[float, MemoryUnit]] = []
        for mu in mus:
            if mu.user_pinned:
                continue
            result = self._compute(mu, now=resolved)
            val = result.utility if by_utility else result.salience
            if val < threshold:
                candidates.append((val, mu))
        # Ascending: lowest score first (most urgently compress/forgotten).
        return [mu for _, mu in sorted(candidates, key=lambda t: t[0])]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute(self, mu: MemoryUnit, *, now: datetime) -> SalienceResult:
        w = self.weights

        sub: dict[str, float] = {
            "importance": _clamp(mu.importance),
            "confidence": _clamp(mu.confidence),
            "recency": self._recency(mu, now),
            "retrieval_frequency": self._retrieval_frequency(mu),
            "user_pinned": 1.0 if mu.user_pinned else 0.0,
            "uniqueness": _clamp(mu.uniqueness),
        }

        weighted_sum = (
            w.importance         * sub["importance"]
            + w.confidence       * sub["confidence"]
            + w.recency          * sub["recency"]
            + w.retrieval_frequency * sub["retrieval_frequency"]
            + w.user_pinned      * sub["user_pinned"]
            + w.uniqueness       * sub["uniqueness"]
        )

        salience = _clamp(weighted_sum / w.total)
        cost = self._storage_cost(mu)
        utility = salience / cost

        return SalienceResult(
            mu_id=mu.mu_id,
            salience=salience,
            utility=utility,
            storage_cost_factor=cost,
            sub_scores=sub,
        )

    def _recency(self, mu: MemoryUnit, now: datetime) -> float:
        """Exponential decay from last access (or creation) to now."""
        ref = mu.last_accessed or mu.created_at
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - ref).total_seconds() / 86_400.0)
        decay_k = math.log(2.0) / self.half_life_days
        return math.exp(-decay_k * age_days)

    def _retrieval_frequency(self, mu: MemoryUnit) -> float:
        """Normalised retrieval count via n/(n+k) saturation curve."""
        n = mu.retrieval_count
        k = self.retrieval_normalization
        return n / (n + k)

    def _storage_cost(self, mu: MemoryUnit) -> float:
        """Cost factor ≥ 1.0; baseline = 100 characters of claim text."""
        return max(1.0, len(mu.claim) / _STORAGE_BASELINE_CHARS)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _resolve_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = ["SalienceResult", "SalienceScorer", "SalienceWeights"]
