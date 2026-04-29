"""Lifecycle Engine — Phase 2 Milestone 5.

Owns the active-memory capacity trigger and orchestrates state transitions
when a conversation's memory store approaches its configured cap.

Transition policy
-----------------
1. Check ``store.storage_pressure(conv_id, config.active_cap)``.
2. If pressure < ``config.transition_trigger_pct`` (default 0.90) → no-op.
3. Score every active MU using :class:`~locomo_memory.phase2.salience.SalienceScorer`.
4. Rank non-pinned MUs by salience ascending (lowest first).
5. Transition MUs until pressure drops below ``config.target_pressure_pct`` (0.70):
   - salience < ``config.salience_forget_threshold`` (0.15) → FORGOTTEN
   - else                                                   → COMPRESSED
6. Compressed MUs get a rule-based :class:`CompressedLabel` + :class:`ArchivedEntry`
   (built by :class:`LabelBuilder`, no LLM call needed).

Capacity trigger constant
-------------------------
    transition_trigger_pct = 0.90  (fire at 90 % fill)
    target_pressure_pct    = 0.70  (compress until below 70 %)

User-pinned MUs are never transitioned by the engine, regardless of salience.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

from loguru import logger

from locomo_memory.phase2.ingestion.importance import TopicImportanceEstimator
from locomo_memory.phase2.salience.scorer import SalienceScorer, SalienceWeights
from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    MemoryStatus,
    MemoryUnit,
    new_archive_id,
    new_label_id,
    utcnow,
)
from locomo_memory.phase2.store.sqlite_store import (
    IllegalStateTransitionError,
    MemoryStore,
    MemoryStoreError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRIGGER_PCT: Final[float] = 0.90
_TARGET_PCT: Final[float] = 0.70
_FORGET_THRESHOLD: Final[float] = 0.15
_COMPRESS_THRESHOLD: Final[float] = 0.40
_DEFAULT_CAP: Final[int] = 500

_SUMMARY_MAX_CHARS: Final[int] = 120


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleConfig:
    """Tunable parameters for the Lifecycle Engine.

    All percentages are in [0, 1].
    """

    active_cap: int = _DEFAULT_CAP
    """Hard cap on active MUs per conversation. Storage pressure = active / cap."""

    transition_trigger_pct: float = _TRIGGER_PCT
    """Fire a compression pass when pressure reaches this fraction."""

    target_pressure_pct: float = _TARGET_PCT
    """Run the pass until pressure drops below this fraction."""

    salience_forget_threshold: float = _FORGET_THRESHOLD
    """MUs with salience below this go directly to FORGOTTEN (not COMPRESSED)."""

    salience_compress_threshold: float = _COMPRESS_THRESHOLD
    """MUs with salience in [forget_threshold, compress_threshold) are compressed."""

    def __post_init__(self) -> None:
        if self.active_cap < 1:
            raise ValueError(f"active_cap must be >= 1, got {self.active_cap}")
        for name, val in [
            ("transition_trigger_pct", self.transition_trigger_pct),
            ("target_pressure_pct", self.target_pressure_pct),
            ("salience_forget_threshold", self.salience_forget_threshold),
            ("salience_compress_threshold", self.salience_compress_threshold),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if self.salience_forget_threshold >= self.salience_compress_threshold:
            raise ValueError(
                "salience_forget_threshold must be < salience_compress_threshold"
            )
        if self.target_pressure_pct >= self.transition_trigger_pct:
            raise ValueError(
                "target_pressure_pct must be < transition_trigger_pct"
            )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransitionRecord:
    """A single MU state change executed by the engine."""

    mu_id: str
    from_status: MemoryStatus
    to_status: MemoryStatus
    salience: float
    reason: str  # "forget" | "compress"


@dataclass(slots=True)
class LifecycleBatch:
    """Summary of one compression pass for a conversation."""

    conversation_id: str
    triggered: bool
    """False when pressure was below trigger_pct and no pass was run."""

    pressure_before: float
    pressure_after: float
    n_scored: int
    transitions: list[TransitionRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_forgotten(self) -> int:
        return sum(1 for t in self.transitions if t.to_status == MemoryStatus.FORGOTTEN)

    @property
    def n_compressed(self) -> int:
        return sum(1 for t in self.transitions if t.to_status == MemoryStatus.COMPRESSED)


# ---------------------------------------------------------------------------
# Rule-based label builder (no LLM)
# ---------------------------------------------------------------------------


_ENTITY_RE = re.compile(r"(?<![.!?\n])\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b")
_estimator = TopicImportanceEstimator()


class LabelBuilder:
    """Build a CompressedLabel + ArchivedEntry for an active MemoryUnit.

    Uses rule-based topic detection and heuristic entity extraction.
    No LLM call required.
    """

    def build(self, mu: MemoryUnit) -> tuple[CompressedLabel, ArchivedEntry]:
        """Return (label, archive) ready for ``store.compress_atomic``."""
        archive_entry_id = new_archive_id()

        label = CompressedLabel(
            archived_pointer=archive_entry_id,
            mu_id=mu.mu_id,
            conversation_id=mu.conversation_id,
            topic=_estimator.detect_topic(mu.claim),
            short_summary=mu.claim[:_SUMMARY_MAX_CHARS],
            key_entities=_estimator.extract_entities(mu.claim),
            time_range=mu.timestamp,
            original_dia_ids=list(mu.source_dia_ids),
        )

        archive = ArchivedEntry(
            archived_entry_id=archive_entry_id,
            label_pointer=label.label_id,
            mu_id=mu.mu_id,
            conversation_id=mu.conversation_id,
            full_memory_unit_json=mu.model_dump_json(),
            full_original_text=mu.original_text,
        )

        return label, archive


# ---------------------------------------------------------------------------
# Lifecycle Engine
# ---------------------------------------------------------------------------


class LifecycleEngine:
    """Orchestrates active-memory capacity management for SPARC-LTM.

    Args:
        store: the SQLite-backed :class:`MemoryStore`.
        scorer: the :class:`SalienceScorer` used to rank MUs. Pass ``None``
            to use a default scorer with balanced weights.
        config: lifecycle tuning parameters. Pass ``None`` for defaults
            (trigger at 90 %, target 75 %, cap 500).
        label_builder: rule-based label factory. Pass ``None`` to use the
            default :class:`LabelBuilder`.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        scorer: SalienceScorer | None = None,
        config: LifecycleConfig | None = None,
        label_builder: LabelBuilder | None = None,
    ) -> None:
        self.store = store
        self.scorer = scorer or SalienceScorer()
        self.config = config or LifecycleConfig()
        self.label_builder = label_builder or LabelBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pressure(self, conversation_id: str) -> float:
        """Return current storage pressure (active_count / active_cap)."""
        return self.store.storage_pressure(conversation_id, self.config.active_cap)

    def maybe_run(
        self,
        conversation_id: str,
        *,
        now: datetime | None = None,
    ) -> LifecycleBatch:
        """Run a compression pass only if pressure >= trigger threshold.

        Returns a :class:`LifecycleBatch` with ``triggered=False`` when the
        store is below the threshold and no action was taken.
        """
        p = self.pressure(conversation_id)
        if p < self.config.transition_trigger_pct:
            logger.debug(
                "Lifecycle skip conv={}: pressure={:.2%} < trigger={:.2%}",
                conversation_id, p, self.config.transition_trigger_pct,
            )
            return LifecycleBatch(
                conversation_id=conversation_id,
                triggered=False,
                pressure_before=p,
                pressure_after=p,
                n_scored=0,
            )
        return self.run_pass(conversation_id, now=now)

    def run_pass(
        self,
        conversation_id: str,
        *,
        now: datetime | None = None,
    ) -> LifecycleBatch:
        """Unconditionally run one compression pass for a conversation.

        Scores all active MUs, then transitions the lowest-salience
        non-pinned ones until pressure drops below ``target_pressure_pct``.
        """
        resolved = _resolve_now(now)
        active_mus = self.store.list_active(conversation_id)
        pressure_before = len(active_mus) / max(1, self.config.active_cap)

        n_scored = len(active_mus)
        scored: list[tuple[float, MemoryUnit]] = [
            (self.scorer.score(mu, now=resolved), mu)
            for mu in active_mus
        ]
        # Sort ascending: lowest salience first = highest compression priority
        scored.sort(key=lambda t: t[0])

        n_active = len(active_mus)
        target_active = math.floor(self.config.target_pressure_pct * self.config.active_cap)
        n_to_transition = max(0, n_active - target_active)

        # Only non-pinned candidates
        candidates = [
            (s, mu) for s, mu in scored if not mu.user_pinned
        ][:n_to_transition]

        transitions: list[TransitionRecord] = []
        errors: list[str] = []

        for salience, mu in candidates:
            try:
                if salience < self.config.salience_forget_threshold:
                    self._forget(mu, salience, transitions)
                else:
                    self._compress(mu, salience, transitions)
            except (MemoryStoreError, IllegalStateTransitionError, Exception) as exc:
                msg = f"transition failed for {mu.mu_id}: {exc}"
                logger.warning("Lifecycle: {}", msg)
                errors.append(msg)

        n_transitioned = len(transitions)
        n_remaining = n_active - n_transitioned
        pressure_after = n_remaining / max(1, self.config.active_cap)

        logger.info(
            "Lifecycle pass conv={}: scored={} forgotten={} compressed={} "
            "pressure {:.2%} → {:.2%}",
            conversation_id, n_scored,
            sum(1 for t in transitions if t.to_status == MemoryStatus.FORGOTTEN),
            sum(1 for t in transitions if t.to_status == MemoryStatus.COMPRESSED),
            pressure_before, pressure_after,
        )

        return LifecycleBatch(
            conversation_id=conversation_id,
            triggered=True,
            pressure_before=pressure_before,
            pressure_after=pressure_after,
            n_scored=n_scored,
            transitions=transitions,
            errors=errors,
        )

    def score_and_update_all(
        self,
        conversation_id: str,
        *,
        now: datetime | None = None,
    ) -> int:
        """Re-score all active MUs and persist their updated salience_score.

        Returns the number of MUs updated. Call this periodically (e.g., once
        per new session) to keep salience scores fresh as recency decays.
        """
        resolved = _resolve_now(now)
        updated = 0
        for mu in self.store.list_active(conversation_id):
            self.scorer.score_and_update(mu, now=resolved)
            self.store.update_memory_unit(mu)
            updated += 1
        logger.debug(
            "score_and_update_all conv={}: {} MUs re-scored", conversation_id, updated
        )
        return updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forget(
        self,
        mu: MemoryUnit,
        salience: float,
        transitions: list[TransitionRecord],
    ) -> None:
        self.store.forget_atomic(mu.mu_id)
        transitions.append(TransitionRecord(
            mu_id=mu.mu_id,
            from_status=MemoryStatus.ACTIVE,
            to_status=MemoryStatus.FORGOTTEN,
            salience=salience,
            reason="forget",
        ))
        logger.debug("Lifecycle: forgot mu={} salience={:.3f}", mu.mu_id, salience)

    def _compress(
        self,
        mu: MemoryUnit,
        salience: float,
        transitions: list[TransitionRecord],
    ) -> None:
        label, archive = self.label_builder.build(mu)
        self.store.compress_atomic(mu.mu_id, label, archive)
        transitions.append(TransitionRecord(
            mu_id=mu.mu_id,
            from_status=MemoryStatus.ACTIVE,
            to_status=MemoryStatus.COMPRESSED,
            salience=salience,
            reason="compress",
        ))
        logger.debug(
            "Lifecycle: compressed mu={} salience={:.3f} label={}",
            mu.mu_id, salience, label.label_id,
        )


# ---------------------------------------------------------------------------
# Utility
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

__all__ = [
    "LabelBuilder",
    "LifecycleBatch",
    "LifecycleConfig",
    "LifecycleEngine",
    "TransitionRecord",
]
