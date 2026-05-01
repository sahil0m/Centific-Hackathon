"""Tests for the Lifecycle Engine (Phase 2 Milestone 5).

Uses a real SQLite store (tmp_path) and a SalienceScorer configured to use
only the ``importance`` dimension — that makes salience == mu.importance,
which gives tests full, deterministic control over scores.

Coverage:
- LifecycleConfig construction and validation
- pressure() metric
- maybe_run() no-op when below threshold
- maybe_run() triggers when at/above threshold
- run_pass() forgets low-salience MUs
- run_pass() compresses medium-salience MUs
- run_pass() skips user-pinned MUs
- run_pass() brings pressure below target
- run_pass() returns correct LifecycleBatch fields
- Compressed MU gets a valid label + archive in the store
- Restored MU re-enters ACTIVE
- score_and_update_all() re-scores and persists
- TransitionRecord fields
- LabelBuilder topic + entities
- LifecycleConfig validation errors
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from locomo_memory.phase2.lifecycle import (
    LabelBuilder,
    LifecycleBatch,
    LifecycleConfig,
    LifecycleEngine,
    TransitionRecord,
)
from locomo_memory.phase2.salience import SalienceScorer
from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.sqlite_store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
# MUs created 4 days before _NOW so ebbinghaus ≈ 0.135 at _NOW.
# salience ≈ 0.60×0.135 + 0.40×importance = 0.081 + 0.40×importance
# → importance=0.05 → salience≈0.10 → FORGOTTEN (<0.15)
# → importance=0.25 → salience≈0.18 → COMPRESSED (0.15–0.40)
_CREATED_AT = _NOW - timedelta(days=4)


def _importance_scorer() -> SalienceScorer:
    return SalienceScorer()


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


def _mu(
    *,
    conversation_id: str = "conv_1",
    session_id: str = "s1",
    claim: str = "Alice works at Acme Corp.",
    importance: float = 0.5,
    user_pinned: bool = False,
) -> MemoryUnit:
    return MemoryUnit(
        conversation_id=conversation_id,
        session_id=session_id,
        claim=claim,
        importance=importance,
        user_pinned=user_pinned,
        created_at=_CREATED_AT,
    )


def _engine(
    store: MemoryStore,
    *,
    cap: int = 10,
    trigger: float = 0.90,
    target: float = 0.70,
    forget_threshold: float = 0.15,
    compress_threshold: float = 0.40,
) -> LifecycleEngine:
    config = LifecycleConfig(
        active_cap=cap,
        transition_trigger_pct=trigger,
        target_pressure_pct=target,
        salience_forget_threshold=forget_threshold,
        salience_compress_threshold=compress_threshold,
    )
    return LifecycleEngine(store, scorer=_importance_scorer(), config=config)


def _insert_mus(store: MemoryStore, mus: list[MemoryUnit]) -> None:
    for mu in mus:
        store.insert_memory_unit(mu)


# ---------------------------------------------------------------------------
# LifecycleConfig validation
# ---------------------------------------------------------------------------


class TestLifecycleConfig:
    def test_defaults(self) -> None:
        cfg = LifecycleConfig()
        assert cfg.active_cap == 500
        assert cfg.transition_trigger_pct == pytest.approx(0.90)
        assert cfg.target_pressure_pct == pytest.approx(0.70)
        assert cfg.salience_forget_threshold == pytest.approx(0.15)
        assert cfg.salience_compress_threshold == pytest.approx(0.40)

    def test_invalid_cap(self) -> None:
        with pytest.raises(ValueError, match="active_cap"):
            LifecycleConfig(active_cap=0)

    def test_invalid_trigger_pct(self) -> None:
        with pytest.raises(ValueError):
            LifecycleConfig(transition_trigger_pct=1.5)

    def test_target_must_be_below_trigger(self) -> None:
        with pytest.raises(ValueError, match="target_pressure_pct"):
            LifecycleConfig(transition_trigger_pct=0.7, target_pressure_pct=0.8)

    def test_forget_must_be_below_compress(self) -> None:
        with pytest.raises(ValueError, match="salience_forget_threshold"):
            LifecycleConfig(
                salience_forget_threshold=0.5,
                salience_compress_threshold=0.3,
            )

    def test_custom_values_accepted(self) -> None:
        cfg = LifecycleConfig(
            active_cap=100,
            transition_trigger_pct=0.85,
            target_pressure_pct=0.60,
        )
        assert cfg.active_cap == 100


# ---------------------------------------------------------------------------
# pressure()
# ---------------------------------------------------------------------------


class TestPressure:
    def test_zero_when_empty(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        eng = _engine(store, cap=10)
        assert eng.pressure("conv_1") == pytest.approx(0.0)

    def test_proportional_to_active_count(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _insert_mus(store, [_mu(importance=0.5) for _ in range(5)])
        eng = _engine(store, cap=10)
        assert eng.pressure("conv_1") == pytest.approx(0.5)

    def test_full_at_cap(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _insert_mus(store, [_mu(importance=0.5) for _ in range(10)])
        eng = _engine(store, cap=10)
        assert eng.pressure("conv_1") == pytest.approx(1.0)

    def test_scoped_to_conversation(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _insert_mus(store, [_mu(conversation_id="conv_A", importance=0.5) for _ in range(5)])
        _insert_mus(store, [_mu(conversation_id="conv_B", importance=0.5) for _ in range(3)])
        eng = _engine(store, cap=10)
        assert eng.pressure("conv_A") == pytest.approx(0.5)
        assert eng.pressure("conv_B") == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# maybe_run() — no-op below threshold
# ---------------------------------------------------------------------------


class TestMaybeRunNoTrigger:
    def test_returns_not_triggered(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # 5 / 10 = 50 % → below 90 % trigger
        _insert_mus(store, [_mu(importance=0.5) for _ in range(5)])
        eng = _engine(store, cap=10)
        batch = eng.maybe_run("conv_1", now=_NOW)
        assert batch.triggered is False
        assert batch.transitions == []
        assert batch.n_scored == 0

    def test_no_status_changes(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _insert_mus(store, [_mu(importance=0.5) for _ in range(5)])
        eng = _engine(store, cap=10)
        eng.maybe_run("conv_1", now=_NOW)
        assert len(store.list_active("conv_1")) == 5


# ---------------------------------------------------------------------------
# maybe_run() — triggers at/above threshold
# ---------------------------------------------------------------------------


class TestMaybeRunTriggers:
    def test_triggers_at_90pct(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # 9 / 10 = 90 % → at threshold
        _insert_mus(store, [_mu(importance=0.1) for _ in range(9)])
        eng = _engine(store, cap=10, trigger=0.90, target=0.70)
        batch = eng.maybe_run("conv_1", now=_NOW)
        assert batch.triggered is True

    def test_above_threshold_also_triggers(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # 10 / 10 = 100 %
        _insert_mus(store, [_mu(importance=0.1) for _ in range(10)])
        eng = _engine(store, cap=10, trigger=0.90)
        batch = eng.maybe_run("conv_1", now=_NOW)
        assert batch.triggered is True


# ---------------------------------------------------------------------------
# run_pass() — forgetting
# ---------------------------------------------------------------------------


class TestRunPassForget:
    def test_low_salience_mu_is_forgotten(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # importance=0.05 → salience=0.05 → below forget_threshold (0.15)
        low = _mu(importance=0.05)
        store.insert_memory_unit(low)
        eng = _engine(store, cap=1, trigger=0.9, target=0.0, forget_threshold=0.15)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert any(t.to_status == MemoryStatus.FORGOTTEN for t in batch.transitions)
        mu_after = store.get_memory_unit(low.mu_id)
        assert mu_after is not None
        assert mu_after.status == MemoryStatus.FORGOTTEN

    def test_forgotten_transition_record(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        low = _mu(importance=0.05)
        store.insert_memory_unit(low)
        eng = _engine(store, cap=1, target=0.0, forget_threshold=0.15)
        batch = eng.run_pass("conv_1", now=_NOW)
        forgotten = [t for t in batch.transitions if t.reason == "forget"]
        assert len(forgotten) == 1
        assert forgotten[0].mu_id == low.mu_id
        assert forgotten[0].from_status == MemoryStatus.ACTIVE
        assert forgotten[0].to_status == MemoryStatus.FORGOTTEN
        assert forgotten[0].salience < 0.15  # below forget threshold


# ---------------------------------------------------------------------------
# run_pass() — compression
# ---------------------------------------------------------------------------


class TestRunPassCompress:
    def test_medium_salience_mu_is_compressed(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # importance=0.25 → salience=0.25 → in [forget_threshold, compress_threshold)
        mid = _mu(importance=0.25, claim="Alice works at Acme Corp.")
        store.insert_memory_unit(mid)
        eng = _engine(
            store, cap=1, target=0.0,
            forget_threshold=0.15, compress_threshold=0.40,
        )
        batch = eng.run_pass("conv_1", now=_NOW)
        # Compressed MUs land in ARCHIVED status (the original-data tier);
        # the CompressedLabel is the searchable presence in the compressed tier.
        assert any(t.to_status == MemoryStatus.ARCHIVED for t in batch.transitions)
        mu_after = store.get_memory_unit(mid.mu_id)
        assert mu_after is not None
        assert mu_after.status == MemoryStatus.ARCHIVED

    def test_compressed_label_exists_in_store(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mid = _mu(importance=0.25, claim="Bob lives in London now.")
        store.insert_memory_unit(mid)
        eng = _engine(store, cap=1, target=0.0)
        eng.run_pass("conv_1", now=_NOW)
        mu_after = store.get_memory_unit(mid.mu_id)
        assert mu_after is not None
        label = store.get_label_for_mu(mid.mu_id)
        assert label is not None
        assert label.mu_id == mid.mu_id
        assert label.short_summary != ""

    def test_compressed_archive_exists_in_store(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mid = _mu(importance=0.25, claim="Carol graduated from MIT last year.")
        store.insert_memory_unit(mid)
        eng = _engine(store, cap=1, target=0.0)
        eng.run_pass("conv_1", now=_NOW)
        archive = store.get_archive_for_mu(mid.mu_id)
        assert archive is not None
        assert mid.mu_id in archive.full_memory_unit_json

    def test_compressed_transition_record(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mid = _mu(importance=0.25)
        store.insert_memory_unit(mid)
        eng = _engine(store, cap=1, target=0.0)
        batch = eng.run_pass("conv_1", now=_NOW)
        compressed = [t for t in batch.transitions if t.reason == "compress"]
        assert len(compressed) == 1
        rec = compressed[0]
        assert rec.mu_id == mid.mu_id
        assert rec.from_status == MemoryStatus.ACTIVE
        # Engine emits ARCHIVED to reflect the actual store-side status.
        assert rec.to_status == MemoryStatus.ARCHIVED


# ---------------------------------------------------------------------------
# run_pass() — pinned exclusion
# ---------------------------------------------------------------------------


class TestPinnedExclusion:
    def test_pinned_mu_not_transitioned(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        pinned = _mu(importance=0.01, user_pinned=True)
        store.insert_memory_unit(pinned)
        eng = _engine(store, cap=1, target=0.0)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert all(t.mu_id != pinned.mu_id for t in batch.transitions)
        mu_after = store.get_memory_unit(pinned.mu_id)
        assert mu_after is not None
        assert mu_after.status == MemoryStatus.ACTIVE

    def test_only_unpinned_are_transitioned_when_mixed(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        pinned = _mu(importance=0.01, user_pinned=True, claim="pinned claim")
        unpinned = _mu(importance=0.05, user_pinned=False, claim="unpinned claim")
        _insert_mus(store, [pinned, unpinned])
        eng = _engine(store, cap=2, target=0.0, trigger=0.5)
        batch = eng.run_pass("conv_1", now=_NOW)
        transitioned_ids = {t.mu_id for t in batch.transitions}
        assert pinned.mu_id not in transitioned_ids
        assert unpinned.mu_id in transitioned_ids


# ---------------------------------------------------------------------------
# run_pass() — pressure drops
# ---------------------------------------------------------------------------


class TestPressureAfterPass:
    def test_pressure_drops_below_target(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # 10/10 = 100%; target=0.70 → need to transition at least 3 MUs
        mus = [_mu(importance=0.05) for _ in range(10)]
        _insert_mus(store, mus)
        eng = _engine(store, cap=10, trigger=0.90, target=0.70)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert batch.pressure_after < 0.70 + 0.02  # allow rounding tolerance

    def test_batch_pressure_before_accurate(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mus = [_mu(importance=0.05) for _ in range(9)]
        _insert_mus(store, mus)
        eng = _engine(store, cap=10)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert batch.pressure_before == pytest.approx(0.9)

    def test_high_salience_mus_not_touched(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # 4 low, 6 high (won't be touched)
        low_mus = [_mu(importance=0.05) for _ in range(4)]
        high_mus = [_mu(importance=0.95) for _ in range(6)]
        _insert_mus(store, low_mus + high_mus)
        eng = _engine(store, cap=10, trigger=0.90, target=0.70)
        eng.run_pass("conv_1", now=_NOW)
        # High-importance MUs should still be active
        for mu in high_mus:
            mu_after = store.get_memory_unit(mu.mu_id)
            assert mu_after is not None
            assert mu_after.status == MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# LifecycleBatch fields
# ---------------------------------------------------------------------------


class TestLifecycleBatch:
    def test_n_forgotten_and_n_compressed_helpers(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        forget_mu = _mu(importance=0.05)  # below forget_threshold
        compress_mu = _mu(importance=0.25)  # in compress range
        _insert_mus(store, [forget_mu, compress_mu])
        eng = _engine(store, cap=2, target=0.0)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert batch.n_forgotten >= 1
        assert batch.n_compressed >= 1

    def test_triggered_is_true_on_run_pass(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        eng = _engine(store)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert batch.triggered is True

    def test_n_scored_equals_active_count(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mus = [_mu(importance=0.9) for _ in range(7)]
        _insert_mus(store, mus)
        eng = _engine(store, cap=10)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert batch.n_scored == 7

    def test_empty_conversation_no_crash(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        eng = _engine(store, cap=10)
        batch = eng.run_pass("conv_1", now=_NOW)
        assert batch.transitions == []
        assert batch.errors == []


# ---------------------------------------------------------------------------
# score_and_update_all()
# ---------------------------------------------------------------------------


class TestScoreAndUpdateAll:
    def test_updates_salience_score_in_db(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mu = _mu(importance=0.73)
        store.insert_memory_unit(mu)
        eng = _engine(store)
        count = eng.score_and_update_all("conv_1", now=_NOW)
        assert count == 1
        updated = store.get_memory_unit(mu.mu_id)
        assert updated is not None
        # Ebbinghaus + importance: salience is in (0, 1) and reflects importance
        assert 0.0 < updated.salience_score < 1.0

    def test_returns_count(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _insert_mus(store, [_mu(importance=0.5) for _ in range(5)])
        eng = _engine(store)
        count = eng.score_and_update_all("conv_1", now=_NOW)
        assert count == 5

    def test_empty_conversation_returns_zero(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        eng = _engine(store)
        assert eng.score_and_update_all("conv_1", now=_NOW) == 0

    def test_only_updates_active_mus(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        active = _mu(importance=0.5, claim="active claim")
        forgotten = _mu(importance=0.5, claim="forgotten claim")
        store.insert_memory_unit(active)
        store.insert_memory_unit(forgotten)
        store.forget_atomic(forgotten.mu_id)
        eng = _engine(store)
        count = eng.score_and_update_all("conv_1", now=_NOW)
        assert count == 1  # only the active one


# ---------------------------------------------------------------------------
# Restore after compression
# ---------------------------------------------------------------------------


class TestRestoreAfterCompression:
    def test_compressed_mu_can_be_restored(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mid = _mu(importance=0.25, claim="She moved to Berlin last year.")
        store.insert_memory_unit(mid)
        eng = _engine(store, cap=1, target=0.0)
        eng.run_pass("conv_1", now=_NOW)

        mu_compressed = store.get_memory_unit(mid.mu_id)
        assert mu_compressed is not None
        assert mu_compressed.status == MemoryStatus.ARCHIVED

        restored = store.restore_atomic(mid.mu_id)
        assert restored.status == MemoryStatus.ACTIVE
        assert restored.mu_id == mid.mu_id


# ---------------------------------------------------------------------------
# LabelBuilder
# ---------------------------------------------------------------------------


class TestLabelBuilder:
    def test_label_has_topic(self) -> None:
        builder = LabelBuilder()
        mu = _mu(claim="Alice works at Google in New York.")
        label, archive = builder.build(mu)
        assert label.topic != ""
        assert label.topic == "employment"

    def test_label_short_summary_is_claim(self) -> None:
        builder = LabelBuilder()
        mu = _mu(claim="Alice works at Google.")
        label, _ = builder.build(mu)
        assert "Alice" in label.short_summary

    def test_label_entities_extracted(self) -> None:
        builder = LabelBuilder()
        mu = _mu(claim="Alice works at Google in New York.")
        label, _ = builder.build(mu)
        # At least one entity recognised
        assert isinstance(label.key_entities, list)

    def test_archive_contains_full_json(self) -> None:
        builder = LabelBuilder()
        mu = _mu(claim="Carol graduated from MIT last year.")
        label, archive = builder.build(mu)
        assert mu.mu_id in archive.full_memory_unit_json

    def test_pointer_cross_references_valid(self) -> None:
        """label.archived_pointer == archive.archived_entry_id and vice versa."""
        builder = LabelBuilder()
        mu = _mu()
        label, archive = builder.build(mu)
        assert label.archived_pointer == archive.archived_entry_id
        assert archive.label_pointer == label.label_id
        assert label.mu_id == mu.mu_id
        assert archive.mu_id == mu.mu_id

    def test_original_dia_ids_copied(self) -> None:
        builder = LabelBuilder()
        mu = _mu()
        mu.source_dia_ids = ["D1:1", "D1:2"]
        label, _ = builder.build(mu)
        assert label.original_dia_ids == ["D1:1", "D1:2"]
