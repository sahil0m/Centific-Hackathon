"""Tests for the Compression, Archive, and Restore Service (Milestone 6).

Uses a real SQLite MemoryStore (tmp_path) so all atomic operations are
exercised end-to-end. No mocks, no stubs.

Coverage:
- compress() happy path: status → COMPRESSED, label + archive created
- compress() non-existent mu_id → success=False
- compress() non-active MU (forgotten/compressed) → success=False
- CompressionResult fields (label_id, archive_id, topic, short_summary)
- compress_mu() with pre-fetched object
- compress_batch() happy path, partial errors, returns all results
- compress_mus() with list of MU objects
- restore() happy path: COMPRESSED → ACTIVE, label + archive removed
- restore() non-compressed MU → raises IllegalStateTransitionError
- restore() non-existent MU → raises MemoryUnitNotFoundError
- restore_forgotten() happy path: FORGOTTEN → ACTIVE
- restore_forgotten() non-forgotten MU → raises
- peek_archive() returns MemoryUnit without changing status
- peek_archive() on non-compressed MU → None
- peek_archive_from_label() via label_id
- verify() valid compressed MU
- verify() non-existent MU
- verify() non-compressed MU
- verify() missing label (pointer set but row deleted)
- verify() missing archive (pointer set but row deleted)
- verify() corrupt full_memory_unit_json
- verify() mu_id mismatch in archive JSON
- stats() returns correct counts per status
- CompressionStats.total property
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locomo_memory.phase2.compression import (
    CompressionResult,
    CompressionService,
    CompressionStats,
    VerificationResult,
)
from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.sqlite_store import (
    IllegalStateTransitionError,
    MemoryStore,
    MemoryUnitNotFoundError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


def _svc(store: MemoryStore) -> CompressionService:
    return CompressionService(store)


def _mu(
    *,
    conversation_id: str = "conv_1",
    session_id: str = "s1",
    claim: str = "Alice works at Google in New York.",
) -> MemoryUnit:
    return MemoryUnit(conversation_id=conversation_id, session_id=session_id, claim=claim)


def _active_mu(store: MemoryStore, **kwargs) -> MemoryUnit:
    mu = _mu(**kwargs)
    store.insert_memory_unit(mu)
    return mu


def _compressed_mu(store: MemoryStore, svc: CompressionService, **kwargs) -> MemoryUnit:
    mu = _active_mu(store, **kwargs)
    result = svc.compress(mu.mu_id)
    assert result.success, f"compression failed: {result.error}"
    return store.get_memory_unit(mu.mu_id)  # type: ignore[return-value]


def _forgotten_mu(store: MemoryStore, **kwargs) -> MemoryUnit:
    mu = _active_mu(store, **kwargs)
    store.forget_atomic(mu.mu_id)
    return store.get_memory_unit(mu.mu_id)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# compress() — happy path
# ---------------------------------------------------------------------------


class TestCompressHappyPath:
    def test_result_success(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.compress(mu.mu_id)
        assert result.success is True
        assert result.error is None

    def test_mu_status_becomes_compressed(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        svc.compress(mu.mu_id)
        updated = store.get_memory_unit(mu.mu_id)
        assert updated is not None
        # New design: compressed MU sits in ARCHIVED status with a CompressedLabel
        # acting as the searchable pointer in the compressed tier.
        assert updated.status == MemoryStatus.ARCHIVED

    def test_label_exists_in_store(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.compress(mu.mu_id)
        label = store.get_compressed_label(result.label_id)  # type: ignore[arg-type]
        assert label is not None
        assert label.mu_id == mu.mu_id

    def test_archive_exists_in_store(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.compress(mu.mu_id)
        archive = store.get_archived_entry(result.archive_id)  # type: ignore[arg-type]
        assert archive is not None
        assert archive.mu_id == mu.mu_id

    def test_result_has_label_and_archive_ids(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.compress(mu.mu_id)
        assert result.label_id is not None and result.label_id.startswith("lbl_")
        assert result.archive_id is not None and result.archive_id.startswith("arc_")

    def test_result_has_topic_and_summary(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store, claim="Alice works at Google in New York.")
        result = svc.compress(mu.mu_id)
        assert result.topic == "employment"
        assert result.short_summary is not None and "Alice" in result.short_summary

    def test_archive_json_contains_original_claim(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store, claim="Bob graduated from MIT last year.")
        result = svc.compress(mu.mu_id)
        archive = store.get_archived_entry(result.archive_id)  # type: ignore[arg-type]
        assert archive is not None
        assert "Bob graduated from MIT last year." in archive.full_memory_unit_json

    def test_pointer_cross_references_valid(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.compress(mu.mu_id)
        label = store.get_compressed_label(result.label_id)  # type: ignore[arg-type]
        archive = store.get_archived_entry(result.archive_id)  # type: ignore[arg-type]
        assert label is not None and archive is not None
        assert label.archived_pointer == archive.archived_entry_id
        assert archive.label_pointer == label.label_id


# ---------------------------------------------------------------------------
# compress() — error cases
# ---------------------------------------------------------------------------


class TestCompressErrors:
    def test_non_existent_mu_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        result = svc.compress("nonexistent_id")
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_already_compressed_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        svc.compress(mu.mu_id)
        result2 = svc.compress(mu.mu_id)
        assert result2.success is False
        assert result2.error is not None

    def test_forgotten_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _forgotten_mu(store)
        result = svc.compress(mu.mu_id)
        assert result.success is False
        assert "active" in result.error.lower()  # type: ignore[operator]

    def test_deleted_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        store.delete_atomic(mu.mu_id)
        result = svc.compress(mu.mu_id)
        assert result.success is False


# ---------------------------------------------------------------------------
# compress_mu() — pre-fetched object
# ---------------------------------------------------------------------------


class TestCompressMu:
    def test_compress_mu_equivalent_to_compress(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.compress_mu(mu)
        assert result.success is True
        assert result.mu_id == mu.mu_id

    def test_compress_mu_non_active_fails(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _forgotten_mu(store)
        result = svc.compress_mu(mu)
        assert result.success is False


# ---------------------------------------------------------------------------
# compress_batch() and compress_mus()
# ---------------------------------------------------------------------------


class TestBatchCompression:
    def test_batch_all_succeed(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mus = [_active_mu(store, claim=f"fact {i}") for i in range(5)]
        results = svc.compress_batch([mu.mu_id for mu in mus])
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_batch_partial_errors(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        good = _active_mu(store)
        ids = [good.mu_id, "bad_id_1", "bad_id_2"]
        results = svc.compress_batch(ids)
        assert len(results) == 3
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures) == 2

    def test_batch_empty_list(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        assert svc.compress_batch([]) == []

    def test_compress_mus_with_objects(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mus = [_active_mu(store, claim=f"claim {i}") for i in range(3)]
        results = svc.compress_mus(mus)
        assert all(r.success for r in results)
        assert [r.mu_id for r in results] == [mu.mu_id for mu in mus]


# ---------------------------------------------------------------------------
# restore()
# ---------------------------------------------------------------------------


class TestRestore:
    def test_restore_compressed_returns_active_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        restored = svc.restore(mu.mu_id)
        assert restored.status == MemoryStatus.ACTIVE
        assert restored.mu_id == mu.mu_id

    def test_restore_removes_label_row(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        label_id = mu.compressed_label_id
        svc.restore(mu.mu_id)
        assert store.get_compressed_label(label_id) is None  # type: ignore[arg-type]

    def test_restore_removes_archive_row(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        archive_id = mu.archived_entry_id
        svc.restore(mu.mu_id)
        assert store.get_archived_entry(archive_id) is None  # type: ignore[arg-type]

    def test_restore_marks_for_reindex(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        restored = svc.restore(mu.mu_id)
        assert restored.needs_reindex is True

    def test_restore_non_compressed_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        with pytest.raises(IllegalStateTransitionError):
            svc.restore(mu.mu_id)

    def test_restore_non_existent_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        with pytest.raises(MemoryUnitNotFoundError):
            svc.restore("ghost_id")


# ---------------------------------------------------------------------------
# restore_forgotten()
# ---------------------------------------------------------------------------


class TestRestoreForgotten:
    def test_restore_forgotten_returns_active_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _forgotten_mu(store)
        restored = svc.restore_forgotten(mu.mu_id)
        assert restored.status == MemoryStatus.ACTIVE

    def test_restore_forgotten_non_forgotten_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        with pytest.raises(IllegalStateTransitionError):
            svc.restore_forgotten(mu.mu_id)

    def test_restore_forgotten_non_existent_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        with pytest.raises(MemoryUnitNotFoundError):
            svc.restore_forgotten("ghost_id")


# ---------------------------------------------------------------------------
# peek_archive()
# ---------------------------------------------------------------------------


class TestPeekArchive:
    def test_returns_memory_unit_object(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc, claim="Carol lives in Paris now.")
        result = svc.peek_archive(mu.mu_id)
        assert isinstance(result, MemoryUnit)

    def test_original_claim_preserved(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc, claim="Carol lives in Paris now.")
        result = svc.peek_archive(mu.mu_id)
        assert result is not None
        assert result.claim == "Carol lives in Paris now."

    def test_does_not_change_status(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        svc.peek_archive(mu.mu_id)
        still_compressed = store.get_memory_unit(mu.mu_id)
        assert still_compressed is not None
        assert still_compressed.status == MemoryStatus.ARCHIVED

    def test_non_compressed_mu_returns_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.peek_archive(mu.mu_id)
        assert result is None

    def test_unknown_mu_id_returns_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        assert svc.peek_archive("ghost") is None

    def test_peek_archive_from_label(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc, claim="Dave plans to move to Tokyo.")
        live_mu = store.get_memory_unit(mu.mu_id)
        assert live_mu is not None
        label_id = live_mu.compressed_label_id
        result = svc.peek_archive_from_label(label_id)  # type: ignore[arg-type]
        assert result is not None
        assert result.claim == "Dave plans to move to Tokyo."

    def test_peek_archive_from_label_unknown(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        assert svc.peek_archive_from_label("ghost_label") is None


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


class TestVerify:
    def test_valid_compressed_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        result = svc.verify(mu.mu_id)
        assert result.valid is True
        assert result.issues == []

    def test_non_existent_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        result = svc.verify("ghost")
        assert result.valid is False
        assert any("not found" in i.lower() for i in result.issues)

    def test_non_compressed_mu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        result = svc.verify(mu.mu_id)
        assert result.valid is False
        assert any("compressed" in i for i in result.issues)

    def test_missing_label_row(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        live_mu = store.get_memory_unit(mu.mu_id)
        assert live_mu is not None
        # Manually delete the label row to simulate corruption.
        with store.transaction() as conn:
            conn.execute(
                "DELETE FROM compressed_labels WHERE label_id = ?",
                (live_mu.compressed_label_id,),
            )
        result = svc.verify(mu.mu_id)
        assert result.valid is False
        assert any("not found" in i.lower() for i in result.issues)

    def test_missing_archive_row(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        live_mu = store.get_memory_unit(mu.mu_id)
        assert live_mu is not None
        # Manually delete the archive row to simulate corruption.
        with store.transaction() as conn:
            conn.execute(
                "DELETE FROM archived_entries WHERE archived_entry_id = ?",
                (live_mu.archived_entry_id,),
            )
        result = svc.verify(mu.mu_id)
        assert result.valid is False
        assert any("not found" in i.lower() for i in result.issues)

    def test_corrupt_archive_json(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        live_mu = store.get_memory_unit(mu.mu_id)
        assert live_mu is not None
        # Overwrite the JSON with garbage.
        with store.transaction() as conn:
            conn.execute(
                "UPDATE archived_entries SET full_memory_unit_json = ? "
                "WHERE archived_entry_id = ?",
                ("not-valid-json{{{", live_mu.archived_entry_id),
            )
        result = svc.verify(mu.mu_id)
        assert result.valid is False
        assert any("parseable" in i.lower() for i in result.issues)

    def test_mu_id_mismatch_in_archive_json(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        live_mu = store.get_memory_unit(mu.mu_id)
        assert live_mu is not None
        # Overwrite the archive JSON with a different mu_id.
        tampered = json.loads(
            store.get_archive_for_mu(mu.mu_id).full_memory_unit_json  # type: ignore[union-attr]
        )
        tampered["mu_id"] = "mu_different_id"
        # Need a non-empty claim for MemoryUnit to be valid.
        with store.transaction() as conn:
            conn.execute(
                "UPDATE archived_entries SET full_memory_unit_json = ? "
                "WHERE archived_entry_id = ?",
                (json.dumps(tampered), live_mu.archived_entry_id),
            )
        result = svc.verify(mu.mu_id)
        assert result.valid is False
        assert any("mu_id" in i.lower() for i in result.issues)


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_conversation(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        stats = svc.stats("conv_1")
        assert stats.active == 0
        assert stats.compressed == 0
        assert stats.forgotten == 0
        assert stats.total == 0

    def test_counts_correctly(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        # 3 active, 2 compressed, 1 forgotten
        active_mus = [_active_mu(store, claim=f"active {i}") for i in range(3)]
        compress_mus = [_active_mu(store, claim=f"compress {i}") for i in range(2)]
        for m in compress_mus:
            svc.compress(m.mu_id)
        forgotten_mu = _active_mu(store, claim="forgotten claim")
        store.forget_atomic(forgotten_mu.mu_id)

        stats = svc.stats("conv_1")
        assert stats.active == 3
        # Compressed MUs now live in ARCHIVED status (the original-data tier);
        # the legacy COMPRESSED bucket is reserved for pre-migration label rows.
        assert stats.archived == 2
        assert stats.compressed == 0
        assert stats.forgotten == 1

    def test_total_property(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        for _ in range(4):
            _active_mu(store)
        stats = svc.stats("conv_1")
        assert stats.total == stats.active + stats.compressed + stats.forgotten + stats.archived

    def test_scoped_to_conversation(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        _active_mu(store, conversation_id="conv_A")
        _active_mu(store, conversation_id="conv_A")
        _active_mu(store, conversation_id="conv_B")
        assert svc.stats("conv_A").active == 2
        assert svc.stats("conv_B").active == 1


# ---------------------------------------------------------------------------
# CompressionResult and CompressionStats schemas
# ---------------------------------------------------------------------------


class TestResultSchemas:
    def test_compression_result_fields_on_success(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _active_mu(store)
        r = svc.compress(mu.mu_id)
        assert r.mu_id == mu.mu_id
        assert r.success is True
        assert r.label_id is not None
        assert r.archive_id is not None
        assert r.topic is not None
        assert r.short_summary is not None
        assert r.error is None

    def test_compression_result_fields_on_failure(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        r = svc.compress("bad_id")
        assert r.success is False
        assert r.label_id is None
        assert r.archive_id is None
        assert r.error is not None

    def test_verification_result_fields(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        svc = _svc(store)
        mu = _compressed_mu(store, svc)
        v = svc.verify(mu.mu_id)
        assert isinstance(v, VerificationResult)
        assert v.mu_id == mu.mu_id
        assert isinstance(v.valid, bool)
        assert isinstance(v.issues, list)
