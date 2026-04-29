"""Tests for the SQLite source-of-truth store.

Covers: idempotent init, CRUD, transactional rollback, atomic compound state
transitions (compress / restore / forget / delete), reindex flag handling,
audit trail, and edge integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    EdgeRecord,
    EdgeType,
    MemoryStatus,
    MemoryUnit,
)
from locomo_memory.phase2.store.sqlite_store import (
    DELETED_PLACEHOLDER,
    IllegalStateTransitionError,
    MemoryStore,
    MemoryStoreError,
    MemoryUnitNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


def _make_mu(
    *,
    conversation_id: str = "conv_1",
    session_id: str = "session_1",
    claim: str = "fact A",
    salience: float = 0.5,
    pinned: bool = False,
) -> MemoryUnit:
    return MemoryUnit(
        conversation_id=conversation_id,
        session_id=session_id,
        claim=claim,
        original_text=f"raw turn for: {claim}",
        source_dia_ids=["D1:1"],
        source_speaker="Speaker",
        salience_score=salience,
        user_pinned=pinned,
    )


def _build_compression_pair(mu: MemoryUnit) -> tuple[CompressedLabel, ArchivedEntry]:
    """Construct a label/archive pair for a given MU with cross-pointers wired."""
    archive = ArchivedEntry(
        label_pointer="placeholder",
        mu_id=mu.mu_id,
        conversation_id=mu.conversation_id,
        full_memory_unit_json=mu.model_dump_json(),
        full_original_text=mu.original_text,
    )
    label = CompressedLabel(
        archived_pointer=archive.archived_entry_id,
        mu_id=mu.mu_id,
        conversation_id=mu.conversation_id,
        topic="topic",
        short_summary="summary",
        key_entities=["entity"],
    )
    archive.label_pointer = label.label_id
    return label, archive


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


class TestSchemaInit:
    def test_idempotent_init(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.db"
        s1 = MemoryStore(path)
        s2 = MemoryStore(path)
        mu = _make_mu()
        s1.insert_memory_unit(mu)
        loaded = s2.get_memory_unit(mu.mu_id)
        assert loaded is not None
        assert loaded.mu_id == mu.mu_id

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "memory.db"
        MemoryStore(nested)
        assert nested.parent.is_dir()

    def test_third_init_no_duplicate_version_row(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.db"
        for _ in range(3):
            MemoryStore(path)
        # only one schema_version row should exist (insert is gated)
        s = MemoryStore(path)
        with s.reader() as conn:
            count = conn.execute("SELECT COUNT(*) AS n FROM schema_version").fetchone()["n"]
        assert count == 1


# ---------------------------------------------------------------------------
# Memory Unit CRUD
# ---------------------------------------------------------------------------


class TestMemoryUnitCRUD:
    def test_insert_and_get(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        loaded = store.get_memory_unit(mu.mu_id)
        assert loaded is not None
        assert loaded.mu_id == mu.mu_id
        assert loaded.claim == mu.claim
        assert loaded.status == MemoryStatus.ACTIVE
        assert loaded.source_dia_ids == ["D1:1"]
        assert loaded.user_pinned is False

    def test_get_nonexistent_returns_none(self, store: MemoryStore) -> None:
        assert store.get_memory_unit("mu_nonexistent") is None

    def test_get_or_raise(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryUnitNotFoundError):
            store.get_memory_unit_or_raise("mu_nonexistent")

    def test_update_memory_unit(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        mu.salience_score = 0.9
        mu.confidence = 0.8
        mu.retrieval_count = 5
        store.update_memory_unit(mu)
        reloaded = store.get_memory_unit(mu.mu_id)
        assert reloaded is not None
        assert reloaded.salience_score == 0.9
        assert reloaded.confidence == 0.8
        assert reloaded.retrieval_count == 5

    def test_update_nonexistent_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        # not inserted
        with pytest.raises(MemoryUnitNotFoundError):
            store.update_memory_unit(mu)

    def test_increment_retrieval_count(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        assert mu.retrieval_count == 0
        store.increment_retrieval_count(mu.mu_id)
        store.increment_retrieval_count(mu.mu_id)
        reloaded = store.get_memory_unit(mu.mu_id)
        assert reloaded is not None
        assert reloaded.retrieval_count == 2
        assert reloaded.last_accessed is not None

    def test_increment_nonexistent_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryUnitNotFoundError):
            store.increment_retrieval_count("mu_ghost")

    def test_set_pinned(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        store.set_pinned(mu.mu_id, True)
        loaded1 = store.get_memory_unit(mu.mu_id)
        assert loaded1 is not None and loaded1.user_pinned is True
        store.set_pinned(mu.mu_id, False)
        loaded2 = store.get_memory_unit(mu.mu_id)
        assert loaded2 is not None and loaded2.user_pinned is False

    def test_set_pinned_nonexistent_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryUnitNotFoundError):
            store.set_pinned("mu_ghost", True)

    def test_list_by_status_filters_correctly(self, store: MemoryStore) -> None:
        for i in range(5):
            store.insert_memory_unit(_make_mu(claim=f"fact {i}"))
        all_active = store.list_active("conv_1")
        assert len(all_active) == 5

        store.update_status(all_active[0].mu_id, MemoryStatus.FORGOTTEN)
        actives = store.list_active("conv_1")
        forgotten = store.list_by_status("conv_1", MemoryStatus.FORGOTTEN)
        assert len(actives) == 4
        assert len(forgotten) == 1

    def test_list_active_per_conversation(self, store: MemoryStore) -> None:
        store.insert_memory_unit(_make_mu(claim="a", conversation_id="c1"))
        store.insert_memory_unit(_make_mu(claim="b", conversation_id="c2"))
        assert len(store.list_active("c1")) == 1
        assert len(store.list_active("c2")) == 1

    def test_list_all_includes_all_statuses(self, store: MemoryStore) -> None:
        a = _make_mu(claim="a")
        b = _make_mu(claim="b")
        store.insert_memory_unit(a)
        store.insert_memory_unit(b)
        store.forget_atomic(a.mu_id)
        all_mus = store.list_all("conv_1")
        assert len(all_mus) == 2

    def test_count_by_status(self, store: MemoryStore) -> None:
        for i in range(3):
            store.insert_memory_unit(_make_mu(claim=f"fact {i}"))
        counts = store.count_by_status("conv_1")
        assert counts[MemoryStatus.ACTIVE] == 3
        assert counts[MemoryStatus.COMPRESSED] == 0
        assert counts[MemoryStatus.FORGOTTEN] == 0
        assert counts[MemoryStatus.DELETED] == 0

    def test_storage_pressure(self, store: MemoryStore) -> None:
        for i in range(7):
            store.insert_memory_unit(_make_mu(claim=f"fact {i}"))
        assert store.storage_pressure("conv_1", cap=10) == 0.7
        assert store.storage_pressure("conv_1", cap=0) == 0.0
        assert store.storage_pressure("conv_1", cap=-5) == 0.0

    def test_iter_active_yields_only_active(self, store: MemoryStore) -> None:
        for i in range(3):
            store.insert_memory_unit(_make_mu(claim=f"fact {i}"))
        # Forget one
        all_mus = store.list_active("conv_1")
        store.forget_atomic(all_mus[0].mu_id)
        items = list(store.iter_active("conv_1"))
        assert len(items) == 2
        assert all(m.status == MemoryStatus.ACTIVE for m in items)


# ---------------------------------------------------------------------------
# Reindex flag
# ---------------------------------------------------------------------------


class TestReindexFlag:
    def test_mark_and_clear(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        assert mu.needs_reindex is False
        store.mark_needs_reindex(mu.mu_id)
        needing = store.list_needing_reindex()
        assert len(needing) == 1
        assert needing[0].mu_id == mu.mu_id
        store.clear_reindex_flag(mu.mu_id)
        assert store.list_needing_reindex() == []

    def test_mark_nonexistent_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryUnitNotFoundError):
            store.mark_needs_reindex("mu_ghost")

    def test_clear_nonexistent_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryUnitNotFoundError):
            store.clear_reindex_flag("mu_ghost")


# ---------------------------------------------------------------------------
# Compression / Restoration / Forget
# ---------------------------------------------------------------------------


class TestCompression:
    def test_compress_atomic_happy_path(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)

        store.compress_atomic(mu.mu_id, label, archive)

        loaded_mu = store.get_memory_unit(mu.mu_id)
        assert loaded_mu is not None
        assert loaded_mu.status == MemoryStatus.COMPRESSED
        assert loaded_mu.compressed_label_id == label.label_id
        assert loaded_mu.archived_entry_id == archive.archived_entry_id

        loaded_label = store.get_compressed_label(label.label_id)
        assert loaded_label is not None
        assert loaded_label.short_summary == "summary"

        loaded_archive = store.get_archived_entry(archive.archived_entry_id)
        assert loaded_archive is not None
        assert loaded_archive.mu_id == mu.mu_id

    def test_compress_mismatched_mu_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        archive.mu_id = "mu_other"
        with pytest.raises(MemoryStoreError):
            store.compress_atomic(mu.mu_id, label, archive)

    def test_compress_label_pointer_mismatch_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        label.archived_pointer = "arc_wrong"
        with pytest.raises(MemoryStoreError):
            store.compress_atomic(mu.mu_id, label, archive)

    def test_compress_archive_pointer_mismatch_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        archive.label_pointer = "lbl_wrong"
        with pytest.raises(MemoryStoreError):
            store.compress_atomic(mu.mu_id, label, archive)

    def test_compress_nonexistent_mu_raises(self, store: MemoryStore) -> None:
        ghost = _make_mu(claim="ghost")
        label, archive = _build_compression_pair(ghost)
        with pytest.raises(MemoryUnitNotFoundError):
            store.compress_atomic(ghost.mu_id, label, archive)

    def test_compress_already_compressed_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        l1, a1 = _build_compression_pair(mu)
        store.compress_atomic(mu.mu_id, l1, a1)

        l2, a2 = _build_compression_pair(mu)
        with pytest.raises(IllegalStateTransitionError):
            store.compress_atomic(mu.mu_id, l2, a2)

    def test_compress_rolls_back_on_pointer_mismatch(self, store: MemoryStore) -> None:
        # Even though pointer mismatch is checked before the transaction,
        # ensure no partial state was created.
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        label.archived_pointer = "arc_wrong"
        with pytest.raises(MemoryStoreError):
            store.compress_atomic(mu.mu_id, label, archive)
        # MU still active, no label or archive present
        loaded = store.get_memory_unit(mu.mu_id)
        assert loaded is not None and loaded.status == MemoryStatus.ACTIVE
        assert store.get_compressed_label(label.label_id) is None
        assert store.get_archived_entry(archive.archived_entry_id) is None


class TestRestoration:
    def test_restore_atomic_clears_pointers_and_marks_reindex(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        store.compress_atomic(mu.mu_id, label, archive)

        restored = store.restore_atomic(mu.mu_id)
        assert restored.status == MemoryStatus.ACTIVE
        assert restored.compressed_label_id is None
        assert restored.archived_entry_id is None
        assert restored.needs_reindex is True

        # Label and archive removed
        assert store.get_compressed_label(label.label_id) is None
        assert store.get_archived_entry(archive.archived_entry_id) is None

    def test_restore_nonexistent_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryUnitNotFoundError):
            store.restore_atomic("mu_ghost")

    def test_restore_active_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        with pytest.raises(IllegalStateTransitionError):
            store.restore_atomic(mu.mu_id)

    def test_restore_from_forgotten(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        store.forget_atomic(mu.mu_id)
        restored = store.restore_from_forgotten(mu.mu_id)
        assert restored.status == MemoryStatus.ACTIVE
        assert restored.needs_reindex is True

    def test_restore_from_forgotten_when_active_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        with pytest.raises(IllegalStateTransitionError):
            store.restore_from_forgotten(mu.mu_id)


class TestForget:
    def test_forget_active_mu(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        store.forget_atomic(mu.mu_id)
        loaded = store.get_memory_unit(mu.mu_id)
        assert loaded is not None
        assert loaded.status == MemoryStatus.FORGOTTEN
        assert loaded.needs_reindex is True
        # Content preserved (forgotten != deleted)
        assert loaded.claim == "fact A"
        assert loaded.original_text == "raw turn for: fact A"

    def test_forget_idempotent(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        store.forget_atomic(mu.mu_id)
        store.forget_atomic(mu.mu_id)  # should not raise
        loaded = store.get_memory_unit(mu.mu_id)
        assert loaded is not None and loaded.status == MemoryStatus.FORGOTTEN

    def test_forget_compressed_removes_label_and_archive(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        store.compress_atomic(mu.mu_id, label, archive)

        store.forget_atomic(mu.mu_id)

        loaded = store.get_memory_unit(mu.mu_id)
        assert loaded is not None and loaded.status == MemoryStatus.FORGOTTEN
        assert loaded.compressed_label_id is None
        assert loaded.archived_entry_id is None
        assert store.get_compressed_label(label.label_id) is None
        assert store.get_archived_entry(archive.archived_entry_id) is None

    def test_forget_deleted_raises(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        store.delete_atomic(mu.mu_id)
        with pytest.raises(IllegalStateTransitionError):
            store.forget_atomic(mu.mu_id)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


class TestDeletion:
    def test_delete_creates_audit_and_nulls_content(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        store.delete_atomic(mu.mu_id, deleted_by="test_user")

        loaded = store.get_memory_unit(mu.mu_id)
        assert loaded is not None
        assert loaded.status == MemoryStatus.DELETED
        assert loaded.claim == DELETED_PLACEHOLDER
        assert loaded.original_text == DELETED_PLACEHOLDER
        # Provenance preserved (mu_id, conversation_id, dia_ids)
        assert loaded.source_dia_ids == ["D1:1"]

        audit = store.list_deletion_audit("conv_1")
        assert len(audit) == 1
        assert audit[0].mu_id == mu.mu_id
        assert audit[0].deleted_by == "test_user"

    def test_delete_idempotent(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        store.delete_atomic(mu.mu_id)
        store.delete_atomic(mu.mu_id)  # second call no-op
        audit = store.list_deletion_audit()
        assert len(audit) == 1

    def test_delete_nonexistent_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryUnitNotFoundError):
            store.delete_atomic("mu_ghost")

    def test_delete_compressed_removes_label_archive_and_edges(
        self, store: MemoryStore
    ) -> None:
        mu = _make_mu()
        other = _make_mu(claim="other")
        store.insert_memory_unit(mu)
        store.insert_memory_unit(other)

        label, archive = _build_compression_pair(mu)
        store.compress_atomic(mu.mu_id, label, archive)

        edge = EdgeRecord(
            source_mu_id=other.mu_id,
            target_mu_id=mu.mu_id,
            edge_type=EdgeType.RELATED_TO,
        )
        store.insert_edge(edge)

        store.delete_atomic(mu.mu_id)

        loaded = store.get_memory_unit(mu.mu_id)
        assert loaded is not None and loaded.status == MemoryStatus.DELETED
        assert store.get_compressed_label(label.label_id) is None
        assert store.get_archived_entry(archive.archived_entry_id) is None
        # Edge cleared on both sides
        assert store.edges_to(mu.mu_id) == []
        assert store.edges_from(mu.mu_id) == []


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class TestEdges:
    def _two_mus(self, store: MemoryStore) -> tuple[MemoryUnit, MemoryUnit]:
        a = _make_mu(claim="a")
        b = _make_mu(claim="b")
        store.insert_memory_unit(a)
        store.insert_memory_unit(b)
        return a, b

    def test_insert_and_get(self, store: MemoryStore) -> None:
        a, b = self._two_mus(store)
        edge = EdgeRecord(
            source_mu_id=a.mu_id,
            target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        )
        store.insert_edge(edge)
        loaded = store.get_edge(edge.edge_id)
        assert loaded is not None
        assert loaded.source_mu_id == a.mu_id
        assert loaded.edge_type == EdgeType.RELATED_TO

    def test_duplicate_edge_raises(self, store: MemoryStore) -> None:
        a, b = self._two_mus(store)
        edge1 = EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        )
        edge2 = EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        )
        store.insert_edge(edge1)
        with pytest.raises(MemoryStoreError):
            store.insert_edge(edge2)

    def test_different_edge_types_allowed(self, store: MemoryStore) -> None:
        a, b = self._two_mus(store)
        store.insert_edge(EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        ))
        store.insert_edge(EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.SUPERSEDED_BY,
        ))
        edges = store.edges_from(a.mu_id)
        assert len(edges) == 2
        types = {e.edge_type for e in edges}
        assert types == {EdgeType.RELATED_TO, EdgeType.SUPERSEDED_BY}

    def test_edges_from_filter_by_type(self, store: MemoryStore) -> None:
        a = _make_mu(claim="a")
        b = _make_mu(claim="b")
        c = _make_mu(claim="c")
        for m in (a, b, c):
            store.insert_memory_unit(m)
        store.insert_edge(EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        ))
        store.insert_edge(EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=c.mu_id,
            edge_type=EdgeType.SUPERSEDED_BY,
        ))
        related = store.edges_from(a.mu_id, edge_type=EdgeType.RELATED_TO)
        assert len(related) == 1
        assert related[0].target_mu_id == b.mu_id

    def test_edges_to_filter_by_type(self, store: MemoryStore) -> None:
        a, b = self._two_mus(store)
        store.insert_edge(EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        ))
        incoming = store.edges_to(b.mu_id, edge_type=EdgeType.RELATED_TO)
        assert len(incoming) == 1

    def test_remove_edge(self, store: MemoryStore) -> None:
        a, b = self._two_mus(store)
        edge = EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        )
        store.insert_edge(edge)
        store.remove_edge(edge.edge_id)
        assert store.get_edge(edge.edge_id) is None
        assert store.edges_from(a.mu_id) == []

    def test_remove_nonexistent_edge_silent(self, store: MemoryStore) -> None:
        store.remove_edge("edg_ghost")  # should not raise

    def test_iter_edges(self, store: MemoryStore) -> None:
        a, b = self._two_mus(store)
        store.insert_edge(EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        ))
        edges = list(store.iter_edges())
        assert len(edges) == 1


# ---------------------------------------------------------------------------
# Transactional rollback
# ---------------------------------------------------------------------------


class TestTransactionRollback:
    def test_rollback_on_exception(self, store: MemoryStore) -> None:
        mu = _make_mu()
        try:
            with store.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO memory_units (
                        mu_id, conversation_id, session_id, claim, original_text,
                        source_dia_ids, source_speaker, timestamp, extracted_at,
                        salience_score, importance, recency_weight, uniqueness,
                        retrieval_count, prompt_frequency, last_accessed,
                        status, confidence, needs_reindex,
                        compressed_label_id, archived_entry_id,
                        user_pinned, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        mu.mu_id, mu.conversation_id, mu.session_id, mu.claim, mu.original_text,
                        json.dumps(mu.source_dia_ids), mu.source_speaker, mu.timestamp,
                        mu.extracted_at.isoformat(),
                        0.5, 0.5, 1.0, 1.0,
                        0, 0.0, None,
                        "active", 0.9, 0,
                        None, None,
                        0, mu.created_at.isoformat(), mu.updated_at.isoformat(),
                    ),
                )
                raise RuntimeError("forced failure mid-transaction")
        except RuntimeError:
            pass
        assert store.get_memory_unit(mu.mu_id) is None


# ---------------------------------------------------------------------------
# Compressed labels and archives - lookups
# ---------------------------------------------------------------------------


class TestLabelArchiveLookup:
    def test_get_label_for_mu(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        store.compress_atomic(mu.mu_id, label, archive)
        found = store.get_label_for_mu(mu.mu_id)
        assert found is not None
        assert found.label_id == label.label_id

    def test_get_archive_for_mu(self, store: MemoryStore) -> None:
        mu = _make_mu()
        store.insert_memory_unit(mu)
        label, archive = _build_compression_pair(mu)
        store.compress_atomic(mu.mu_id, label, archive)
        found = store.get_archive_for_mu(mu.mu_id)
        assert found is not None
        assert found.archived_entry_id == archive.archived_entry_id

    def test_iter_labels_filtered_by_conversation(self, store: MemoryStore) -> None:
        # Two MUs in different conversations, both compressed
        mu1 = _make_mu(conversation_id="c1", claim="a")
        mu2 = _make_mu(conversation_id="c2", claim="b")
        store.insert_memory_unit(mu1)
        store.insert_memory_unit(mu2)
        l1, a1 = _build_compression_pair(mu1)
        l2, a2 = _build_compression_pair(mu2)
        store.compress_atomic(mu1.mu_id, l1, a1)
        store.compress_atomic(mu2.mu_id, l2, a2)

        labels_c1 = list(store.iter_labels("c1"))
        labels_c2 = list(store.iter_labels("c2"))
        labels_all = list(store.iter_labels())
        assert len(labels_c1) == 1
        assert len(labels_c2) == 1
        assert len(labels_all) == 2
