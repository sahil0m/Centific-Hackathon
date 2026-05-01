"""Tests for Phase 2 Pydantic schemas.

These cover validation, defaults, ID generation, serialization roundtrips,
and edge cases like self-loops and out-of-range scores.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    DeletionAudit,
    EdgeRecord,
    EdgeType,
    MemoryStatus,
    MemoryUnit,
    new_archive_id,
    new_edge_id,
    new_label_id,
    new_mu_id,
    utcnow,
)


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------


class TestIdGenerators:
    def test_mu_id_prefix_and_uniqueness(self) -> None:
        ids = {new_mu_id() for _ in range(200)}
        assert len(ids) == 200
        for i in ids:
            assert i.startswith("mu_")
            assert len(i) > len("mu_")

    def test_label_id_prefix(self) -> None:
        assert new_label_id().startswith("lbl_")

    def test_archive_id_prefix(self) -> None:
        assert new_archive_id().startswith("arc_")

    def test_edge_id_prefix(self) -> None:
        assert new_edge_id().startswith("edg_")

    def test_utcnow_is_timezone_aware(self) -> None:
        dt = utcnow()
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt) == timezone.utc.utcoffset(dt)


# ---------------------------------------------------------------------------
# MemoryUnit
# ---------------------------------------------------------------------------


def _valid_mu_kwargs() -> dict:
    return {
        "conversation_id": "conv_1",
        "session_id": "session_2",
        "claim": "Caroline researches adoption agencies",
        "original_text": "I've been looking into adoption agencies",
        "source_dia_ids": ["D2:5"],
        "source_speaker": "Caroline",
    }


class TestMemoryUnitValidation:
    def test_minimal_valid_construction(self) -> None:
        mu = MemoryUnit(**_valid_mu_kwargs())
        assert mu.mu_id.startswith("mu_")
        assert mu.status == MemoryStatus.ACTIVE
        assert mu.salience_score == 0.5
        assert mu.confidence == 0.9
        assert mu.user_pinned is False
        assert mu.needs_reindex is False
        assert mu.retrieval_count == 0
        assert mu.last_accessed is None
        assert mu.compressed_label_id is None
        assert mu.archived_entry_id is None

    def test_empty_claim_rejected(self) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["claim"] = ""
        with pytest.raises(ValidationError):
            MemoryUnit(**kwargs)

    def test_empty_conversation_id_rejected(self) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["conversation_id"] = ""
        with pytest.raises(ValidationError):
            MemoryUnit(**kwargs)

    def test_empty_session_id_rejected(self) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["session_id"] = ""
        with pytest.raises(ValidationError):
            MemoryUnit(**kwargs)

    @pytest.mark.parametrize("bad_value", [-0.01, 1.01, 2.0, -5.0])
    def test_salience_out_of_range_rejected(self, bad_value: float) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["salience_score"] = bad_value
        with pytest.raises(ValidationError):
            MemoryUnit(**kwargs)

    @pytest.mark.parametrize(
        "field", ["importance", "recency_weight", "uniqueness", "confidence", "prompt_frequency"]
    )
    def test_other_score_fields_out_of_range(self, field: str) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs[field] = 1.5
        with pytest.raises(ValidationError):
            MemoryUnit(**kwargs)

    def test_negative_retrieval_count_rejected(self) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["retrieval_count"] = -1
        with pytest.raises(ValidationError):
            MemoryUnit(**kwargs)

    def test_dia_ids_stripped_and_filtered(self) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["source_dia_ids"] = [" D2:5 ", "", "  ", "D3:1"]
        mu = MemoryUnit(**kwargs)
        assert mu.source_dia_ids == ["D2:5", "D3:1"]

    def test_status_assignment_validates(self) -> None:
        mu = MemoryUnit(**_valid_mu_kwargs())
        mu.status = MemoryStatus.COMPRESSED
        assert mu.status == MemoryStatus.COMPRESSED

    def test_validate_assignment_blocks_bad_score(self) -> None:
        mu = MemoryUnit(**_valid_mu_kwargs())
        with pytest.raises(ValidationError):
            mu.salience_score = 2.0

    def test_serialization_roundtrip(self) -> None:
        mu = MemoryUnit(**_valid_mu_kwargs())
        data = mu.model_dump()
        mu2 = MemoryUnit(**data)
        assert mu2.mu_id == mu.mu_id
        assert mu2.status == mu.status
        assert mu2.source_dia_ids == mu.source_dia_ids

    def test_json_roundtrip(self) -> None:
        mu = MemoryUnit(**_valid_mu_kwargs())
        json_str = mu.model_dump_json()
        data = json.loads(json_str)
        mu2 = MemoryUnit(**data)
        assert mu2.mu_id == mu.mu_id

    def test_extracted_at_default_is_utc(self) -> None:
        mu = MemoryUnit(**_valid_mu_kwargs())
        assert mu.extracted_at.tzinfo is not None

    def test_status_string_coerced(self) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["status"] = "compressed"
        mu = MemoryUnit(**kwargs)
        assert mu.status == MemoryStatus.COMPRESSED

    def test_invalid_status_string_rejected(self) -> None:
        kwargs = _valid_mu_kwargs()
        kwargs["status"] = "not_a_real_status"
        with pytest.raises(ValidationError):
            MemoryUnit(**kwargs)


# ---------------------------------------------------------------------------
# CompressedLabel
# ---------------------------------------------------------------------------


class TestCompressedLabelValidation:
    def test_minimal_valid(self) -> None:
        label = CompressedLabel(
            archived_pointer="arc_xxx",
            mu_id="mu_xxx",
            conversation_id="conv_1",
            topic="Career",
            short_summary="Caroline: Google -> Microsoft",
            key_entities=["Caroline", "Google", "Microsoft"],
        )
        assert label.label_id.startswith("lbl_")
        assert label.retrieval_count == 0
        assert label.last_label_match is None

    def test_empty_topic_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompressedLabel(
                archived_pointer="arc_xxx",
                mu_id="mu_xxx",
                conversation_id="conv_1",
                topic="",
                short_summary="x",
            )

    def test_empty_summary_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompressedLabel(
                archived_pointer="arc_xxx",
                mu_id="mu_xxx",
                conversation_id="conv_1",
                topic="t",
                short_summary="",
            )

    def test_empty_archived_pointer_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompressedLabel(
                archived_pointer="",
                mu_id="mu_xxx",
                conversation_id="conv_1",
                topic="t",
                short_summary="s",
            )


# ---------------------------------------------------------------------------
# ArchivedEntry
# ---------------------------------------------------------------------------


class TestArchivedEntryValidation:
    def test_minimal_valid(self) -> None:
        a = ArchivedEntry(
            label_pointer="lbl_xxx",
            mu_id="mu_xxx",
            conversation_id="conv_1",
            full_memory_unit_json='{"mu_id":"mu_xxx"}',
            full_original_text="text",
        )
        assert a.archived_entry_id.startswith("arc_")
        assert a.restoration_count == 0

    def test_empty_json_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArchivedEntry(
                label_pointer="lbl_xxx",
                mu_id="mu_xxx",
                conversation_id="conv_1",
                full_memory_unit_json="",
                full_original_text="text",
            )


# ---------------------------------------------------------------------------
# EdgeRecord
# ---------------------------------------------------------------------------


class TestEdgeRecordValidation:
    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            EdgeRecord(
                source_mu_id="mu_a",
                target_mu_id="mu_a",
                edge_type=EdgeType.RELATED_TO,
            )
        assert "self-loop" in str(excinfo.value).lower()

    def test_valid_edge(self) -> None:
        e = EdgeRecord(
            source_mu_id="mu_a",
            target_mu_id="mu_b",
            edge_type=EdgeType.SUPERSEDED_BY,
        )
        assert e.edge_id.startswith("edg_")
        assert e.weight == 1.0
        assert e.created_at.tzinfo is not None

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EdgeRecord(
                source_mu_id="mu_a",
                target_mu_id="mu_b",
                edge_type=EdgeType.RELATED_TO,
                weight=-0.5,
            )

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EdgeRecord(
                source_mu_id="",
                target_mu_id="mu_b",
                edge_type=EdgeType.RELATED_TO,
            )

    def test_invalid_edge_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EdgeRecord(
                source_mu_id="mu_a",
                target_mu_id="mu_b",
                edge_type="not_a_type",  # type: ignore[arg-type]
            )

    def test_all_edge_types_valid(self) -> None:
        for t in EdgeType:
            e = EdgeRecord(source_mu_id="mu_a", target_mu_id="mu_b", edge_type=t)
            assert e.edge_type == t


# ---------------------------------------------------------------------------
# DeletionAudit
# ---------------------------------------------------------------------------


class TestDeletionAudit:
    def test_minimal_valid(self) -> None:
        a = DeletionAudit(mu_id="mu_x", conversation_id="conv_1")
        assert a.audit_id is None  # set by SQLite
        assert a.deleted_by == "user"
        assert a.deleted_at.tzinfo is not None

    def test_audit_id_can_be_set(self) -> None:
        a = DeletionAudit(audit_id=42, mu_id="mu_x", conversation_id="conv_1")
        assert a.audit_id == 42


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class TestMemoryStatus:
    def test_all_statuses_are_strings(self) -> None:
        for s in MemoryStatus:
            assert isinstance(s.value, str)

    def test_status_values_distinct(self) -> None:
        values = {s.value for s in MemoryStatus}
        assert len(values) == len(list(MemoryStatus))

    def test_expected_states_present(self) -> None:
        # User-initiated deletion is hard-delete (row removed + audit row);
        # the 4 remaining lifecycle states cover every visible MU.
        expected = {"active", "compressed", "archived", "forgotten"}
        actual = {s.value for s in MemoryStatus}
        assert expected == actual
