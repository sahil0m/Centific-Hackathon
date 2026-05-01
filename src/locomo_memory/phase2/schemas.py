"""Pydantic schemas for Phase 2 memory units, labels, archives, and edges.

These are the canonical typed data models. SQLite columns mirror these fields,
and FAISS / NetworkX indexes are derived from them.

All datetimes are timezone-aware UTC. All IDs are short prefixed UUIDs for
log-readability (e.g. ``mu_abc123def456``).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryStatus(str, Enum):
    """Lifecycle states of a Memory Unit.

    See PHASE2_METHODOLOGY.md §3 for the state diagram.
    """

    ACTIVE = "active"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class EdgeType(str, Enum):
    """Typed relationships between Memory Units."""

    SUPERSEDED_BY = "superseded_by"
    CONFLICTS_WITH = "conflicts_with"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------


def _short_uuid() -> str:
    return uuid.uuid4().hex[:12]


def new_mu_id() -> str:
    return f"mu_{_short_uuid()}"


def new_label_id() -> str:
    return f"lbl_{_short_uuid()}"


def new_archive_id() -> str:
    return f"arc_{_short_uuid()}"


def new_edge_id() -> str:
    return f"edg_{_short_uuid()}"


def utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Memory Unit
# ---------------------------------------------------------------------------


class MemoryUnit(BaseModel):
    """An atomic claim extracted from a conversation turn.

    Lives in the Active tier by default. Status governs visibility in retrieval;
    salience drives lifecycle decisions during state transitions.
    """

    model_config = ConfigDict(validate_assignment=True)

    mu_id: str = Field(default_factory=new_mu_id)
    conversation_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    # Content
    claim: str = Field(min_length=1)
    original_text: str = ""

    # Provenance
    source_dia_ids: list[str] = Field(default_factory=list)
    source_speaker: str = ""
    timestamp: str | None = None
    extracted_at: datetime = Field(default_factory=utcnow)

    # Salience tracking
    salience_score: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    uniqueness: float = Field(default=1.0, ge=0.0, le=1.0)
    retrieval_count: int = Field(default=0, ge=0)
    prompt_frequency: float = Field(default=0.0, ge=0.0, le=1.0)
    last_accessed: datetime | None = None

    # State
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    needs_reindex: bool = False

    # Compression linkage (set when compressed)
    compressed_label_id: str | None = None
    archived_entry_id: str | None = None

    # User control
    user_pinned: bool = False

    # Audit timestamps
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("source_dia_ids")
    @classmethod
    def _strip_dia_ids(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if s and s.strip()]


# ---------------------------------------------------------------------------
# Compressed Label
# ---------------------------------------------------------------------------


class CompressedLabel(BaseModel):
    """A short summary label that lives in the compressed tier.

    Acts as a smart pointer: searchable in the compressed FAISS index, points
    to the archived full data. When matched in a query, full data is restored
    from the archive and promoted back to active.
    """

    model_config = ConfigDict(validate_assignment=True)

    label_id: str = Field(default_factory=new_label_id)
    archived_pointer: str = Field(min_length=1)
    mu_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)

    topic: str = Field(min_length=1)
    short_summary: str = Field(min_length=1)
    key_entities: list[str] = Field(default_factory=list)
    time_range: str | None = None

    original_dia_ids: list[str] = Field(default_factory=list)

    compressed_at: datetime = Field(default_factory=utcnow)
    retrieval_count: int = Field(default=0, ge=0)
    last_label_match: datetime | None = None


# ---------------------------------------------------------------------------
# Archived Entry
# ---------------------------------------------------------------------------


class ArchivedEntry(BaseModel):
    """The exact recovery layer for compressed Memory Units.

    Stores the full original Memory Unit state at compression time. Only
    accessed via the Compressed label's ``archived_pointer``. Restoration
    recreates the active MU from this snapshot.
    """

    model_config = ConfigDict(validate_assignment=True)

    archived_entry_id: str = Field(default_factory=new_archive_id)
    label_pointer: str = Field(min_length=1)
    mu_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)

    full_memory_unit_json: str = Field(min_length=1)
    full_original_text: str = ""

    archived_at: datetime = Field(default_factory=utcnow)
    restoration_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Edge Record
# ---------------------------------------------------------------------------


class EdgeRecord(BaseModel):
    """A typed edge between two Memory Units in the relationship graph."""

    model_config = ConfigDict(validate_assignment=True)

    edge_id: str = Field(default_factory=new_edge_id)
    source_mu_id: str = Field(min_length=1)
    target_mu_id: str = Field(min_length=1)
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0)
    created_at: datetime = Field(default_factory=utcnow)
    metadata_json: str | None = None

    @model_validator(mode="after")
    def _no_self_loops(self) -> EdgeRecord:
        if self.source_mu_id == self.target_mu_id:
            raise ValueError(
                f"self-loop edge not allowed: source==target=={self.source_mu_id}"
            )
        return self


# ---------------------------------------------------------------------------
# Deletion Audit
# ---------------------------------------------------------------------------


class DeletionAudit(BaseModel):
    """Tombstone record for a permanently deleted Memory Unit.

    Created when a user explicitly deletes a MU. The MU row itself is hard-
    deleted from ``memory_units`` (along with any compressed label, archive
    snapshot, and edges). This audit row is the only surviving trace of the
    deletion event and is used for compliance/audit reporting.
    """

    model_config = ConfigDict(validate_assignment=True)

    audit_id: int | None = None  # auto-assigned by SQLite
    mu_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    source_dia_ids: list[str] = Field(default_factory=list)
    deleted_at: datetime = Field(default_factory=utcnow)
    deleted_by: str = "user"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ArchivedEntry",
    "CompressedLabel",
    "DeletionAudit",
    "EdgeRecord",
    "EdgeType",
    "MemoryStatus",
    "MemoryUnit",
    "new_archive_id",
    "new_edge_id",
    "new_label_id",
    "new_mu_id",
    "utcnow",
]
