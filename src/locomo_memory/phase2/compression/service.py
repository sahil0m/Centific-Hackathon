"""Compression, Archive, and Restore Service — Phase 2 Milestone 6.

Abstracts the mechanics of compressing, archiving, and restoring individual
MemoryUnit objects. The Lifecycle Engine decides *which* MUs to act on based
on salience and capacity; this service decides *how*.

Responsibilities
----------------
- **Compress**   : build a CompressedLabel + ArchivedEntry for an Active MU
  and atomically persist both via ``store.compress_atomic``.
- **Restore**    : promote a Compressed MU back to Active via
  ``store.restore_atomic`` (with full archive round-trip verification).
- **Restore forgotten** : promote a Forgotten MU back to Active via
  ``store.restore_from_forgotten``.
- **Peek archive**: deserialize the full_memory_unit_json snapshot stored in
  the archive back into a ``MemoryUnit`` object *without* changing status —
  useful for UI previews, debugging, and search result expansion.
- **Verify**     : check that a Compressed MU's label, archive, and pointer
  cross-references are internally consistent.
- **Batch compress**: compress a list of MUs, collecting per-item errors
  without aborting the whole batch.
- **Stats**      : count MUs per lifecycle status for a conversation.

Relationship to other modules
------------------------------
- The Lifecycle Engine imports this service (via ``LabelBuilder``) to execute
  the compression step it decides upon; it does not duplicate the logic here.
- The MemoryStore owns the atomic SQLite operations; this service is a thin
  orchestration layer above it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger
from pydantic import ValidationError

from locomo_memory.phase2.lifecycle.engine import LabelBuilder
from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    MemoryStatus,
    MemoryUnit,
)
from locomo_memory.phase2.store.sqlite_store import (
    IllegalStateTransitionError,
    MemoryStore,
    MemoryStoreError,
    MemoryUnitNotFoundError,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompressionResult:
    """Outcome of a single compression attempt."""

    mu_id: str
    success: bool
    label_id: str | None = None
    archive_id: str | None = None
    topic: str | None = None
    short_summary: str | None = None
    error: str | None = None


@dataclass(slots=True)
class VerificationResult:
    """Integrity check result for a compressed MU."""

    mu_id: str
    valid: bool
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompressionStats:
    """Per-status counts for a conversation's memory store.

    Hard-deleted MUs do not appear here — their rows are removed from the
    ``memory_units`` table; only the audit row in ``deletion_audit``
    survives. See ``MemoryStore.list_deletion_audit`` for that view.
    """

    conversation_id: str
    active: int = 0
    compressed: int = 0
    forgotten: int = 0
    archived: int = 0

    @property
    def total(self) -> int:
        return self.active + self.compressed + self.forgotten + self.archived


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CompressionService:
    """Compress, archive, and restore MemoryUnit objects.

    Args:
        store: the SQLite-backed source of truth.
        label_builder: rule-based label factory. Pass ``None`` to use the
            default :class:`~locomo_memory.phase2.lifecycle.engine.LabelBuilder`.

    All public methods are safe to call concurrently when the underlying
    ``MemoryStore`` is thread-safe (each call opens its own connection).
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        label_builder: LabelBuilder | None = None,
    ) -> None:
        self.store = store
        self.label_builder = label_builder or LabelBuilder()

    # ------------------------------------------------------------------
    # Single-MU compression
    # ------------------------------------------------------------------

    def compress(self, mu_id: str, *, strict: bool = False) -> CompressionResult:
        """Compress a single Active MU identified by ``mu_id``.

        Fetches the MU from the store, builds a rule-based label + archive
        snapshot, and commits the atomic compress transition.

        Modes
        -----
        strict=False (default — *safe-result mode*):
            Returns a :class:`CompressionResult` regardless of outcome.
            Check ``result.success`` and ``result.error``. Never raises.
            Use this in pipeline code that must keep running after failures.

        strict=True (*strict mode*):
            Raises :class:`~locomo_memory.phase2.store.sqlite_store.MemoryStoreError`
            or :class:`ValueError` on any failure. Use this in code that
            must fail loudly when compression state is unexpectedly wrong.
        """
        mu = self.store.get_memory_unit(mu_id)
        if mu is None:
            if strict:
                raise MemoryUnitNotFoundError(mu_id)
            return CompressionResult(
                mu_id=mu_id, success=False,
                error=f"MU not found: {mu_id}",
            )
        return self._compress_mu(mu, strict=strict)

    def compress_mu(self, mu: MemoryUnit, *, strict: bool = False) -> CompressionResult:
        """Compress a pre-fetched Active MU.

        Prefer this over :meth:`compress` when the caller already holds the
        MU object (avoids a redundant DB read). Accepts the same ``strict``
        flag as :meth:`compress`.
        """
        return self._compress_mu(mu, strict=strict)

    def _compress_mu(self, mu: MemoryUnit, *, strict: bool = False) -> CompressionResult:
        if mu.status != MemoryStatus.ACTIVE:
            msg = f"cannot compress MU in status '{mu.status.value}'; must be active"
            if strict:
                raise IllegalStateTransitionError(msg)
            return CompressionResult(mu_id=mu.mu_id, success=False, error=msg)
        try:
            label, archive = self.label_builder.build(mu)
            self.store.compress_atomic(mu.mu_id, label, archive)
            logger.debug(
                "CompressionService: compressed mu={} label={} archive={}",
                mu.mu_id, label.label_id, archive.archived_entry_id,
            )
            return CompressionResult(
                mu_id=mu.mu_id,
                success=True,
                label_id=label.label_id,
                archive_id=archive.archived_entry_id,
                topic=label.topic,
                short_summary=label.short_summary,
            )
        except (MemoryStoreError, IllegalStateTransitionError) as exc:
            if strict:
                raise
            return CompressionResult(mu_id=mu.mu_id, success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Batch compression
    # ------------------------------------------------------------------

    def compress_batch(self, mu_ids: list[str]) -> list[CompressionResult]:
        """Compress multiple MUs by id (*safe-result mode only*).

        Errors on individual MUs are captured as :class:`CompressionResult`
        entries with ``success=False``; the batch always runs to completion.
        Use :meth:`compress` with ``strict=True`` if you need loud failure for
        a single item before proceeding.
        """
        results: list[CompressionResult] = []
        for mu_id in mu_ids:
            results.append(self.compress(mu_id))
        return results

    def compress_mus(self, mus: list[MemoryUnit]) -> list[CompressionResult]:
        """Compress a list of pre-fetched MU objects (avoids extra DB reads)."""
        return [self._compress_mu(mu) for mu in mus]

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, mu_id: str) -> MemoryUnit:
        """Restore a Compressed MU back to Active.

        Delegates to ``store.restore_atomic`` which removes the label and
        archive rows and marks the MU for FAISS reindex.

        Raises:
            MemoryUnitNotFoundError: MU does not exist.
            IllegalStateTransitionError: MU is not currently compressed.
        """
        return self.store.restore_atomic(mu_id)

    def restore_forgotten(self, mu_id: str) -> MemoryUnit:
        """Restore a Forgotten MU back to Active.

        Raises:
            MemoryUnitNotFoundError: MU does not exist.
            IllegalStateTransitionError: MU is not currently forgotten.
        """
        return self.store.restore_from_forgotten(mu_id)

    # ------------------------------------------------------------------
    # Peek archive (non-destructive)
    # ------------------------------------------------------------------

    def peek_archive(self, mu_id: str) -> MemoryUnit | None:
        """Reconstruct the original MU from the archive snapshot.

        Deserializes ``full_memory_unit_json`` without changing any DB state.
        Returns ``None`` if the MU or its archive cannot be found.

        Useful for:
        - UI previews of what a compressed memory contained.
        - Search expansion: showing full claim text before deciding to restore.
        - Debugging archive integrity.
        """
        archive = self.store.get_archive_for_mu(mu_id)
        if archive is None:
            return None
        return _deserialize_mu(archive.full_memory_unit_json)

    def peek_archive_from_label(self, label_id: str) -> MemoryUnit | None:
        """Reconstruct the original MU given a label_id rather than mu_id."""
        label = self.store.get_compressed_label(label_id)
        if label is None:
            return None
        return self.peek_archive(label.mu_id)

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify(self, mu_id: str) -> VerificationResult:
        """Check internal consistency of a compressed MU's label + archive.

        Checks performed
        ----------------
        1. MU exists.
        2. MU status is COMPRESSED.
        3. ``mu.compressed_label_id`` is set and the label exists.
        4. ``mu.archived_entry_id`` is set and the archive exists.
        5. ``label.archived_pointer == archive.archived_entry_id``.
        6. ``archive.label_pointer == label.label_id``.
        7. ``archive.full_memory_unit_json`` is parseable as a MemoryUnit.
        8. The archived MU's ``mu_id`` matches the live MU's ``mu_id``.
        """
        issues: list[str] = []

        mu = self.store.get_memory_unit(mu_id)
        if mu is None:
            return VerificationResult(mu_id=mu_id, valid=False,
                                      issues=[f"MU not found: {mu_id}"])

        # A compressed MU sits in ARCHIVED status (the original-data tier);
        # legacy COMPRESSED rows are still acceptable for backward compatibility.
        if mu.status not in (MemoryStatus.ARCHIVED, MemoryStatus.COMPRESSED):
            issues.append(
                f"status is '{mu.status.value}', expected 'archived' (or legacy 'compressed')"
            )
            return VerificationResult(mu_id=mu_id, valid=False, issues=issues)

        # Label pointer
        if not mu.compressed_label_id:
            issues.append("compressed_label_id is not set")
            label = None
        else:
            label = self.store.get_compressed_label(mu.compressed_label_id)
            if label is None:
                issues.append(
                    f"label '{mu.compressed_label_id}' not found in compressed_labels"
                )

        # Archive pointer
        if not mu.archived_entry_id:
            issues.append("archived_entry_id is not set")
            archive = None
        else:
            archive = self.store.get_archived_entry(mu.archived_entry_id)
            if archive is None:
                issues.append(
                    f"archive '{mu.archived_entry_id}' not found in archived_entries"
                )

        # Cross-reference checks
        if label is not None and archive is not None:
            if label.archived_pointer != archive.archived_entry_id:
                issues.append(
                    f"label.archived_pointer '{label.archived_pointer}' != "
                    f"archive.archived_entry_id '{archive.archived_entry_id}'"
                )
            if archive.label_pointer != label.label_id:
                issues.append(
                    f"archive.label_pointer '{archive.label_pointer}' != "
                    f"label.label_id '{label.label_id}'"
                )

        # JSON integrity
        if archive is not None:
            archived_mu = _deserialize_mu(archive.full_memory_unit_json)
            if archived_mu is None:
                issues.append(
                    "full_memory_unit_json is not parseable as a valid MemoryUnit"
                )
            elif archived_mu.mu_id != mu_id:
                issues.append(
                    f"archived mu_id '{archived_mu.mu_id}' != live mu_id '{mu_id}'"
                )

        return VerificationResult(
            mu_id=mu_id,
            valid=len(issues) == 0,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self, conversation_id: str) -> CompressionStats:
        """Return per-status MU counts for the conversation."""
        counts = self.store.count_by_status(conversation_id)
        return CompressionStats(
            conversation_id=conversation_id,
            active=counts.get(MemoryStatus.ACTIVE, 0),
            compressed=counts.get(MemoryStatus.COMPRESSED, 0),
            forgotten=counts.get(MemoryStatus.FORGOTTEN, 0),
            archived=counts.get(MemoryStatus.ARCHIVED, 0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deserialize_mu(json_str: str) -> MemoryUnit | None:
    """Parse a JSON string as a MemoryUnit. Returns None on any error."""
    try:
        data = json.loads(json_str)
        return MemoryUnit.model_validate(data)
    except (json.JSONDecodeError, ValidationError, Exception):
        return None


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "CompressionResult",
    "CompressionService",
    "CompressionStats",
    "VerificationResult",
]
