"""SQLite source-of-truth store for Phase 2 memory.

This module is the single transactional layer. All FAISS indices and the
NetworkX graph are derived caches that can be rebuilt from this store at any
time. SQLite is the only thing that has to survive a crash.

Concurrency model
-----------------
- WAL journal mode enables concurrent reads.
- Writes are serialized by SQLite.
- Each ``transaction()`` opens a fresh connection. No shared mutable state in
  the ``MemoryStore`` instance, so a single instance is safe to share across
  threads (each call gets its own connection).
- For higher write throughput, instantiate one ``MemoryStore`` per thread,
  pointing to the same ``db_path``.

API stability
-------------
The public API is intentionally narrow and oriented around use-cases (insert,
update, status transitions, atomic compound operations). The schema is
versioned so future migrations stay backwards-compatible.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    DeletionAudit,
    EdgeRecord,
    EdgeType,
    MemoryStatus,
    MemoryUnit,
    utcnow,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MemoryStoreError(Exception):
    """Base exception for all MemoryStore failures."""


class MemoryUnitNotFoundError(MemoryStoreError):
    """Raised when a Memory Unit lookup or update targets a missing mu_id."""


class IllegalStateTransitionError(MemoryStoreError):
    """Raised when a state transition is not permitted from the current status."""


# ---------------------------------------------------------------------------
# Schema definition (idempotent: safe to run on every connect)
# ---------------------------------------------------------------------------


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_units (
        mu_id              TEXT PRIMARY KEY,
        conversation_id    TEXT NOT NULL,
        session_id         TEXT NOT NULL,

        claim              TEXT NOT NULL,
        original_text      TEXT NOT NULL,

        source_dia_ids     TEXT NOT NULL,
        source_speaker     TEXT NOT NULL,
        timestamp          TEXT,
        extracted_at       TEXT NOT NULL,

        salience_score     REAL NOT NULL,
        importance         REAL NOT NULL,
        recency_weight     REAL NOT NULL,
        uniqueness         REAL NOT NULL,
        retrieval_count    INTEGER NOT NULL DEFAULT 0,
        prompt_frequency   REAL NOT NULL DEFAULT 0.0,
        last_accessed      TEXT,

        status             TEXT NOT NULL,
        confidence         REAL NOT NULL,
        needs_reindex      INTEGER NOT NULL DEFAULT 0,

        compressed_label_id TEXT,
        archived_entry_id   TEXT,

        user_pinned        INTEGER NOT NULL DEFAULT 0,

        created_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mu_conv ON memory_units(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_mu_status ON memory_units(status)",
    "CREATE INDEX IF NOT EXISTS idx_mu_status_conv ON memory_units(conversation_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_mu_reindex ON memory_units(needs_reindex)",
    """
    CREATE TABLE IF NOT EXISTS compressed_labels (
        label_id           TEXT PRIMARY KEY,
        archived_pointer   TEXT NOT NULL,
        mu_id              TEXT NOT NULL,
        conversation_id    TEXT NOT NULL,

        topic              TEXT NOT NULL,
        short_summary      TEXT NOT NULL,
        key_entities       TEXT NOT NULL,
        time_range         TEXT,

        original_dia_ids   TEXT NOT NULL,

        compressed_at      TEXT NOT NULL,
        retrieval_count    INTEGER NOT NULL DEFAULT 0,
        last_label_match   TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_label_mu ON compressed_labels(mu_id)",
    "CREATE INDEX IF NOT EXISTS idx_label_conv ON compressed_labels(conversation_id)",
    """
    CREATE TABLE IF NOT EXISTS archived_entries (
        archived_entry_id     TEXT PRIMARY KEY,
        label_pointer         TEXT NOT NULL,
        mu_id                 TEXT NOT NULL,
        conversation_id       TEXT NOT NULL,

        full_memory_unit_json TEXT NOT NULL,
        full_original_text    TEXT NOT NULL,

        archived_at           TEXT NOT NULL,
        restoration_count     INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_archive_mu ON archived_entries(mu_id)",
    "CREATE INDEX IF NOT EXISTS idx_archive_label ON archived_entries(label_pointer)",
    """
    CREATE TABLE IF NOT EXISTS edges (
        edge_id           TEXT PRIMARY KEY,
        source_mu_id      TEXT NOT NULL,
        target_mu_id      TEXT NOT NULL,
        edge_type         TEXT NOT NULL,
        weight            REAL NOT NULL DEFAULT 1.0,
        created_at        TEXT NOT NULL,
        metadata_json     TEXT,
        UNIQUE(source_mu_id, target_mu_id, edge_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_mu_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_mu_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)",
    """
    CREATE TABLE IF NOT EXISTS deletion_audit (
        audit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        mu_id             TEXT NOT NULL,
        conversation_id   TEXT NOT NULL,
        source_dia_ids    TEXT NOT NULL,
        deleted_at        TEXT NOT NULL,
        deleted_by        TEXT NOT NULL DEFAULT 'user'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_conv ON deletion_audit(conversation_id)",
)

CURRENT_SCHEMA_VERSION = 1

# Placeholder text written to claim/original_text on permanent deletion.
# Keeps the row valid against the MemoryUnit schema (claim has min_length=1)
# while clearly signalling that the original content has been removed.
DELETED_PLACEHOLDER = "[deleted]"


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """SQLite-backed source of truth for Phase 2.

    Args:
        db_path: filesystem path for the SQLite database. Parent directories
            are created if missing. The schema is initialized idempotently
            on construction.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Opening MemoryStore at {}", self.db_path)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), isolation_level="DEFERRED")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager yielding a connection inside a transaction.

        Commits on success, rolls back on any exception, always closes.
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        """Read-only connection context manager."""
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema init (idempotent)
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self.transaction() as conn:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(stmt)
            row = conn.execute(
                "SELECT MAX(version) AS v FROM schema_version"
            ).fetchone()
            installed = row["v"] if row and row["v"] is not None else 0
            if installed < CURRENT_SCHEMA_VERSION:
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (CURRENT_SCHEMA_VERSION, _utcnow_iso()),
                )
                logger.info(
                    "Initialized SQLite schema v{} at {}",
                    CURRENT_SCHEMA_VERSION,
                    self.db_path,
                )

    # ==================================================================
    # Memory Units
    # ==================================================================

    def insert_memory_unit(self, mu: MemoryUnit) -> None:
        """Insert a new Memory Unit. Raises ``sqlite3.IntegrityError`` on duplicate id."""
        with self.transaction() as conn:
            self._insert_mu(conn, mu)

    @staticmethod
    def _insert_mu(conn: sqlite3.Connection, mu: MemoryUnit) -> None:
        conn.execute(
            """
            INSERT INTO memory_units (
                mu_id, conversation_id, session_id,
                claim, original_text,
                source_dia_ids, source_speaker, timestamp, extracted_at,
                salience_score, importance, recency_weight, uniqueness,
                retrieval_count, prompt_frequency, last_accessed,
                status, confidence, needs_reindex,
                compressed_label_id, archived_entry_id,
                user_pinned, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                mu.mu_id, mu.conversation_id, mu.session_id,
                mu.claim, mu.original_text,
                json.dumps(mu.source_dia_ids), mu.source_speaker,
                mu.timestamp, _iso(mu.extracted_at),
                mu.salience_score, mu.importance, mu.recency_weight, mu.uniqueness,
                mu.retrieval_count, mu.prompt_frequency, _iso_or_none(mu.last_accessed),
                mu.status.value, mu.confidence, int(mu.needs_reindex),
                mu.compressed_label_id, mu.archived_entry_id,
                int(mu.user_pinned), _iso(mu.created_at), _iso(mu.updated_at),
            ),
        )

    def get_memory_unit(self, mu_id: str) -> MemoryUnit | None:
        """Fetch a Memory Unit by id, or ``None`` if not found."""
        with self.reader() as conn:
            row = conn.execute(
                "SELECT * FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
        return _row_to_mu(row) if row else None

    def get_memory_unit_or_raise(self, mu_id: str) -> MemoryUnit:
        """Fetch a Memory Unit by id, raising ``MemoryUnitNotFoundError`` if missing."""
        mu = self.get_memory_unit(mu_id)
        if mu is None:
            raise MemoryUnitNotFoundError(mu_id)
        return mu

    def update_memory_unit(self, mu: MemoryUnit) -> None:
        """Replace all mutable fields of an existing MU. Bumps ``updated_at``."""
        mu.updated_at = utcnow()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_units SET
                    conversation_id = ?, session_id = ?,
                    claim = ?, original_text = ?,
                    source_dia_ids = ?, source_speaker = ?,
                    timestamp = ?, extracted_at = ?,
                    salience_score = ?, importance = ?, recency_weight = ?, uniqueness = ?,
                    retrieval_count = ?, prompt_frequency = ?, last_accessed = ?,
                    status = ?, confidence = ?, needs_reindex = ?,
                    compressed_label_id = ?, archived_entry_id = ?,
                    user_pinned = ?, updated_at = ?
                WHERE mu_id = ?
                """,
                (
                    mu.conversation_id, mu.session_id,
                    mu.claim, mu.original_text,
                    json.dumps(mu.source_dia_ids), mu.source_speaker,
                    mu.timestamp, _iso(mu.extracted_at),
                    mu.salience_score, mu.importance, mu.recency_weight, mu.uniqueness,
                    mu.retrieval_count, mu.prompt_frequency, _iso_or_none(mu.last_accessed),
                    mu.status.value, mu.confidence, int(mu.needs_reindex),
                    mu.compressed_label_id, mu.archived_entry_id,
                    int(mu.user_pinned), _iso(mu.updated_at),
                    mu.mu_id,
                ),
            )
            if cursor.rowcount == 0:
                raise MemoryUnitNotFoundError(mu.mu_id)

    def update_status(
        self,
        mu_id: str,
        new_status: MemoryStatus,
        compressed_label_id: str | None = None,
        archived_entry_id: str | None = None,
    ) -> None:
        """Update only the status fields and compression linkage."""
        with self.transaction() as conn:
            self._update_status(
                conn, mu_id, new_status, compressed_label_id, archived_entry_id
            )

    @staticmethod
    def _update_status(
        conn: sqlite3.Connection,
        mu_id: str,
        new_status: MemoryStatus,
        compressed_label_id: str | None,
        archived_entry_id: str | None,
    ) -> None:
        cursor = conn.execute(
            """
            UPDATE memory_units
               SET status = ?,
                   compressed_label_id = ?,
                   archived_entry_id = ?,
                   updated_at = ?
             WHERE mu_id = ?
            """,
            (
                new_status.value, compressed_label_id, archived_entry_id,
                _utcnow_iso(), mu_id,
            ),
        )
        if cursor.rowcount == 0:
            raise MemoryUnitNotFoundError(mu_id)

    def increment_retrieval_count(self, mu_id: str) -> None:
        """Atomically increment retrieval_count and bump ``last_accessed``."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_units
                   SET retrieval_count = retrieval_count + 1,
                       last_accessed = ?,
                       updated_at = ?
                 WHERE mu_id = ?
                """,
                (_utcnow_iso(), _utcnow_iso(), mu_id),
            )
            if cursor.rowcount == 0:
                raise MemoryUnitNotFoundError(mu_id)

    def set_pinned(self, mu_id: str, pinned: bool) -> None:
        """Pin or unpin a Memory Unit."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE memory_units SET user_pinned = ?, updated_at = ? WHERE mu_id = ?",
                (int(pinned), _utcnow_iso(), mu_id),
            )
            if cursor.rowcount == 0:
                raise MemoryUnitNotFoundError(mu_id)

    def list_by_status(
        self, conversation_id: str, status: MemoryStatus
    ) -> list[MemoryUnit]:
        """List all MUs in a conversation with the given status, ordered by creation time."""
        with self.reader() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_units "
                "WHERE conversation_id = ? AND status = ? "
                "ORDER BY created_at",
                (conversation_id, status.value),
            ).fetchall()
        return [_row_to_mu(r) for r in rows]

    def list_active(self, conversation_id: str) -> list[MemoryUnit]:
        """Convenience: list active MUs for a conversation."""
        return self.list_by_status(conversation_id, MemoryStatus.ACTIVE)

    def list_all(self, conversation_id: str) -> list[MemoryUnit]:
        """List all MUs for a conversation regardless of status."""
        with self.reader() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_units WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        return [_row_to_mu(r) for r in rows]

    def count_by_status(self, conversation_id: str) -> dict[MemoryStatus, int]:
        """Return a dict {status: count} for a conversation. Missing statuses → 0."""
        with self.reader() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n "
                "FROM memory_units WHERE conversation_id = ? "
                "GROUP BY status",
                (conversation_id,),
            ).fetchall()
        result: dict[MemoryStatus, int] = {s: 0 for s in MemoryStatus}
        for row in rows:
            try:
                result[MemoryStatus(row["status"])] = row["n"]
            except ValueError:
                logger.warning("Unknown status string in DB: {}", row["status"])
        return result

    def storage_pressure(self, conversation_id: str, cap: int) -> float:
        """Active MU count divided by cap. Returns 0.0 if cap <= 0."""
        if cap <= 0:
            return 0.0
        active = self.count_by_status(conversation_id).get(MemoryStatus.ACTIVE, 0)
        return active / cap

    def iter_active(
        self, conversation_id: str | None = None
    ) -> Iterator[MemoryUnit]:
        """Stream active MUs (optionally scoped to one conversation)."""
        with self.reader() as conn:
            if conversation_id is not None:
                cursor = conn.execute(
                    "SELECT * FROM memory_units WHERE conversation_id = ? AND status = ?",
                    (conversation_id, MemoryStatus.ACTIVE.value),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM memory_units WHERE status = ?",
                    (MemoryStatus.ACTIVE.value,),
                )
            for row in cursor:
                yield _row_to_mu(row)

    # ------------------------------------------------------------------
    # Reindex flag (used by FAISS / graph rebuild sweeper)
    # ------------------------------------------------------------------

    def list_needing_reindex(self) -> list[MemoryUnit]:
        with self.reader() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_units WHERE needs_reindex = 1"
            ).fetchall()
        return [_row_to_mu(r) for r in rows]

    def mark_needs_reindex(self, mu_id: str) -> None:
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE memory_units SET needs_reindex = 1, updated_at = ? WHERE mu_id = ?",
                (_utcnow_iso(), mu_id),
            )
            if cursor.rowcount == 0:
                raise MemoryUnitNotFoundError(mu_id)

    def clear_reindex_flag(self, mu_id: str) -> None:
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE memory_units SET needs_reindex = 0, updated_at = ? WHERE mu_id = ?",
                (_utcnow_iso(), mu_id),
            )
            if cursor.rowcount == 0:
                raise MemoryUnitNotFoundError(mu_id)

    # ==================================================================
    # Compressed Labels
    # ==================================================================

    def insert_compressed_label(self, label: CompressedLabel) -> None:
        with self.transaction() as conn:
            self._insert_label(conn, label)

    @staticmethod
    def _insert_label(conn: sqlite3.Connection, label: CompressedLabel) -> None:
        conn.execute(
            """
            INSERT INTO compressed_labels (
                label_id, archived_pointer, mu_id, conversation_id,
                topic, short_summary, key_entities, time_range,
                original_dia_ids, compressed_at, retrieval_count, last_label_match
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                label.label_id, label.archived_pointer,
                label.mu_id, label.conversation_id,
                label.topic, label.short_summary,
                json.dumps(label.key_entities), label.time_range,
                json.dumps(label.original_dia_ids),
                _iso(label.compressed_at), label.retrieval_count,
                _iso_or_none(label.last_label_match),
            ),
        )

    def get_compressed_label(self, label_id: str) -> CompressedLabel | None:
        with self.reader() as conn:
            row = conn.execute(
                "SELECT * FROM compressed_labels WHERE label_id = ?", (label_id,)
            ).fetchone()
        return _row_to_label(row) if row else None

    def get_label_for_mu(self, mu_id: str) -> CompressedLabel | None:
        with self.reader() as conn:
            row = conn.execute(
                "SELECT * FROM compressed_labels WHERE mu_id = ?", (mu_id,)
            ).fetchone()
        return _row_to_label(row) if row else None

    def list_compressed_labels(self, conversation_id: str) -> list[CompressedLabel]:
        with self.reader() as conn:
            rows = conn.execute(
                "SELECT * FROM compressed_labels WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
        return [_row_to_label(r) for r in rows]

    def iter_labels(
        self, conversation_id: str | None = None
    ) -> Iterator[CompressedLabel]:
        with self.reader() as conn:
            if conversation_id is not None:
                cursor = conn.execute(
                    "SELECT * FROM compressed_labels WHERE conversation_id = ?",
                    (conversation_id,),
                )
            else:
                cursor = conn.execute("SELECT * FROM compressed_labels")
            for row in cursor:
                yield _row_to_label(row)

    # ==================================================================
    # Archived Entries
    # ==================================================================

    def insert_archived_entry(self, entry: ArchivedEntry) -> None:
        with self.transaction() as conn:
            self._insert_archive(conn, entry)

    @staticmethod
    def _insert_archive(conn: sqlite3.Connection, entry: ArchivedEntry) -> None:
        conn.execute(
            """
            INSERT INTO archived_entries (
                archived_entry_id, label_pointer, mu_id, conversation_id,
                full_memory_unit_json, full_original_text,
                archived_at, restoration_count
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                entry.archived_entry_id, entry.label_pointer,
                entry.mu_id, entry.conversation_id,
                entry.full_memory_unit_json, entry.full_original_text,
                _iso(entry.archived_at), entry.restoration_count,
            ),
        )

    def get_archived_entry(self, entry_id: str) -> ArchivedEntry | None:
        with self.reader() as conn:
            row = conn.execute(
                "SELECT * FROM archived_entries WHERE archived_entry_id = ?",
                (entry_id,),
            ).fetchone()
        return _row_to_archive(row) if row else None

    def get_archive_for_mu(self, mu_id: str) -> ArchivedEntry | None:
        with self.reader() as conn:
            row = conn.execute(
                "SELECT * FROM archived_entries WHERE mu_id = ?", (mu_id,)
            ).fetchone()
        return _row_to_archive(row) if row else None

    # ==================================================================
    # Edges
    # ==================================================================

    def insert_edge(self, edge: EdgeRecord) -> None:
        """Insert a typed edge. Raises ``MemoryStoreError`` on duplicate (src, tgt, type)."""
        with self.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO edges (
                        edge_id, source_mu_id, target_mu_id, edge_type, weight,
                        created_at, metadata_json
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        edge.edge_id, edge.source_mu_id, edge.target_mu_id,
                        edge.edge_type.value, edge.weight,
                        _iso(edge.created_at), edge.metadata_json,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise MemoryStoreError(
                    f"Duplicate edge: {edge.source_mu_id} -[{edge.edge_type.value}]-> "
                    f"{edge.target_mu_id}"
                ) from exc

    def get_edge(self, edge_id: str) -> EdgeRecord | None:
        with self.reader() as conn:
            row = conn.execute(
                "SELECT * FROM edges WHERE edge_id = ?", (edge_id,)
            ).fetchone()
        return _row_to_edge(row) if row else None

    def remove_edge(self, edge_id: str) -> None:
        """Delete an edge by id. No-op if not present."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM edges WHERE edge_id = ?", (edge_id,))

    def edges_from(
        self, mu_id: str, edge_type: EdgeType | None = None
    ) -> list[EdgeRecord]:
        with self.reader() as conn:
            if edge_type is None:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE source_mu_id = ?", (mu_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE source_mu_id = ? AND edge_type = ?",
                    (mu_id, edge_type.value),
                ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def edges_to(
        self, mu_id: str, edge_type: EdgeType | None = None
    ) -> list[EdgeRecord]:
        with self.reader() as conn:
            if edge_type is None:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE target_mu_id = ?", (mu_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE target_mu_id = ? AND edge_type = ?",
                    (mu_id, edge_type.value),
                ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def iter_edges(self) -> Iterator[EdgeRecord]:
        with self.reader() as conn:
            cursor = conn.execute("SELECT * FROM edges")
            for row in cursor:
                yield _row_to_edge(row)

    # ==================================================================
    # Atomic compound state transitions
    # ==================================================================

    def compress_atomic(
        self,
        mu_id: str,
        label: CompressedLabel,
        archive: ArchivedEntry,
    ) -> None:
        """Atomically compress an Active Memory Unit.

        Inserts the archived entry, the compressed label, and updates the MU
        status to COMPRESSED with linkage. All writes succeed together or none.

        Raises:
            MemoryUnitNotFoundError: if the MU does not exist.
            IllegalStateTransitionError: if the MU is not currently active.
            MemoryStoreError: if pointer cross-references are inconsistent.
        """
        if archive.mu_id != mu_id or label.mu_id != mu_id:
            raise MemoryStoreError(
                f"compress_atomic: mu_id mismatch (mu_id={mu_id}, "
                f"label.mu_id={label.mu_id}, archive.mu_id={archive.mu_id})"
            )
        if label.archived_pointer != archive.archived_entry_id:
            raise MemoryStoreError(
                "compress_atomic: label.archived_pointer does not match "
                "archive.archived_entry_id"
            )
        if archive.label_pointer != label.label_id:
            raise MemoryStoreError(
                "compress_atomic: archive.label_pointer does not match label.label_id"
            )

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
            if row is None:
                raise MemoryUnitNotFoundError(mu_id)
            current_status = MemoryStatus(row["status"])
            if current_status != MemoryStatus.ACTIVE:
                raise IllegalStateTransitionError(
                    f"cannot compress MU in status {current_status.value}; "
                    "only active MUs can be compressed"
                )
            self._insert_archive(conn, archive)
            self._insert_label(conn, label)
            self._update_status(
                conn, mu_id, MemoryStatus.COMPRESSED,
                label.label_id, archive.archived_entry_id,
            )
            logger.info(
                "Compressed mu_id={} (label={}, archive={})",
                mu_id, label.label_id, archive.archived_entry_id,
            )

    def restore_atomic(self, mu_id: str) -> MemoryUnit:
        """Atomically restore a Compressed MU back to Active.

        Sets status to ACTIVE, removes the label and archive (clean invariant),
        increments archive restoration count, and marks the MU for reindex
        (FAISS active-index needs the embedding back).

        Returns the updated Memory Unit.

        Raises:
            MemoryUnitNotFoundError: if the MU does not exist.
            IllegalStateTransitionError: if the MU is not currently compressed.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
            if row is None:
                raise MemoryUnitNotFoundError(mu_id)
            mu = _row_to_mu(row)
            if mu.status != MemoryStatus.COMPRESSED:
                raise IllegalStateTransitionError(
                    f"cannot restore MU in status {mu.status.value}; "
                    "only compressed MUs are restorable via this path"
                )
            label_id = mu.compressed_label_id
            archive_id = mu.archived_entry_id

            if archive_id:
                conn.execute(
                    "UPDATE archived_entries "
                    "SET restoration_count = restoration_count + 1 "
                    "WHERE archived_entry_id = ?",
                    (archive_id,),
                )

            conn.execute(
                """
                UPDATE memory_units
                   SET status = ?,
                       compressed_label_id = NULL,
                       archived_entry_id = NULL,
                       needs_reindex = 1,
                       updated_at = ?
                 WHERE mu_id = ?
                """,
                (MemoryStatus.ACTIVE.value, _utcnow_iso(), mu_id),
            )

            if label_id:
                conn.execute(
                    "DELETE FROM compressed_labels WHERE label_id = ?", (label_id,)
                )
            if archive_id:
                conn.execute(
                    "DELETE FROM archived_entries WHERE archived_entry_id = ?",
                    (archive_id,),
                )

            logger.info("Restored mu_id={} from compressed", mu_id)

            updated = conn.execute(
                "SELECT * FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
            return _row_to_mu(updated)

    def forget_atomic(self, mu_id: str) -> None:
        """Atomically move a MU to FORGOTTEN.

        - Removes any compressed label + archive linkage.
        - Marks for reindex (will be removed from active FAISS).
        - Preserves MU content in SQLite — restorable later.
        - Idempotent on already-forgotten MUs.

        Raises:
            MemoryUnitNotFoundError: if the MU does not exist.
            IllegalStateTransitionError: if the MU is currently DELETED.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
            if row is None:
                raise MemoryUnitNotFoundError(mu_id)
            mu = _row_to_mu(row)
            if mu.status == MemoryStatus.DELETED:
                raise IllegalStateTransitionError(
                    "cannot forget a deleted MU; deletion is terminal"
                )
            if mu.status == MemoryStatus.FORGOTTEN:
                return  # idempotent

            label_id = mu.compressed_label_id
            archive_id = mu.archived_entry_id

            conn.execute(
                """
                UPDATE memory_units
                   SET status = ?,
                       compressed_label_id = NULL,
                       archived_entry_id = NULL,
                       needs_reindex = 1,
                       updated_at = ?
                 WHERE mu_id = ?
                """,
                (MemoryStatus.FORGOTTEN.value, _utcnow_iso(), mu_id),
            )

            if label_id:
                conn.execute(
                    "DELETE FROM compressed_labels WHERE label_id = ?", (label_id,)
                )
            if archive_id:
                conn.execute(
                    "DELETE FROM archived_entries WHERE archived_entry_id = ?",
                    (archive_id,),
                )

            logger.info("Forgot mu_id={}", mu_id)

    def restore_from_forgotten(self, mu_id: str) -> MemoryUnit:
        """User-triggered restoration of a Forgotten MU back to Active.

        Returns the updated Memory Unit.

        Raises:
            MemoryUnitNotFoundError: if the MU does not exist.
            IllegalStateTransitionError: if the MU is not currently forgotten.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
            if row is None:
                raise MemoryUnitNotFoundError(mu_id)
            mu = _row_to_mu(row)
            if mu.status != MemoryStatus.FORGOTTEN:
                raise IllegalStateTransitionError(
                    f"cannot restore from forgotten: MU is in status {mu.status.value}"
                )
            conn.execute(
                """
                UPDATE memory_units
                   SET status = ?, needs_reindex = 1, updated_at = ?
                 WHERE mu_id = ?
                """,
                (MemoryStatus.ACTIVE.value, _utcnow_iso(), mu_id),
            )
            logger.info("Restored mu_id={} from forgotten", mu_id)

            updated = conn.execute(
                "SELECT * FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
            return _row_to_mu(updated)

    def delete_atomic(self, mu_id: str, deleted_by: str = "user") -> None:
        """Permanently delete a MU. The ONLY path to true deletion.

        - Status → DELETED, content fields nulled out.
        - Audit row inserted in deletion_audit.
        - Linked label + archive removed.
        - Edges referencing this MU removed.
        - Idempotent on already-deleted MUs.

        Raises:
            MemoryUnitNotFoundError: if the MU does not exist.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM memory_units WHERE mu_id = ?", (mu_id,)
            ).fetchone()
            if row is None:
                raise MemoryUnitNotFoundError(mu_id)
            mu = _row_to_mu(row)
            if mu.status == MemoryStatus.DELETED:
                return  # idempotent

            label_id = mu.compressed_label_id
            archive_id = mu.archived_entry_id

            conn.execute(
                """
                INSERT INTO deletion_audit
                    (mu_id, conversation_id, source_dia_ids, deleted_at, deleted_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    mu.mu_id, mu.conversation_id,
                    json.dumps(mu.source_dia_ids),
                    _utcnow_iso(), deleted_by,
                ),
            )

            conn.execute(
                """
                UPDATE memory_units
                   SET status = ?,
                       claim = ?,
                       original_text = ?,
                       compressed_label_id = NULL,
                       archived_entry_id = NULL,
                       needs_reindex = 1,
                       updated_at = ?
                 WHERE mu_id = ?
                """,
                (
                    MemoryStatus.DELETED.value,
                    DELETED_PLACEHOLDER,
                    DELETED_PLACEHOLDER,
                    _utcnow_iso(),
                    mu_id,
                ),
            )

            if label_id:
                conn.execute(
                    "DELETE FROM compressed_labels WHERE label_id = ?", (label_id,)
                )
            if archive_id:
                conn.execute(
                    "DELETE FROM archived_entries WHERE archived_entry_id = ?",
                    (archive_id,),
                )

            conn.execute(
                "DELETE FROM edges WHERE source_mu_id = ? OR target_mu_id = ?",
                (mu_id, mu_id),
            )

            logger.info("Deleted mu_id={} by {}", mu_id, deleted_by)

    def list_deletion_audit(
        self, conversation_id: str | None = None
    ) -> list[DeletionAudit]:
        with self.reader() as conn:
            if conversation_id is not None:
                rows = conn.execute(
                    "SELECT * FROM deletion_audit WHERE conversation_id = ? ORDER BY audit_id",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM deletion_audit ORDER BY audit_id"
                ).fetchall()
        return [_row_to_audit(r) for r in rows]


# ---------------------------------------------------------------------------
# Row → model converters
# ---------------------------------------------------------------------------


def _row_to_mu(row: sqlite3.Row) -> MemoryUnit:
    return MemoryUnit(
        mu_id=row["mu_id"],
        conversation_id=row["conversation_id"],
        session_id=row["session_id"],
        claim=row["claim"],
        original_text=row["original_text"],
        source_dia_ids=json.loads(row["source_dia_ids"]),
        source_speaker=row["source_speaker"],
        timestamp=row["timestamp"],
        extracted_at=_parse_iso(row["extracted_at"]),
        salience_score=row["salience_score"],
        importance=row["importance"],
        recency_weight=row["recency_weight"],
        uniqueness=row["uniqueness"],
        retrieval_count=row["retrieval_count"],
        prompt_frequency=row["prompt_frequency"],
        last_accessed=_parse_iso_or_none(row["last_accessed"]),
        status=MemoryStatus(row["status"]),
        confidence=row["confidence"],
        needs_reindex=bool(row["needs_reindex"]),
        compressed_label_id=row["compressed_label_id"],
        archived_entry_id=row["archived_entry_id"],
        user_pinned=bool(row["user_pinned"]),
        created_at=_parse_iso(row["created_at"]),
        updated_at=_parse_iso(row["updated_at"]),
    )


def _row_to_label(row: sqlite3.Row) -> CompressedLabel:
    return CompressedLabel(
        label_id=row["label_id"],
        archived_pointer=row["archived_pointer"],
        mu_id=row["mu_id"],
        conversation_id=row["conversation_id"],
        topic=row["topic"],
        short_summary=row["short_summary"],
        key_entities=json.loads(row["key_entities"]),
        time_range=row["time_range"],
        original_dia_ids=json.loads(row["original_dia_ids"]),
        compressed_at=_parse_iso(row["compressed_at"]),
        retrieval_count=row["retrieval_count"],
        last_label_match=_parse_iso_or_none(row["last_label_match"]),
    )


def _row_to_archive(row: sqlite3.Row) -> ArchivedEntry:
    return ArchivedEntry(
        archived_entry_id=row["archived_entry_id"],
        label_pointer=row["label_pointer"],
        mu_id=row["mu_id"],
        conversation_id=row["conversation_id"],
        full_memory_unit_json=row["full_memory_unit_json"],
        full_original_text=row["full_original_text"],
        archived_at=_parse_iso(row["archived_at"]),
        restoration_count=row["restoration_count"],
    )


def _row_to_edge(row: sqlite3.Row) -> EdgeRecord:
    return EdgeRecord(
        edge_id=row["edge_id"],
        source_mu_id=row["source_mu_id"],
        target_mu_id=row["target_mu_id"],
        edge_type=EdgeType(row["edge_type"]),
        weight=row["weight"],
        created_at=_parse_iso(row["created_at"]),
        metadata_json=row["metadata_json"],
    )


def _row_to_audit(row: sqlite3.Row) -> DeletionAudit:
    return DeletionAudit(
        audit_id=row["audit_id"],
        mu_id=row["mu_id"],
        conversation_id=row["conversation_id"],
        source_dia_ids=json.loads(row["source_dia_ids"]),
        deleted_at=_parse_iso(row["deleted_at"]),
        deleted_by=row["deleted_by"],
    )


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _iso_or_none(dt: datetime | None) -> str | None:
    return _iso(dt) if dt is not None else None


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_iso_or_none(s: str | None) -> datetime | None:
    return _parse_iso(s) if s else None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
