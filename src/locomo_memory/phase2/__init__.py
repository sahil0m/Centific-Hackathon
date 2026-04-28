"""Phase 2 — SPARC-LTM.

Salience and Provenance Aware Reconciliation and Compression for Long-Term Memory.

Implements a 4-tier memory system (Active / Compressed / Archived / Forgotten)
plus a permanent Deleted audit state, with SQLite as the single source of truth
and FAISS + NetworkX as rebuildable derived indexes.

See PHASE2_METHODOLOGY.md at the repo root for the full design.
"""

from __future__ import annotations

__all__: list[str] = []
