"""Phase 2 Compression, Archive, and Restore Service (Milestone 6).

Handles the mechanics of compressing individual MemoryUnits into
CompressedLabel + ArchivedEntry pairs, restoring them to active, and
verifying archive integrity. The Lifecycle Engine uses this service for
the actual compression step.
"""

from __future__ import annotations

from locomo_memory.phase2.compression.service import (
    CompressionResult,
    CompressionService,
    CompressionStats,
    VerificationResult,
)

__all__ = [
    "CompressionResult",
    "CompressionService",
    "CompressionStats",
    "VerificationResult",
]
