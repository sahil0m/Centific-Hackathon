"""Phase 2 Lifecycle Engine (Milestone 5).

Owns the active-memory capacity trigger (90 %) and orchestrates
Active → Compressed / Forgotten transitions.
"""

from __future__ import annotations

from locomo_memory.phase2.lifecycle.engine import (
    LabelBuilder,
    LifecycleBatch,
    LifecycleConfig,
    LifecycleEngine,
    TransitionRecord,
)

__all__ = [
    "LabelBuilder",
    "LifecycleBatch",
    "LifecycleConfig",
    "LifecycleEngine",
    "TransitionRecord",
]
