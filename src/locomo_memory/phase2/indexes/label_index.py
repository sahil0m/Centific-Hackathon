"""FAISS dense index over CompressedLabel short summaries — Phase 2 Milestone 8B.

Mirrors :class:`~locomo_memory.phase2.indexes.faiss_index.MemoryFAISSIndex`
but indexes :class:`~locomo_memory.phase2.schemas.CompressedLabel` objects
instead of :class:`~locomo_memory.phase2.schemas.MemoryUnit`.

The text embedded for each label is::

    "{topic}: {short_summary} entities: {key_entities_csv}"

This enriches the embedding with topical signal so label search can find
compressed memories that match a query semantically even when the original
claim text is unavailable.

Design notes
------------
- Uses the same ``IndexIDMap(IndexFlatIP(dim))`` pattern as ``MemoryFAISSIndex``.
- Soft deletion via ``_deleted`` set, same as the active-MU index.
- ``label_id`` is mapped to/from a monotonic ``int64``.
- Thread safety: same constraints as ``MemoryFAISSIndex`` (not thread-safe on writes).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from loguru import logger

from locomo_memory.phase2.indexes.faiss_index import EmbedFn
from locomo_memory.phase2.schemas import CompressedLabel
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_COMPACT_THRESHOLD: Final[float] = 0.30


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LabelSearchResult:
    """One hit from a compressed-label FAISS search."""

    label_id: str
    mu_id: str
    score: float
    rank: int
    conversation_id: str
    short_summary: str


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class CompressedLabelFAISSIndex:
    """Dense FAISS index over CompressedLabel short summaries.

    Args:
        embed_fn: same injected embed function as ``MemoryFAISSIndex``.
        dim: embedding dimensionality.
        normalize: L2-normalize vectors (inner product → cosine similarity).
        compact_threshold: fraction of soft-deleted entries that triggers
            an automatic compact on the next search.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        dim: int = 384,
        *,
        normalize: bool = True,
        compact_threshold: float = _DEFAULT_COMPACT_THRESHOLD,
    ) -> None:
        import faiss

        self.embed_fn = embed_fn
        self.dim = dim
        self.normalize = normalize
        self.compact_threshold = compact_threshold

        self._faiss = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._id_to_label: dict[int, str] = {}           # int64 → label_id
        self._label_to_id: dict[str, int] = {}           # label_id → int64
        self._id_to_mu: dict[int, str] = {}              # int64 → mu_id
        self._id_to_conv: dict[int, str] = {}            # int64 → conversation_id
        self._id_to_summary: dict[int, str] = {}         # int64 → short_summary
        self._deleted: set[int] = set()                  # soft-deleted int64 ids
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._label_to_id)

    def __repr__(self) -> str:
        return f"CompressedLabelFAISSIndex(size={len(self)}, dim={self.dim})"

    def size(self) -> int:
        return len(self)

    def label_ids(self) -> list[str]:
        live_ids = set(self._id_to_label.values()) - {
            self._id_to_label[d] for d in self._deleted if d in self._id_to_label
        }
        return list(live_ids)

    @property
    def needs_compact(self) -> bool:
        total = self._faiss.ntotal
        if total == 0:
            return False
        return (len(self._deleted) / total) > self.compact_threshold

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _label_text(label: CompressedLabel) -> str:
        """Build the embedding string: topic + summary + entities."""
        parts = []
        if label.topic:
            parts.append(f"{label.topic}:")
        parts.append(label.short_summary)
        if label.key_entities:
            parts.append("entities: " + ", ".join(label.key_entities))
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_label(self, label: CompressedLabel) -> None:
        """Embed and add a single label. Re-adds replace the old entry."""
        self.add_labels([label])

    def add_labels(self, labels: list[CompressedLabel]) -> None:
        """Batch add labels."""
        if not labels:
            return
        texts = [self._label_text(lb) for lb in labels]
        vecs = self._embed(texts)

        ids: list[int] = []
        for label, _vec in zip(labels, vecs):
            if label.label_id in self._label_to_id:
                old_id = self._label_to_id[label.label_id]
                self._deleted.add(old_id)

            new_id = self._next_id
            self._next_id += 1
            self._id_to_label[new_id] = label.label_id
            self._label_to_id[label.label_id] = new_id
            self._id_to_mu[new_id] = label.mu_id
            self._id_to_conv[new_id] = label.conversation_id
            self._id_to_summary[new_id] = label.short_summary
            ids.append(new_id)

        int_ids = np.array(ids, dtype=np.int64)
        self._faiss.add_with_ids(vecs, int_ids)
        logger.debug("CompressedLabelFAISSIndex: added {} label(s)", len(labels))

    def remove_label(self, label_id: str) -> bool:
        """Soft-delete a label. Returns ``True`` if found."""
        int_id = self._label_to_id.pop(label_id, None)
        if int_id is None:
            return False
        self._deleted.add(int_id)
        return True

    def compact(self) -> int:
        """Clean up deleted-entry metadata. Returns live count."""
        for del_id in list(self._deleted):
            self._id_to_label.pop(del_id, None)
            self._id_to_mu.pop(del_id, None)
            self._id_to_conv.pop(del_id, None)
            self._id_to_summary.pop(del_id, None)
        self._deleted.clear()
        return len(self._label_to_id)

    def rebuild(self, labels: list[CompressedLabel]) -> None:
        """Full clear + re-embed + re-add."""
        import faiss

        self._faiss = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self._id_to_label = {}
        self._label_to_id = {}
        self._id_to_mu = {}
        self._id_to_conv = {}
        self._id_to_summary = {}
        self._deleted = set()
        self._next_id = 0
        if labels:
            self.add_labels(labels)

    def rebuild_from_store(
        self,
        store: MemoryStore,
        *,
        conversation_id: str | None = None,
    ) -> int:
        """Rebuild from all compressed labels in the store.

        Returns the number of labels indexed.
        """
        if conversation_id is not None:
            labels = store.list_compressed_labels(conversation_id)
        else:
            labels = list(store.iter_labels())
        self.rebuild(labels)
        logger.info(
            "CompressedLabelFAISSIndex: rebuild_from_store: {} label(s) indexed (conv={})",
            len(labels), conversation_id or "all",
        )
        return len(labels)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        top_k: int,
        *,
        conversation_id: str | None = None,
    ) -> list[LabelSearchResult]:
        """Embed ``query_text`` and search the label index.

        Returns a ranked list of :class:`LabelSearchResult` (best first).
        """
        if top_k <= 0 or self._faiss.ntotal == 0:
            return []
        if self.needs_compact:
            self.compact()

        query_vec = self._embed([query_text])[0]
        return self.search_vector(query_vec, top_k, conversation_id=conversation_id)

    def search_vector(
        self,
        query_vec: np.ndarray,
        top_k: int,
        *,
        conversation_id: str | None = None,
    ) -> list[LabelSearchResult]:
        """Search using a pre-computed query vector."""
        if top_k <= 0 or self._faiss.ntotal == 0:
            return []

        vec = query_vec.astype(np.float32).reshape(1, -1)
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        oversample = top_k * 4 + len(self._deleted) + 10
        k = min(oversample, self._faiss.ntotal)
        if k <= 0:
            return []

        scores_arr, ids_arr = self._faiss.search(vec, k)
        scores_flat = scores_arr[0]
        ids_flat = ids_arr[0]

        results: list[LabelSearchResult] = []
        rank = 0
        for score, int_id in zip(scores_flat, ids_flat):
            if int_id < 0 or int_id in self._deleted:
                continue
            label_id = self._id_to_label.get(int_id)
            if label_id is None:
                continue
            conv = self._id_to_conv.get(int_id, "")
            if conversation_id is not None and conv != conversation_id:
                continue
            mu_id = self._id_to_mu.get(int_id, "")
            summary = self._id_to_summary.get(int_id, "")
            rank += 1
            results.append(LabelSearchResult(
                label_id=label_id,
                mu_id=mu_id,
                score=float(score),
                rank=rank,
                conversation_id=conv,
                short_summary=summary,
            ))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize the index to disk (two files: .faiss and .meta)."""
        import faiss

        path = Path(path)
        faiss.write_index(self._faiss, str(path.with_suffix(".faiss")))
        meta = {
            "id_to_label": self._id_to_label,
            "label_to_id": self._label_to_id,
            "id_to_mu": self._id_to_mu,
            "id_to_conv": self._id_to_conv,
            "id_to_summary": self._id_to_summary,
            "deleted": self._deleted,
            "next_id": self._next_id,
            "dim": self.dim,
            "normalize": self.normalize,
        }
        with open(path.with_suffix(".meta"), "wb") as f:
            pickle.dump(meta, f)
        logger.debug("CompressedLabelFAISSIndex: saved to {}", path)

    def load(self, path: Path) -> None:
        """Load a previously saved index from disk."""
        import faiss

        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta")
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files not found at {path}")
        self._faiss = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._id_to_label = meta["id_to_label"]
        self._label_to_id = meta["label_to_id"]
        self._id_to_mu = meta["id_to_mu"]
        self._id_to_conv = meta["id_to_conv"]
        self._id_to_summary = meta["id_to_summary"]
        self._deleted = meta["deleted"]
        self._next_id = meta["next_id"]
        self.dim = meta.get("dim", self.dim)
        self.normalize = meta.get("normalize", self.normalize)
        logger.debug("CompressedLabelFAISSIndex: loaded from {}", path)

    # ------------------------------------------------------------------
    # Internal embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embed_fn(texts)
        vecs = np.array(vecs, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms
        return vecs


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = ["CompressedLabelFAISSIndex", "LabelSearchResult"]
