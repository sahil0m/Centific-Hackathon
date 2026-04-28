"""
FAISS-based vector index abstraction.

Designed so it can be replaced with Qdrant / Milvus / pgvector later —
the interface is: build, search, persist, load.

Index is keyed per conversation so retrieval is always scoped to a
single conversation (same_conversation_only constraint).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from locomo_memory.data.schemas import Chunk

logger = logging.getLogger(__name__)


class ConversationIndex:
    """FAISS index for one conversation."""

    def __init__(self, conversation_id: str, dim: int) -> None:
        self.conversation_id = conversation_id
        self.dim = dim
        self._index = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        import faiss  # type: ignore

        assert len(chunks) == embeddings.shape[0], "chunks and embeddings length mismatch"
        self._chunks = chunks
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(embeddings.astype(np.float32))
        logger.debug(
            "Built FAISS index for conversation '%s': %d vectors",
            self.conversation_id,
            len(chunks),
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        import faiss  # type: ignore

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.debug("Saved index for conversation '%s' to %s", self.conversation_id, path)

    @classmethod
    def load(cls, conversation_id: str, path: Path) -> "ConversationIndex":
        import faiss  # type: ignore

        index_path = path / "index.faiss"
        chunks_path = path / "chunks.pkl"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Index files not found in {path}")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        faiss_index = faiss.read_index(str(index_path))
        inst = cls(conversation_id=conversation_id, dim=faiss_index.d)
        inst._index = faiss_index
        inst._chunks = chunks
        logger.debug("Loaded index for conversation '%s' from %s", conversation_id, path)
        return inst


class MultiConversationIndex:
    """
    Manages one ConversationIndex per conversation.

    All retrieval is restricted to the queried conversation_id.
    """

    def __init__(self, index_dir: Path | None = None) -> None:
        self._indices: dict[str, ConversationIndex] = {}
        self.index_dir = Path(index_dir) if index_dir else None

    def build_all(
        self,
        chunks_by_conv: dict[str, list[Chunk]],
        embeddings_by_conv: dict[str, np.ndarray],
        dim: int,
    ) -> None:
        for conv_id, chunks in chunks_by_conv.items():
            if not chunks:
                continue
            idx = ConversationIndex(conversation_id=conv_id, dim=dim)
            idx.build(chunks, embeddings_by_conv[conv_id])
            self._indices[conv_id] = idx

        if self.index_dir:
            self.save()

        logger.info(
            "Built indices for %d conversations", len(self._indices)
        )

    def search(
        self, conversation_id: str, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[Chunk, float]]:
        idx = self._indices.get(conversation_id)
        if idx is None:
            logger.warning("No index found for conversation '%s'", conversation_id)
            return []
        return idx.search(query_vec, top_k)

    def save(self) -> None:
        if self.index_dir is None:
            return
        for conv_id, idx in self._indices.items():
            safe_id = conv_id.replace("/", "_").replace("\\", "_")
            idx.save(self.index_dir / safe_id)
        logger.info("Saved all indices to %s", self.index_dir)

    def load(self, dim: int) -> None:
        if self.index_dir is None or not self.index_dir.exists():
            return
        for sub in self.index_dir.iterdir():
            if sub.is_dir():
                try:
                    idx = ConversationIndex.load(sub.name, sub)
                    self._indices[sub.name] = idx
                except Exception as exc:
                    logger.warning("Could not load index from %s: %s", sub, exc)
        logger.info("Loaded %d conversation indices from %s", len(self._indices), self.index_dir)

    @property
    def conversation_ids(self) -> list[str]:
        return list(self._indices.keys())
