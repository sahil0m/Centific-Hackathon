"""
Tests for the dense retriever — verifies same-conversation-only constraint
using a mock index that doesn't require FAISS or model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from locomo_memory.data.schemas import Chunk, RetrievedChunk
from locomo_memory.retrieval.dense_retriever import DenseRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Returns deterministic random-ish vectors without loading any model."""

    def embed_query(self, query: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(query)) % (2**31))
        v = rng.random(32).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


def _make_chunk(conv_id: str, dia_id: str, turn_idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"{conv_id}_{dia_id}",
        conversation_id=conv_id,
        sample_id=conv_id,
        session_id="S0",
        turn_index_start=turn_idx,
        turn_index_end=turn_idx,
        dia_ids=[dia_id],
        speakers=["Alice"],
        timestamps=["2024-01-01"],
        text=f"[Conversation: {conv_id}] Alice: turn {turn_idx}",
        chunk_strategy="turn",
    )


class MockIndex:
    """Tiny in-memory index that ignores vectors and returns preset results."""

    def __init__(self, conv_chunks: dict[str, list[Chunk]]) -> None:
        self._conv_chunks = conv_chunks

    def search(self, conversation_id: str, query_vec: np.ndarray, top_k: int):
        chunks = self._conv_chunks.get(conversation_id, [])
        return [(c, float(i + 1) / (len(chunks) + 1)) for i, c in enumerate(chunks[:top_k])]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDenseRetriever:
    def _build(self, n_convs: int = 2, n_turns: int = 5, top_k: int = 3):
        conv_chunks = {
            f"conv_{i}": [_make_chunk(f"conv_{i}", f"D{j}", j) for j in range(n_turns)]
            for i in range(n_convs)
        }
        index = MockIndex(conv_chunks)
        embedder = MockEmbedder()
        retriever = DenseRetriever(index=index, embedder=embedder, top_k=top_k)
        return retriever, conv_chunks

    def test_returns_retrieval_result(self):
        retriever, _ = self._build()
        result = retriever.retrieve("conv_0", "q0", "What did Alice say?")
        assert isinstance(result, RetrievalResult)

    def test_same_conversation_only(self):
        retriever, conv_chunks = self._build(n_convs=2)
        result = retriever.retrieve("conv_0", "q0", "test question")
        returned_conv_ids = {c.conversation_id for c in [
            Chunk(**{
                "chunk_id": r.chunk_id,
                "conversation_id": r.chunk_id.split("_")[0] + "_" + r.chunk_id.split("_")[1],
                "sample_id": "x",
                "session_id": r.session_id,
                "turn_index_start": 0,
                "turn_index_end": 0,
                "dia_ids": r.dia_ids,
                "speakers": [r.speaker],
                "timestamps": [""],
                "text": r.text,
                "chunk_strategy": "turn",
            })
            for r in result.retrieved
        ]}
        # All retrieved chunks should come from the queried conversation
        # (mock index enforces this implicitly; verify chunk_ids have conv_0 prefix)
        for r in result.retrieved:
            assert r.chunk_id.startswith("conv_0")

    def test_top_k_respected(self):
        retriever, _ = self._build(n_turns=10, top_k=3)
        result = retriever.retrieve("conv_0", "q0", "hello")
        assert len(result.retrieved) <= 3

    def test_unknown_conversation_returns_empty(self):
        retriever, _ = self._build()
        result = retriever.retrieve("nonexistent_conv", "q0", "hello")
        assert result.retrieved == []

    def test_latency_recorded(self):
        retriever, _ = self._build()
        result = retriever.retrieve("conv_0", "q0", "hello")
        assert result.retrieval_latency_ms >= 0

    def test_batch_retrieve(self):
        retriever, _ = self._build(n_convs=2)
        queries = [
            ("conv_0", "q0", "hello"),
            ("conv_1", "q1", "world"),
        ]
        results = retriever.retrieve_batch(queries)
        assert len(results) == 2
        assert results[0].conversation_id == "conv_0"
        assert results[1].conversation_id == "conv_1"
