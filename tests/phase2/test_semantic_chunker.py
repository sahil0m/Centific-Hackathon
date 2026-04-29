"""Tests for the Semantic Chunker.

Uses a deterministic ``FakeEmbedder`` so we can simulate exact similarity
patterns without loading sentence-transformers. Covers: empty / single-turn
input, fully-coherent and fully-divergent sequences, mixed topic boundaries,
both comparison modes, max/min chunk size, summary-turn filtering, chunk
metadata correctness, and validation errors.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from locomo_memory.data.schemas import Chunk, Conversation, Turn
from locomo_memory.phase2.ingestion.semantic_chunker import (
    SemanticChunker,
    TurnEmbedder,
)


# ---------------------------------------------------------------------------
# Deterministic fake embedder
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Returns one fixed vector per text, looked up by exact text match.

    Vectors are L2-normalised on the way in so cosine similarity is just the
    dot product — same contract as ``EmbeddingGenerator`` with
    ``normalize=True``.
    """

    def __init__(self, mapping: dict[str, Iterable[float]]) -> None:
        self._mapping: dict[str, np.ndarray] = {}
        for text, vec in mapping.items():
            arr = np.array(list(vec), dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            self._mapping[text] = arr

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t not in self._mapping:
                raise KeyError(f"FakeEmbedder has no vector for: {t!r}")
            rows.append(self._mapping[t])
        return np.vstack(rows).astype(np.float32)


def _make_turn(
    *,
    conversation_id: str = "conv_1",
    sample_id: str = "sample_1",
    session_id: str = "session_1",
    turn_index: int = 0,
    dia_id: str = "",
    speaker: str = "Speaker",
    text: str = "",
    timestamp: str = "",
) -> Turn:
    return Turn(
        sample_id=sample_id,
        conversation_id=conversation_id,
        session_id=session_id,
        turn_index=turn_index,
        dia_id=dia_id or f"D1:{turn_index}",
        speaker=speaker,
        text=text,
        timestamp=timestamp,
    )


def _conv(turns: list[Turn], conv_id: str = "conv_1") -> Conversation:
    return Conversation(
        conversation_id=conv_id,
        sample_id="sample_1",
        turns=turns,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_default_construction(self) -> None:
        emb = FakeEmbedder({"x": [1.0, 0.0]})
        c = SemanticChunker(emb)
        assert c.similarity_threshold == 0.65
        assert c.comparison_mode == "previous_turn"
        assert c.max_chunk_size is None

    @pytest.mark.parametrize("bad", [-1.5, 1.5, 2.0])
    def test_invalid_threshold(self, bad: float) -> None:
        emb = FakeEmbedder({})
        with pytest.raises(ValueError):
            SemanticChunker(emb, similarity_threshold=bad)

    def test_invalid_comparison_mode(self) -> None:
        emb = FakeEmbedder({})
        with pytest.raises(ValueError):
            SemanticChunker(emb, comparison_mode="bogus")  # type: ignore[arg-type]

    def test_invalid_max_chunk_size(self) -> None:
        emb = FakeEmbedder({})
        with pytest.raises(ValueError):
            SemanticChunker(emb, max_chunk_size=0)

    def test_invalid_min_chunk_size(self) -> None:
        emb = FakeEmbedder({})
        with pytest.raises(ValueError):
            SemanticChunker(emb, min_chunk_size=0)

    def test_min_greater_than_max(self) -> None:
        emb = FakeEmbedder({})
        with pytest.raises(ValueError):
            SemanticChunker(emb, min_chunk_size=5, max_chunk_size=3)

    def test_protocol_typecheck(self) -> None:
        emb = FakeEmbedder({"x": [1.0]})
        assert isinstance(emb, TurnEmbedder)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_conversation(self) -> None:
        chunker = SemanticChunker(FakeEmbedder({}))
        chunks = chunker.chunk_conversation(_conv([]))
        assert chunks == []

    def test_single_turn_skips_embedding(self) -> None:
        # A single turn shouldn't need an embedder at all — but we still
        # have to give it something to satisfy the type. Provide an empty
        # mapping; it must not be queried.
        emb = FakeEmbedder({})
        chunker = SemanticChunker(emb)
        turn = _make_turn(text="solo turn")
        chunks = chunker.chunk_conversation(_conv([turn]))
        assert len(chunks) == 1
        assert chunks[0].turn_index_start == 0
        assert chunks[0].turn_index_end == 0
        assert chunks[0].chunk_strategy == "semantic"

    def test_summary_turns_filtered(self) -> None:
        # Synthetic "summary" speakers (Phase 1 dataloader) must be skipped.
        turns = [
            _make_turn(turn_index=0, speaker="summary", text="session summary text"),
            _make_turn(turn_index=1, text="real turn"),
        ]
        emb = FakeEmbedder({"real turn": [1.0, 0.0]})
        chunker = SemanticChunker(emb)
        chunks = chunker.chunk_conversation(_conv(turns))
        assert len(chunks) == 1
        assert chunks[0].turn_index_start == 1


# ---------------------------------------------------------------------------
# Boundary detection — fully coherent or fully divergent
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_fully_coherent_yields_one_chunk(self) -> None:
        # All turns share the same vector → cos sim = 1.0 ≥ threshold → one chunk
        turns = [_make_turn(turn_index=i, text=f"t{i}") for i in range(4)]
        same_vec = [1.0, 0.0]
        emb = FakeEmbedder({f"t{i}": same_vec for i in range(4)})
        chunker = SemanticChunker(emb, similarity_threshold=0.65)
        chunks = chunker.chunk_conversation(_conv(turns))
        assert len(chunks) == 1
        assert chunks[0].turn_index_start == 0
        assert chunks[0].turn_index_end == 3
        assert chunks[0].dia_ids == ["D1:0", "D1:1", "D1:2", "D1:3"]

    def test_fully_divergent_yields_per_turn_chunks(self) -> None:
        # Each turn orthogonal → cos sim = 0 < threshold → 4 single-turn chunks
        turns = [_make_turn(turn_index=i, text=f"t{i}") for i in range(4)]
        emb = FakeEmbedder({
            "t0": [1.0, 0.0, 0.0, 0.0],
            "t1": [0.0, 1.0, 0.0, 0.0],
            "t2": [0.0, 0.0, 1.0, 0.0],
            "t3": [0.0, 0.0, 0.0, 1.0],
        })
        chunker = SemanticChunker(emb, similarity_threshold=0.65)
        chunks = chunker.chunk_conversation(_conv(turns))
        assert len(chunks) == 4
        for i, ch in enumerate(chunks):
            assert ch.turn_index_start == i
            assert ch.turn_index_end == i

    def test_mixed_topics_split_at_boundary(self) -> None:
        # 3 turns same topic, then 3 turns different topic
        turns = [_make_turn(turn_index=i, text=f"t{i}") for i in range(6)]
        emb = FakeEmbedder({
            "t0": [1.0, 0.0],
            "t1": [1.0, 0.0],
            "t2": [1.0, 0.0],
            "t3": [0.0, 1.0],
            "t4": [0.0, 1.0],
            "t5": [0.0, 1.0],
        })
        chunker = SemanticChunker(emb, similarity_threshold=0.65)
        chunks = chunker.chunk_conversation(_conv(turns))
        assert len(chunks) == 2
        assert chunks[0].turn_index_start == 0
        assert chunks[0].turn_index_end == 2
        assert chunks[1].turn_index_start == 3
        assert chunks[1].turn_index_end == 5


# ---------------------------------------------------------------------------
# Comparison modes
# ---------------------------------------------------------------------------


class TestComparisonModes:
    def test_chunk_centroid_mode_handles_drift(self) -> None:
        # Vectors drift gradually; previous_turn would split at every step,
        # but centroid mode should keep them together.
        # Each step has cos sim ~ 0.92 with the previous; threshold 0.6 → stay.
        turns = [_make_turn(turn_index=i, text=f"t{i}") for i in range(3)]
        emb = FakeEmbedder({
            "t0": [1.0, 0.0, 0.0],
            "t1": [0.92, 0.39, 0.0],   # cos with t0 ≈ 0.92
            "t2": [0.85, 0.53, 0.0],   # cos with t1 ≈ 0.99 — but with t0 only ≈ 0.85
        })
        chunker = SemanticChunker(
            emb, similarity_threshold=0.6, comparison_mode="chunk_centroid",
        )
        chunks = chunker.chunk_conversation(_conv(turns))
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Size constraints
# ---------------------------------------------------------------------------


class TestSizeConstraints:
    def test_max_chunk_size_forces_split(self) -> None:
        turns = [_make_turn(turn_index=i, text=f"t{i}") for i in range(5)]
        same = [1.0, 0.0]
        emb = FakeEmbedder({f"t{i}": same for i in range(5)})
        chunker = SemanticChunker(
            emb, similarity_threshold=0.65, max_chunk_size=2,
        )
        chunks = chunker.chunk_conversation(_conv(turns))
        # 5 turns capped at 2 → chunks of size 2, 2, 1
        assert [len(c.dia_ids) for c in chunks] == [2, 2, 1]

    def test_min_chunk_size_extends_short_chunks(self) -> None:
        # Without a min, every turn would be its own chunk.
        turns = [_make_turn(turn_index=i, text=f"t{i}") for i in range(4)]
        emb = FakeEmbedder({
            "t0": [1.0, 0.0, 0.0, 0.0],
            "t1": [0.0, 1.0, 0.0, 0.0],
            "t2": [0.0, 0.0, 1.0, 0.0],
            "t3": [0.0, 0.0, 0.0, 1.0],
        })
        chunker = SemanticChunker(
            emb, similarity_threshold=0.99, min_chunk_size=2,
        )
        chunks = chunker.chunk_conversation(_conv(turns))
        # min=2 forces every chunk to have at least 2 turns
        assert all(len(c.dia_ids) >= 2 for c in chunks[:-1])  # last chunk may be partial


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_chunk_id_format(self) -> None:
        turns = [
            _make_turn(turn_index=2, text="t0"),
            _make_turn(turn_index=3, text="t1"),
        ]
        emb = FakeEmbedder({"t0": [1.0], "t1": [1.0]})
        chunker = SemanticChunker(emb, similarity_threshold=0.5)
        chunks = chunker.chunk_conversation(_conv(turns))
        assert chunks[0].chunk_id == "conv_1#semantic#2-3"
        assert chunks[0].chunk_strategy == "semantic"

    def test_no_duplicate_chunk_ids_across_conversations(self) -> None:
        turns_a = [_make_turn(turn_index=0, conversation_id="c1", text="t0")]
        turns_b = [_make_turn(turn_index=0, conversation_id="c2", text="t0")]
        emb = FakeEmbedder({"t0": [1.0]})
        chunker = SemanticChunker(emb)
        all_chunks = chunker.chunk_conversations([_conv(turns_a, "c1"), _conv(turns_b, "c2")])
        ids = {c.chunk_id for c in all_chunks}
        assert len(ids) == len(all_chunks)
        assert "c1#semantic#0-0" in ids
        assert "c2#semantic#0-0" in ids

    def test_chunk_dia_ids_collected(self) -> None:
        turns = [
            _make_turn(turn_index=0, dia_id="D1:1", text="t0"),
            _make_turn(turn_index=1, dia_id="D1:2", text="t1"),
        ]
        emb = FakeEmbedder({"t0": [1.0, 0.0], "t1": [1.0, 0.0]})
        chunker = SemanticChunker(emb, similarity_threshold=0.5)
        chunks = chunker.chunk_conversation(_conv(turns))
        assert chunks[0].dia_ids == ["D1:1", "D1:2"]

    def test_chunk_speakers_and_timestamps(self) -> None:
        turns = [
            _make_turn(turn_index=0, speaker="Alice", timestamp="2024-01-01", text="t0"),
            _make_turn(turn_index=1, speaker="Bob", timestamp="2024-01-02", text="t1"),
        ]
        emb = FakeEmbedder({"t0": [1.0], "t1": [1.0]})
        chunker = SemanticChunker(emb, similarity_threshold=0.5)
        chunks = chunker.chunk_conversation(_conv(turns))
        assert chunks[0].speakers == ["Alice", "Bob"]
        assert chunks[0].timestamps == ["2024-01-01", "2024-01-02"]

    def test_chunk_text_format_default(self) -> None:
        turns = [
            _make_turn(turn_index=0, dia_id="D1:1", speaker="Alice",
                       timestamp="2024-01-01", text="hello"),
            _make_turn(turn_index=1, dia_id="D1:2", speaker="Bob",
                       timestamp="2024-01-02", text="world"),
        ]
        emb = FakeEmbedder({"hello": [1.0], "world": [1.0]})
        chunker = SemanticChunker(emb, similarity_threshold=0.5)
        chunks = chunker.chunk_conversation(_conv(turns))
        text = chunks[0].text
        assert "Conversation: conv_1" in text
        assert "Session: session_1" in text
        assert "Dialog IDs: D1:1,D1:2" in text
        assert "Alice" in text and "Bob" in text
        assert "hello" in text and "world" in text

    def test_chunk_text_omits_disabled_fields(self) -> None:
        turns = [
            _make_turn(turn_index=0, speaker="Alice", timestamp="ts", text="hello"),
        ]
        emb = FakeEmbedder({"hello": [1.0]})
        chunker = SemanticChunker(
            emb,
            include_speaker=False,
            include_timestamp=False,
            include_session_id=False,
        )
        chunks = chunker.chunk_conversation(_conv(turns))
        text = chunks[0].text
        assert "Session:" not in text
        # Speaker block should not be present
        assert "Alice:" not in text
        assert "[ts]" not in text
        assert "hello" in text


# ---------------------------------------------------------------------------
# Embedder contract violations
# ---------------------------------------------------------------------------


class TestEmbedderContract:
    def test_wrong_vector_count_raises(self) -> None:
        class BadEmbedder:
            def embed_texts(self, texts: list[str]) -> np.ndarray:
                # Return one fewer vector than requested
                return np.zeros((len(texts) - 1, 2), dtype=np.float32)

        chunker = SemanticChunker(BadEmbedder())
        turns = [_make_turn(turn_index=i, text=f"t{i}") for i in range(3)]
        with pytest.raises(RuntimeError):
            chunker.chunk_conversation(_conv(turns))


# ---------------------------------------------------------------------------
# Multiple conversations (deduplication)
# ---------------------------------------------------------------------------


class TestMultipleConversations:
    def test_chunks_built_per_conversation(self) -> None:
        turns_a = [_make_turn(turn_index=0, conversation_id="a", text="ta0"),
                   _make_turn(turn_index=1, conversation_id="a", text="ta1")]
        turns_b = [_make_turn(turn_index=0, conversation_id="b", text="tb0")]
        emb = FakeEmbedder({
            "ta0": [1.0, 0.0], "ta1": [1.0, 0.0], "tb0": [0.0, 1.0],
        })
        chunker = SemanticChunker(emb, similarity_threshold=0.5)
        all_chunks = chunker.chunk_conversations([_conv(turns_a, "a"), _conv(turns_b, "b")])
        a_chunks = [c for c in all_chunks if c.conversation_id == "a"]
        b_chunks = [c for c in all_chunks if c.conversation_id == "b"]
        assert len(a_chunks) == 1
        assert len(b_chunks) == 1
