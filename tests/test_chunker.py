"""Tests for chunking strategies — no LLM calls."""

import pytest

from locomo_memory.data.load_locomo import make_synthetic_locomo
from locomo_memory.data.schemas import Chunk
from locomo_memory.indexing.chunkers import build_chunks


def _get_convs(n_turns_per_session=4, sessions=2):
    return make_synthetic_locomo(n_conversations=2, turns_per_session=n_turns_per_session, sessions_per_conv=sessions)


# ---------------------------------------------------------------------------
# Turn strategy
# ---------------------------------------------------------------------------

class TestTurnChunker:
    def test_count(self):
        convs = _get_convs()
        chunks = build_chunks(convs, strategy="turn")
        # 2 convs × 2 sessions × 4 turns = 16
        assert len(chunks) == 16

    def test_chunk_fields(self):
        convs = _get_convs()
        chunks = build_chunks(convs, strategy="turn")
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_strategy == "turn"
            assert c.chunk_id
            assert c.conversation_id
            assert c.session_id
            assert len(c.dia_ids) == 1
            assert c.text

    def test_no_duplicate_ids(self):
        convs = _get_convs()
        chunks = build_chunks(convs, strategy="turn")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_turn_index_start_equals_end(self):
        convs = _get_convs()
        chunks = build_chunks(convs, strategy="turn")
        for c in chunks:
            assert c.turn_index_start == c.turn_index_end

    def test_conversation_isolation(self):
        convs = _get_convs()
        chunks = build_chunks(convs, strategy="turn")
        for c in chunks:
            assert c.conversation_id in ("conv_0", "conv_1")
        conv0_chunks = [c for c in chunks if c.conversation_id == "conv_0"]
        conv1_chunks = [c for c in chunks if c.conversation_id == "conv_1"]
        assert len(conv0_chunks) == 8
        assert len(conv1_chunks) == 8


# ---------------------------------------------------------------------------
# Window3 strategy
# ---------------------------------------------------------------------------

class TestWindow3Chunker:
    def test_count(self):
        convs = make_synthetic_locomo(n_conversations=1, turns_per_session=5, sessions_per_conv=1)
        chunks = build_chunks(convs, strategy="window3", window_size=3)
        # sliding window of size 3 over 5 turns → 5 chunks (starts at 0,1,2,3,4)
        assert len(chunks) == 5

    def test_window_dia_ids(self):
        convs = make_synthetic_locomo(n_conversations=1, turns_per_session=5, sessions_per_conv=1)
        chunks = build_chunks(convs, strategy="window3", window_size=3)
        # First chunk covers turns 0,1,2 → 3 dia_ids
        assert len(chunks[0].dia_ids) == 3
        # Last chunk may have fewer if turns run out
        assert len(chunks[-1].dia_ids) >= 1

    def test_chunk_fields(self):
        convs = _get_convs()
        chunks = build_chunks(convs, strategy="window3", window_size=3)
        for c in chunks:
            assert c.chunk_strategy == "window3"
            assert len(c.dia_ids) >= 1
            assert c.text

    def test_no_duplicate_ids(self):
        convs = _get_convs()
        chunks = build_chunks(convs, strategy="window3", window_size=3)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_window_size_1_equals_turn(self):
        convs = make_synthetic_locomo(n_conversations=1, turns_per_session=3, sessions_per_conv=1)
        turn_chunks = build_chunks(convs, strategy="turn")
        w1_chunks = build_chunks(convs, strategy="window3", window_size=1)
        assert len(turn_chunks) == len(w1_chunks)


# ---------------------------------------------------------------------------
# Session summary strategy — graceful when no summaries
# ---------------------------------------------------------------------------

class TestSessionSummaryChunker:
    def test_no_summaries_produces_no_chunks(self):
        convs = _get_convs()
        # synthetic data has no 'summary' speaker → should produce 0 chunks gracefully
        chunks = build_chunks(convs, strategy="session_summary")
        assert len(chunks) == 0

    def test_invalid_strategy_raises(self):
        convs = _get_convs()
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            build_chunks(convs, strategy="bad_strategy")
