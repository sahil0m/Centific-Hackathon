"""
Chunking strategies for LoCoMo conversations.

Strategy A: turn    — one Turn = one Chunk
Strategy B: window3 — sliding window of 3 adjacent turns
Strategy C: session_summary — use session-level summary text if available
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable

from locomo_memory.data.schemas import Chunk, Conversation, Turn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_chunks(
    conversations: list[Conversation],
    strategy: str,
    window_size: int = 3,
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
    """Build chunks for all conversations using the specified strategy."""
    builder = _get_builder(strategy)
    all_chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    for conv in conversations:
        chunks = builder(
            conv,
            window_size=window_size,
            include_speaker=include_speaker,
            include_timestamp=include_timestamp,
            include_session_id=include_session_id,
        )
        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                logger.warning("Duplicate chunk_id detected: %s — skipping", chunk.chunk_id)
                continue
            seen_ids.add(chunk.chunk_id)
            all_chunks.append(chunk)

    logger.info(
        "Built %d chunks from %d conversations using strategy '%s'",
        len(all_chunks),
        len(conversations),
        strategy,
    )
    return all_chunks


def _get_builder(strategy: str) -> Callable:
    strategies: dict[str, Callable] = {
        "turn": _build_turn_chunks,
        "window3": _build_window_chunks,
        "session_summary": _build_session_summary_chunks,
    }
    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Valid options: {list(strategies.keys())}"
        )
    return strategies[strategy]


# ---------------------------------------------------------------------------
# Strategy A: turn
# ---------------------------------------------------------------------------

def _build_turn_chunks(
    conv: Conversation,
    window_size: int = 1,
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for turn in conv.turns:
        if turn.speaker.lower() == "summary":
            continue  # summary turns belong only to session_summary strategy
        text = _format_turn_text(
            conv.conversation_id,
            turn.session_id,
            [turn.dia_id],
            [(turn.speaker, turn.text, turn.timestamp)],
            include_speaker=include_speaker,
            include_timestamp=include_timestamp,
            include_session_id=include_session_id,
        )
        chunk_id = _make_chunk_id(conv.conversation_id, "turn", [turn.dia_id])
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                conversation_id=conv.conversation_id,
                sample_id=conv.sample_id,
                session_id=turn.session_id,
                turn_index_start=turn.turn_index,
                turn_index_end=turn.turn_index,
                dia_ids=[turn.dia_id],
                speakers=[turn.speaker],
                timestamps=[turn.timestamp],
                text=text,
                chunk_strategy="turn",
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategy B: window3 (sliding window)
# ---------------------------------------------------------------------------

def _build_window_chunks(
    conv: Conversation,
    window_size: int = 3,
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
    if window_size < 1:
        window_size = 3
    turns = [t for t in conv.turns if t.speaker.lower() != "summary"]
    chunks: list[Chunk] = []

    for start in range(len(turns)):
        window: list[Turn] = turns[start : start + window_size]
        if not window:
            continue

        dia_ids = [t.dia_id for t in window]
        speakers = list(dict.fromkeys(t.speaker for t in window))
        timestamps = [t.timestamp for t in window]
        session_id = window[0].session_id

        messages = [(t.speaker, t.text, t.timestamp) for t in window]
        text = _format_turn_text(
            conv.conversation_id,
            session_id,
            dia_ids,
            messages,
            include_speaker=include_speaker,
            include_timestamp=include_timestamp,
            include_session_id=include_session_id,
        )
        chunk_id = _make_chunk_id(conv.conversation_id, "window3", dia_ids)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                conversation_id=conv.conversation_id,
                sample_id=conv.sample_id,
                session_id=session_id,
                turn_index_start=window[0].turn_index,
                turn_index_end=window[-1].turn_index,
                dia_ids=dia_ids,
                speakers=speakers,
                timestamps=timestamps,
                text=text,
                chunk_strategy="window3",
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategy C: session_summary
# ---------------------------------------------------------------------------

def _build_session_summary_chunks(
    conv: Conversation,
    window_size: int = 1,
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
    """
    Uses session summaries if present on turns with speaker=='summary' or
    on the conversation object itself. If none found, logs a warning and
    returns empty list for this conversation.
    """
    summary_turns = [t for t in conv.turns if t.speaker.lower() in ("summary", "system_summary")]

    if not summary_turns:
        logger.warning(
            "No session summaries found for conversation '%s'. "
            "session_summary strategy produces no chunks for this conversation.",
            conv.conversation_id,
        )
        return []

    chunks: list[Chunk] = []
    for turn in summary_turns:
        text = _format_turn_text(
            conv.conversation_id,
            turn.session_id,
            [turn.dia_id],
            [(turn.speaker, turn.text, turn.timestamp)],
            include_speaker=include_speaker,
            include_timestamp=include_timestamp,
            include_session_id=include_session_id,
        )
        chunk_id = _make_chunk_id(conv.conversation_id, "session_summary", [turn.dia_id])
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                conversation_id=conv.conversation_id,
                sample_id=conv.sample_id,
                session_id=turn.session_id,
                turn_index_start=turn.turn_index,
                turn_index_end=turn.turn_index,
                dia_ids=[turn.dia_id],
                speakers=[turn.speaker],
                timestamps=[turn.timestamp],
                text=text,
                chunk_strategy="session_summary",
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Text formatting helpers
# ---------------------------------------------------------------------------

def _format_turn_text(
    conversation_id: str,
    session_id: str,
    dia_ids: list[str],
    messages: list[tuple[str, str, str]],
    include_speaker: bool,
    include_timestamp: bool,
    include_session_id: bool,
) -> str:
    dia_str = ",".join(dia_ids)
    header_parts = [f"Conversation: {conversation_id}"]
    if include_session_id:
        header_parts.append(f"Session: {session_id}")
    header_parts.append(f"Dialog IDs: {dia_str}")
    header = "[" + " | ".join(header_parts) + "]"

    lines = [header]
    for speaker, text, timestamp in messages:
        if include_speaker and include_timestamp and timestamp:
            lines.append(f"[{timestamp}] {speaker}: {text}")
        elif include_speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def _make_chunk_id(conversation_id: str, strategy: str, dia_ids: list[str]) -> str:
    key = f"{conversation_id}|{strategy}|{'_'.join(dia_ids)}"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"{conversation_id}__{strategy}__{h}"
