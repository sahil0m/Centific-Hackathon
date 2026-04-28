"""
Chunking strategies for LoCoMo conversations.

Strategy A: turn            — one dialogue turn = one chunk
Strategy B: window3         — sliding window of N adjacent turns
Strategy C: session_summary — one session summary = one chunk

Contextual enrichment (context_window > 0):
  Each turn chunk embeds ±N surrounding turns as [Prior]/[Next] context lines,
  but dia_ids metadata only contains the central turn. This means the embedding
  captures surrounding context for better semantic matching while evidence recall
  is computed correctly against the actual gold turn.
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
    context_window: int = 0,
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
    """Build chunks for all conversations using the specified strategy.

    Args:
        context_window: For 'turn' strategy only. Number of adjacent turns to
            include as [Prior]/[Next] context in the chunk text. Metadata
            (dia_ids) still points only to the central turn.
    """
    builder = _get_builder(strategy)
    all_chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    for conv in conversations:
        chunks = builder(
            conv,
            window_size=window_size,
            context_window=context_window,
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
        "Built %d chunks from %d conversations using strategy='%s' context_window=%d",
        len(all_chunks),
        len(conversations),
        strategy,
        context_window,
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
# Strategy A: turn (with optional contextual enrichment)
# ---------------------------------------------------------------------------

def _build_turn_chunks(
    conv: Conversation,
    window_size: int = 1,
    context_window: int = 0,
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
    # Exclude summary turns — they belong to session_summary strategy only
    dialogue_turns = [t for t in conv.turns if t.speaker.lower() != "summary"]
    chunks: list[Chunk] = []

    for i, turn in enumerate(dialogue_turns):
        if context_window > 0:
            text = _format_turn_with_context(
                conv.conversation_id,
                turn,
                dialogue_turns,
                i,
                context_window,
                include_speaker=include_speaker,
                include_timestamp=include_timestamp,
                include_session_id=include_session_id,
            )
        else:
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
                dia_ids=[turn.dia_id],        # only the central turn
                speakers=[turn.speaker],
                timestamps=[turn.timestamp],
                text=text,
                chunk_strategy="turn",
            )
        )
    return chunks


def _format_turn_with_context(
    conversation_id: str,
    turn: Turn,
    all_turns: list[Turn],
    idx: int,
    context_window: int,
    include_speaker: bool,
    include_timestamp: bool,
    include_session_id: bool,
) -> str:
    """
    Format a turn chunk with ±context_window surrounding turns as context.
    Prior turns are prefixed with [Prior], next turns with [Next].
    The central turn has no prefix so embeddings weight it naturally higher.
    dia_ids in the Chunk metadata only contains the central turn's dia_id.
    """
    header_parts = [f"Conversation: {conversation_id}"]
    if include_session_id:
        header_parts.append(f"Session: {turn.session_id}")
    header_parts.append(f"Dialog IDs: {turn.dia_id}")
    header = "[" + " | ".join(header_parts) + "]"

    lines = [header]

    before = all_turns[max(0, idx - context_window): idx]
    after  = all_turns[idx + 1: idx + 1 + context_window]

    for ctx in before:
        line = _format_line(ctx.speaker, ctx.text, ctx.timestamp, include_speaker, include_timestamp)
        lines.append(f"[Prior] {line}")

    # Central turn — no prefix
    lines.append(_format_line(turn.speaker, turn.text, turn.timestamp, include_speaker, include_timestamp))

    for ctx in after:
        line = _format_line(ctx.speaker, ctx.text, ctx.timestamp, include_speaker, include_timestamp)
        lines.append(f"[Next] {line}")

    return "\n".join(lines)


def _format_line(
    speaker: str, text: str, timestamp: str,
    include_speaker: bool, include_timestamp: bool,
) -> str:
    if include_speaker and include_timestamp and timestamp:
        return f"[{timestamp}] {speaker}: {text}"
    if include_speaker:
        return f"{speaker}: {text}"
    return text


# ---------------------------------------------------------------------------
# Strategy B: window3 (sliding window)
# ---------------------------------------------------------------------------

def _build_window_chunks(
    conv: Conversation,
    window_size: int = 3,
    context_window: int = 0,   # unused for window strategy
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
    if window_size < 1:
        window_size = 3
    turns = [t for t in conv.turns if t.speaker.lower() != "summary"]
    chunks: list[Chunk] = []

    for start in range(len(turns)):
        window: list[Turn] = turns[start: start + window_size]
        if not window:
            continue

        dia_ids    = [t.dia_id for t in window]
        speakers   = list(dict.fromkeys(t.speaker for t in window))
        timestamps = [t.timestamp for t in window]
        session_id = window[0].session_id

        text = _format_turn_text(
            conv.conversation_id,
            session_id,
            dia_ids,
            [(t.speaker, t.text, t.timestamp) for t in window],
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
    context_window: int = 0,   # unused
    include_speaker: bool = True,
    include_timestamp: bool = True,
    include_session_id: bool = True,
) -> list[Chunk]:
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
        lines.append(_format_line(speaker, text, timestamp, include_speaker, include_timestamp))
    return "\n".join(lines)


def _make_chunk_id(conversation_id: str, strategy: str, dia_ids: list[str]) -> str:
    key = f"{conversation_id}|{strategy}|{'_'.join(dia_ids)}"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"{conversation_id}__{strategy}__{h}"
