"""
LoCoMo dataset loader.

Handles the actual LoCoMo JSON format:
  {
    "sample_id": "...",
    "conversation": {
      "speaker_a": "Ca",
      "speaker_b": "Me",
      "session_1_date_time": "1:56 pm on 8 May, 2023",
      "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "..."}, ...],
      "session_2_date_time": "...",
      "session_2": [...],
      ...
    },
    "session_summary": {
      "session_1_summary": "Caroline and Melanie had ...",
      ...
    },
    "qa": [
      {"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 1},
      {"question": "...", "adversarial_answer": "...", "evidence": ["D2:3"], "category": 5},
    ]
  }

Also handles the generic list/dict formats as a fallback for other LoCoMo variants.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from locomo_memory.data.schemas import Conversation, QAItem, Turn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_locomo(path: str | Path) -> list[Conversation]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at '{path}'.\n"
            "Place locomo10.json in data/raw/ or update the config path.\n"
            "See scripts/download_locomo.sh for download instructions."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    conversations = _parse_raw(raw)

    logger.info(
        "Loaded %d conversations, %d total turns, %d total QA items",
        len(conversations),
        sum(len(c.turns) for c in conversations),
        sum(len(c.qa_items) for c in conversations),
    )
    _log_statistics(conversations)
    return conversations


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def _parse_raw(raw: Any) -> list[Conversation]:
    if isinstance(raw, list):
        return _parse_list(raw)
    if isinstance(raw, dict):
        if "data" in raw:
            return _parse_list(raw["data"])
        return _parse_dict_of_conversations(raw)
    raise ValueError(f"Unexpected top-level JSON type: {type(raw)}")


def _parse_list(items: list[Any]) -> list[Conversation]:
    conversations: list[Conversation] = []
    for idx, item in enumerate(items):
        try:
            conv = _parse_single(item, fallback_index=idx)
            conversations.append(conv)
        except Exception as exc:
            logger.warning("Skipping item %d due to parse error: %s", idx, exc)
    return conversations


def _parse_dict_of_conversations(raw: dict[str, Any]) -> list[Conversation]:
    conversations: list[Conversation] = []
    for key, value in raw.items():
        try:
            if isinstance(value, dict):
                value.setdefault("conversation_id", key)
                value.setdefault("sample_id", key)
                conv = _parse_single(value, fallback_index=key)
                conversations.append(conv)
        except Exception as exc:
            logger.warning("Skipping key '%s' due to parse error: %s", key, exc)
    return conversations


def _parse_single(item: dict[str, Any], fallback_index: Any = 0) -> Conversation:
    conv_id = str(
        item.get("conversation_id")
        or item.get("conv_id")
        or item.get("id")
        or item.get("sample_id")
        or f"conv_{fallback_index}"
    )
    sample_id = str(item.get("sample_id") or item.get("id") or conv_id)

    turns = _extract_turns(item, conv_id, sample_id)
    qa_items = _extract_qa(item, conv_id)

    return Conversation(
        conversation_id=conv_id,
        sample_id=sample_id,
        turns=turns,
        qa_items=qa_items,
    )


# ---------------------------------------------------------------------------
# Turn extraction — handles both LoCoMo-specific and generic formats
# ---------------------------------------------------------------------------

def _extract_turns(
    item: dict[str, Any], conv_id: str, sample_id: str
) -> list[Turn]:
    conv_block = item.get("conversation") or item.get("sessions") or item.get("turns") or []

    # LoCoMo-specific: conversation is a flat dict with session_N keys
    if isinstance(conv_block, dict) and _is_locomo_session_dict(conv_block):
        return _extract_turns_locomo(item, conv_block, conv_id, sample_id)

    # Generic fallback: list of session objects or list of turn dicts
    return _extract_turns_generic(conv_block, conv_id, sample_id)


def _is_locomo_session_dict(conv: dict[str, Any]) -> bool:
    """True if the conversation dict uses the LoCoMo session_N key pattern."""
    return any(re.match(r"^session_\d+$", k) for k in conv)


def _extract_turns_locomo(
    item: dict[str, Any],
    conv: dict[str, Any],
    conv_id: str,
    sample_id: str,
) -> list[Turn]:
    """
    Parse the LoCoMo-specific format where sessions are stored as:
      conv["session_1"] = [{"speaker": ..., "dia_id": ..., "text": ...}, ...]
      conv["session_1_date_time"] = "1:56 pm on 8 May, 2023"

    Also appends synthetic "summary" turns from session_summary when present,
    so the session_summary chunking strategy can find them.
    """
    turns: list[Turn] = []
    global_turn_idx = 0

    session_summaries: dict[str, str] = {}
    raw_summaries = item.get("session_summary", {})
    if isinstance(raw_summaries, dict):
        for k, v in raw_summaries.items():
            m = re.match(r"^session_(\d+)_summary$", k)
            if m and isinstance(v, str):
                session_summaries[f"session_{m.group(1)}"] = v

    # Collect session keys in numeric order
    session_keys = sorted(
        (k for k in conv if re.match(r"^session_\d+$", k)),
        key=lambda k: int(re.search(r"\d+", k).group()),
    )

    for sess_key in session_keys:
        messages = conv[sess_key]
        if not isinstance(messages, list):
            continue

        session_id = sess_key  # e.g. "session_1"
        timestamp = str(conv.get(f"{sess_key}_date_time", ""))

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            speaker = str(msg.get("speaker") or msg.get("role") or "unknown")
            text = str(msg.get("text") or msg.get("content") or "")
            dia_id = str(
                msg.get("dia_id")
                or msg.get("dialog_id")
                or msg.get("id")
                or f"D{global_turn_idx}"
            )
            turns.append(
                Turn(
                    sample_id=sample_id,
                    conversation_id=conv_id,
                    session_id=session_id,
                    turn_index=global_turn_idx,
                    dia_id=dia_id,
                    speaker=speaker,
                    text=text,
                    timestamp=timestamp,
                )
            )
            global_turn_idx += 1

        # Append summary turn so session_summary chunker can find it
        if sess_key in session_summaries:
            turns.append(
                Turn(
                    sample_id=sample_id,
                    conversation_id=conv_id,
                    session_id=session_id,
                    turn_index=global_turn_idx,
                    dia_id=f"{sess_key}_summary",
                    speaker="summary",
                    text=session_summaries[sess_key],
                    timestamp=timestamp,
                )
            )
            global_turn_idx += 1

    return turns


def _extract_turns_generic(
    conv_block: Any,
    conv_id: str,
    sample_id: str,
) -> list[Turn]:
    """Fallback for list-of-sessions or list-of-turns formats."""
    turns: list[Turn] = []
    global_turn_idx = 0

    if isinstance(conv_block, dict):
        conv_block = list(conv_block.values())

    for session_raw in conv_block:
        session_id = str(
            session_raw.get("session_id")
            or session_raw.get("session")
            or session_raw.get("id")
            or "S0"
        ) if isinstance(session_raw, dict) else "S0"

        messages_raw = (
            session_raw.get("conversation")
            or session_raw.get("messages")
            or session_raw.get("turns")
            or session_raw.get("dialog")
            or []
        ) if isinstance(session_raw, dict) else (
            session_raw if isinstance(session_raw, list) else []
        )

        for msg in messages_raw:
            if not isinstance(msg, dict):
                continue
            speaker = str(msg.get("speaker") or msg.get("role") or msg.get("name") or "unknown")
            text = str(msg.get("text") or msg.get("content") or msg.get("utterance") or "")
            dia_id = str(
                msg.get("dia_id")
                or msg.get("dialog_id")
                or msg.get("id")
                or f"D{global_turn_idx}"
            )
            timestamp = str(msg.get("timestamp") or msg.get("date") or "")
            turns.append(
                Turn(
                    sample_id=sample_id,
                    conversation_id=conv_id,
                    session_id=session_id,
                    turn_index=global_turn_idx,
                    dia_id=dia_id,
                    speaker=speaker,
                    text=text,
                    timestamp=timestamp,
                )
            )
            global_turn_idx += 1

    return turns


# ---------------------------------------------------------------------------
# QA extraction
# ---------------------------------------------------------------------------

def _extract_qa(item: dict[str, Any], conv_id: str) -> list[QAItem]:
    qa_items: list[QAItem] = []
    raw_qa = item.get("qa") or item.get("questions") or item.get("qa_pairs") or []

    if isinstance(raw_qa, dict):
        raw_qa = list(raw_qa.values())

    for idx, qa in enumerate(raw_qa):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or qa.get("q") or "")
        if not question:
            continue

        # Category 5 uses "adversarial_answer"; all others use "answer"
        answer_raw = (
            qa.get("answer")
            or qa.get("adversarial_answer")
            or qa.get("answers")
            or qa.get("a")
            or ""
        )
        if isinstance(answer_raw, list):
            answer = " ".join(str(a) for a in answer_raw)
        else:
            answer = str(answer_raw)

        qa_id = str(qa.get("qa_id") or qa.get("id") or f"{conv_id}_qa_{idx}")
        category = str(qa.get("category") or qa.get("type") or "unknown")

        evidence_raw = (
            qa.get("evidence")
            or qa.get("gold_evidence")
            or qa.get("evidence_ids")
            or []
        )
        if isinstance(evidence_raw, list):
            gold_ids = _split_evidence_ids(evidence_raw)
        elif isinstance(evidence_raw, str):
            gold_ids = _split_evidence_ids([evidence_raw]) if evidence_raw else []
        else:
            gold_ids = []

        qa_items.append(
            QAItem(
                qa_id=qa_id,
                conversation_id=conv_id,
                question=question,
                answer=answer,
                category=category,
                gold_evidence_ids=gold_ids,
            )
        )

    return qa_items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_evidence_ids(raw_ids: list[Any]) -> list[str]:
    """
    Normalize gold evidence IDs. Some LoCoMo entries pack multiple IDs into
    a single string separated by '; ' (e.g. 'D8:6; D9:17'). Split those out.
    """
    result: list[str] = []
    for item in raw_ids:
        if not item:
            continue
        s = str(item)
        # Split on semicolons (with optional whitespace)
        parts = [p.strip() for p in re.split(r";\s*", s) if p.strip()]
        result.extend(parts)
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _log_statistics(conversations: list[Conversation]) -> None:
    total_turns = sum(len(c.turns) for c in conversations)
    total_qa = sum(len(c.qa_items) for c in conversations)
    categories: dict[str, int] = {}
    for conv in conversations:
        for qa in conv.qa_items:
            categories[qa.category] = categories.get(qa.category, 0) + 1

    logger.info("Dataset statistics:")
    logger.info("  Conversations : %d", len(conversations))
    logger.info("  Total turns   : %d", total_turns)
    logger.info("  Total QA items: %d", total_qa)
    logger.info("  QA categories : %s", dict(sorted(categories.items())))


# ---------------------------------------------------------------------------
# Synthetic fixture for tests
# ---------------------------------------------------------------------------

def make_synthetic_locomo(
    n_conversations: int = 2,
    turns_per_session: int = 4,
    sessions_per_conv: int = 2,
    qa_per_conv: int = 3,
) -> list[Conversation]:
    """Return a minimal synthetic LoCoMo dataset for unit tests."""
    conversations: list[Conversation] = []
    for ci in range(n_conversations):
        conv_id = f"conv_{ci}"
        turns: list[Turn] = []
        turn_idx = 0
        for si in range(sessions_per_conv):
            session_id = f"session_{si + 1}"
            speakers = ["Alice", "Bob"]
            for ti in range(turns_per_session):
                turns.append(
                    Turn(
                        sample_id=conv_id,
                        conversation_id=conv_id,
                        session_id=session_id,
                        turn_index=turn_idx,
                        dia_id=f"D{si + 1}:{ti + 1}",
                        speaker=speakers[ti % 2],
                        text=f"Conv{ci} S{si} T{ti}: Hello from {speakers[ti % 2]}.",
                        timestamp=f"2024-01-0{si + 1}T10:00:00",
                    )
                )
                turn_idx += 1

        qa_items: list[QAItem] = []
        for qi in range(qa_per_conv):
            qa_items.append(
                QAItem(
                    qa_id=f"{conv_id}_qa_{qi}",
                    conversation_id=conv_id,
                    question=f"What did Alice say in session {qi % sessions_per_conv + 1}?",
                    answer=f"Conv{ci} S{qi % sessions_per_conv} T0: Hello from Alice.",
                    category=["single_hop", "multi_hop", "temporal"][qi % 3],
                    gold_evidence_ids=[f"D{qi % sessions_per_conv + 1}:1"],
                )
            )

        conversations.append(
            Conversation(
                conversation_id=conv_id,
                sample_id=conv_id,
                turns=turns,
                qa_items=qa_items,
            )
        )
    return conversations
