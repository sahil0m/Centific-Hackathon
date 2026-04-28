"""Tests for the LoCoMo data loader — no external files or LLM calls needed."""

import json
import tempfile
from pathlib import Path

import pytest

from locomo_memory.data.load_locomo import load_locomo, make_synthetic_locomo
from locomo_memory.data.schemas import Conversation, QAItem, Turn


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def test_synthetic_dataset_structure():
    convs = make_synthetic_locomo(n_conversations=2, turns_per_session=4, sessions_per_conv=2)
    assert len(convs) == 2
    for conv in convs:
        assert isinstance(conv, Conversation)
        assert len(conv.turns) == 8  # 4 turns × 2 sessions
        assert len(conv.qa_items) == 3


def test_synthetic_turn_fields():
    convs = make_synthetic_locomo(n_conversations=1)
    turn = convs[0].turns[0]
    assert isinstance(turn, Turn)
    assert turn.conversation_id
    assert turn.session_id
    assert turn.speaker in ("Alice", "Bob")
    assert turn.text
    assert isinstance(turn.turn_index, int)


def test_synthetic_qa_fields():
    convs = make_synthetic_locomo(n_conversations=1)
    qa = convs[0].qa_items[0]
    assert isinstance(qa, QAItem)
    assert qa.question
    assert qa.answer
    assert qa.category in ("single_hop", "multi_hop", "temporal")


# ---------------------------------------------------------------------------
# JSON loader — list format
# ---------------------------------------------------------------------------

def _make_locomo_json_list(n: int = 2) -> list[dict]:
    items = []
    for i in range(n):
        items.append(
            {
                "conversation_id": f"c{i}",
                "sample_id": f"c{i}",
                "conversation": [
                    {
                        "session_id": "S0",
                        "conversation": [
                            {"dia_id": "D0", "speaker": "Alice", "text": "Hello", "timestamp": "2024-01-01"},
                            {"dia_id": "D1", "speaker": "Bob", "text": "Hi", "timestamp": "2024-01-01"},
                        ],
                    }
                ],
                "qa": [
                    {
                        "qa_id": f"c{i}_q0",
                        "question": "What did Alice say?",
                        "answer": "Hello",
                        "category": "single_hop",
                        "evidence": ["D0"],
                    }
                ],
            }
        )
    return items


def test_load_from_list_json(tmp_path):
    data = _make_locomo_json_list(2)
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps(data))
    convs = load_locomo(str(p))
    assert len(convs) == 2
    assert all(len(c.turns) == 2 for c in convs)
    assert all(len(c.qa_items) == 1 for c in convs)


def test_load_from_data_wrapper(tmp_path):
    data = {"data": _make_locomo_json_list(3)}
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps(data))
    convs = load_locomo(str(p))
    assert len(convs) == 3


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="LoCoMo dataset not found"):
        load_locomo("/nonexistent/path/locomo10.json")


def test_missing_optional_fields(tmp_path):
    # Minimal item with no timestamp, no evidence
    data = [
        {
            "conversation_id": "c0",
            "conversation": [
                {
                    "session_id": "S0",
                    "conversation": [
                        {"speaker": "Alice", "text": "Hello"},
                    ],
                }
            ],
            "qa": [
                {"question": "What?", "answer": "Hello", "category": "single_hop"}
            ],
        }
    ]
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps(data))
    convs = load_locomo(str(p))
    assert len(convs) == 1
    assert convs[0].turns[0].timestamp == ""
    assert convs[0].qa_items[0].gold_evidence_ids == []


def test_no_qa_items(tmp_path):
    data = [
        {
            "conversation_id": "c0",
            "conversation": [
                {
                    "session_id": "S0",
                    "conversation": [{"speaker": "Alice", "text": "Hello"}],
                }
            ],
        }
    ]
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps(data))
    convs = load_locomo(str(p))
    assert len(convs) == 1
    assert convs[0].qa_items == []
