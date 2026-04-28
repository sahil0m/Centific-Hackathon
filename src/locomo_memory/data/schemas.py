"""Typed schemas for LoCoMo conversations, chunks, and QA items."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    sample_id: str
    conversation_id: str
    session_id: str
    turn_index: int
    dia_id: str
    speaker: str
    text: str
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class QAItem:
    qa_id: str
    conversation_id: str
    question: str
    answer: str
    category: str = "unknown"
    gold_evidence_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Conversation:
    conversation_id: str
    sample_id: str
    turns: list[Turn] = field(default_factory=list)
    qa_items: list[QAItem] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    conversation_id: str
    sample_id: str
    session_id: str
    turn_index_start: int
    turn_index_end: int
    dia_ids: list[str]
    speakers: list[str]
    timestamps: list[str]
    text: str
    chunk_strategy: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RetrievedChunk:
    chunk_id: str
    dia_ids: list[str]
    session_id: str
    speaker: str
    text: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PredictionRow:
    experiment_name: str
    conversation_id: str
    qa_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    category: str
    gold_evidence_ids: list[str]
    retrieved_chunks: list[dict[str, Any]]
    f1: float
    exact_match: bool
    evidence_recall: float | None
    input_tokens: int
    output_tokens: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    end_to_end_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
