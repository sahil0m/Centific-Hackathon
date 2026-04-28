"""Tests for Evidence Recall@k."""

import pytest

from locomo_memory.data.schemas import RetrievedChunk
from locomo_memory.evaluation.evidence_recall import (
    evidence_recall_at_k,
    compute_mean_evidence_recall,
)


def _make_chunk(dia_ids: list[str], score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{'_'.join(dia_ids)}",
        dia_ids=dia_ids,
        session_id="S0",
        speaker="Alice",
        text="dummy text",
        score=score,
    )


class TestEvidenceRecallAtK:
    def test_perfect_recall(self):
        gold = ["D0", "D1"]
        chunks = [_make_chunk(["D0"]), _make_chunk(["D1"])]
        assert evidence_recall_at_k(gold, chunks) == 1.0

    def test_partial_recall(self):
        gold = ["D0", "D1"]
        chunks = [_make_chunk(["D0"])]
        assert evidence_recall_at_k(gold, chunks) == 0.5

    def test_zero_recall(self):
        gold = ["D0", "D1"]
        chunks = [_make_chunk(["D99"])]
        assert evidence_recall_at_k(gold, chunks) == 0.0

    def test_no_gold_returns_none(self):
        chunks = [_make_chunk(["D0"])]
        assert evidence_recall_at_k([], chunks) is None

    def test_empty_chunks(self):
        gold = ["D0"]
        assert evidence_recall_at_k(gold, []) == 0.0

    def test_multi_dia_id_chunk(self):
        # A window chunk covering D0,D1,D2 should satisfy gold evidence D1
        gold = ["D1"]
        chunks = [_make_chunk(["D0", "D1", "D2"])]
        assert evidence_recall_at_k(gold, chunks) == 1.0

    def test_duplicate_gold_ids(self):
        # Even if gold has duplicates, recall should not exceed 1.0
        gold = ["D0", "D0"]
        chunks = [_make_chunk(["D0"])]
        # hits=2, len(gold)=2 → 1.0
        assert evidence_recall_at_k(gold, chunks) == 1.0


class TestMeanEvidenceRecall:
    def test_all_none(self):
        assert compute_mean_evidence_recall([None, None]) is None

    def test_mixed(self):
        result = compute_mean_evidence_recall([1.0, 0.5, None])
        assert result == pytest.approx(0.75, abs=1e-4)

    def test_all_values(self):
        result = compute_mean_evidence_recall([0.5, 0.5])
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_empty(self):
        assert compute_mean_evidence_recall([]) is None
