"""Tests for Phase 2 failure analyzer — Milestone 11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locomo_memory.phase2.analysis.failure_analyzer import (
    FailureClassification,
    Phase2FailureAnalyzer,
    _is_temporal,
)
from locomo_memory.phase2.analysis.report_writer import save_analysis


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_row(
    *,
    qa_id: str = "qa1",
    conv_id: str = "conv1",
    question: str = "What did Alice say?",
    gold_answer: str = "Hello",
    category: int = 1,
    gold_evidence_ids: list[str] | None = None,
    retrieved_dia_ids: list[list[str]] | None = None,
    retrieved_claims: list[str] | None = None,
    evidence_recall: float | None = None,
) -> dict:
    return {
        "qa_id": qa_id,
        "conversation_id": conv_id,
        "question": question,
        "gold_answer": gold_answer,
        "category": category,
        "gold_evidence_ids": gold_evidence_ids or [],
        "retrieved_dia_ids": retrieved_dia_ids or [],
        "retrieved_claims": retrieved_claims or [],
        "evidence_recall": evidence_recall,
    }


# ---------------------------------------------------------------------------
# FailureClassification
# ---------------------------------------------------------------------------


class TestFailureClassification:
    def test_primary_extraction_miss(self):
        clf = FailureClassification(extraction_miss=True)
        assert clf.primary_type() == "extraction_miss"

    def test_primary_ranking_depth_over_retrieval(self):
        clf = FailureClassification(retrieval_miss=True, ranking_depth_issue=True)
        assert clf.primary_type() == "ranking_depth_issue"

    def test_primary_retrieval_miss(self):
        clf = FailureClassification(retrieval_miss=True)
        assert clf.primary_type() == "retrieval_miss"

    def test_primary_provenance(self):
        clf = FailureClassification(provenance_mapping_issue=True)
        assert clf.primary_type() == "provenance_mapping_issue"

    def test_primary_temporal(self):
        clf = FailureClassification(temporal_issue=True)
        assert clf.primary_type() == "temporal_issue"

    def test_primary_unknown(self):
        clf = FailureClassification()
        assert clf.primary_type() == "unknown"

    def test_as_dict_has_primary_type(self):
        clf = FailureClassification(retrieval_miss=True)
        d = clf.as_dict()
        assert d["primary_type"] == "retrieval_miss"
        assert d["retrieval_miss"] is True
        assert d["extraction_miss"] is False


# ---------------------------------------------------------------------------
# Temporal detection
# ---------------------------------------------------------------------------


class TestTemporalDetection:
    @pytest.mark.parametrize("q", [
        "When did Alice join?",
        "What happened before the party?",
        "How long did they stay?",
        "In which year did Bob leave?",
        "What was the sequence of events?",
        "Who came first?",
    ])
    def test_temporal_detected(self, q):
        assert _is_temporal(q)

    @pytest.mark.parametrize("q", [
        "What did Alice say?",
        "Who is Bob's friend?",
        "What job does Carol have?",
    ])
    def test_not_temporal(self, q):
        assert not _is_temporal(q)


# ---------------------------------------------------------------------------
# Extraction miss detection
# ---------------------------------------------------------------------------


class TestExtractionMiss:
    def test_extraction_miss_when_store_has_no_gold(self, tmp_path):
        """Gold dia_id not in store → extraction_miss."""
        # Build a real SQLite store with MUs that don't cover the gold dia_id
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.schemas import MemoryUnit, MemoryStatus

        db_path = tmp_path / "conv1.db"
        store = MemoryStore(db_path)
        mu = MemoryUnit(
            conversation_id="conv1",
            session_id="s1",
            claim="Alice said hello",
            original_text="hello",
            source_dia_ids=["D10", "D11"],
        )
        store.insert_memory_unit(mu)

        row = _make_row(
            conv_id="conv1",
            gold_evidence_ids=["D99"],  # D99 not in store
            retrieved_dia_ids=[["D10"]],
            evidence_recall=0.0,
        )

        analyzer = Phase2FailureAnalyzer(db_dir=tmp_path)
        report = analyzer.analyze([row])

        assert report.n_zero_recall == 1
        failures = report.failures
        assert len(failures) == 1
        assert failures[0].classification.extraction_miss is True
        assert failures[0].classification.retrieval_miss is False

    def test_extraction_miss_in_failure_type_counts(self, tmp_path):
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.schemas import MemoryUnit

        db_path = tmp_path / "conv1.db"
        store = MemoryStore(db_path)
        mu = MemoryUnit(conversation_id="conv1", session_id="s1",
                        claim="X", original_text="X", source_dia_ids=["D1"])
        store.insert_memory_unit(mu)

        rows = [_make_row(conv_id="conv1", gold_evidence_ids=["D99"],
                          retrieved_dia_ids=[], evidence_recall=0.0)]
        analyzer = Phase2FailureAnalyzer(db_dir=tmp_path)
        report = analyzer.analyze(rows)

        assert "extraction_miss" in report.failure_type_counts


# ---------------------------------------------------------------------------
# Retrieval miss detection
# ---------------------------------------------------------------------------


class TestRetrievalMiss:
    def test_retrieval_miss_when_gold_in_store_but_not_retrieved(self, tmp_path):
        """Gold dia_id IS in the store but not in retrieved hits → retrieval_miss."""
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.schemas import MemoryUnit

        db_path = tmp_path / "conv1.db"
        store = MemoryStore(db_path)
        mu = MemoryUnit(conversation_id="conv1", session_id="s1",
                        claim="Alice said hello", original_text="hello",
                        source_dia_ids=["D42"])
        store.insert_memory_unit(mu)

        row = _make_row(
            conv_id="conv1",
            gold_evidence_ids=["D42"],  # D42 IS in store
            retrieved_dia_ids=[["D10", "D11"]],  # but not retrieved
            evidence_recall=0.0,
        )

        analyzer = Phase2FailureAnalyzer(db_dir=tmp_path)
        report = analyzer.analyze([row])

        assert len(report.failures) == 1
        clf = report.failures[0].classification
        assert clf.retrieval_miss is True
        assert clf.extraction_miss is False

    def test_retrieval_miss_without_store(self):
        """Without a db_dir, gold-not-retrieved is classified retrieval_miss by default."""
        row = _make_row(
            gold_evidence_ids=["D42"],
            retrieved_dia_ids=[["D10"]],
            evidence_recall=0.0,
        )
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([row])
        assert report.failures[0].classification.retrieval_miss is True
        assert report.failures[0].classification.extraction_miss is False


# ---------------------------------------------------------------------------
# Ranking depth issue detection
# ---------------------------------------------------------------------------


class TestRankingDepthIssue:
    def test_ranking_depth_detected_from_top10(self):
        """Gold in top-10 but not top-5 → ranking_depth_issue."""
        top5_row = _make_row(
            qa_id="qa1",
            gold_evidence_ids=["D42"],
            retrieved_dia_ids=[["D1"], ["D2"], ["D3"], ["D4"], ["D5"]],
            evidence_recall=0.0,
        )
        top10_row = _make_row(
            qa_id="qa1",
            gold_evidence_ids=["D42"],
            retrieved_dia_ids=[["D1"], ["D2"], ["D3"], ["D4"], ["D5"],
                                ["D6"], ["D7"], ["D42"], ["D9"], ["D10"]],
            evidence_recall=1.0,
        )

        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([top5_row], top10_predictions=[top10_row])

        assert len(report.failures) == 1
        clf = report.failures[0].classification
        assert clf.ranking_depth_issue is True
        assert report.failure_type_counts.get("ranking_depth_issue", 0) >= 1

    def test_no_ranking_depth_when_not_in_top10(self):
        """Gold not in top-10 either → not ranking_depth_issue."""
        top5_row = _make_row(
            qa_id="qa1",
            gold_evidence_ids=["D42"],
            retrieved_dia_ids=[["D1"]],
            evidence_recall=0.0,
        )
        top10_row = _make_row(
            qa_id="qa1",
            gold_evidence_ids=["D42"],
            retrieved_dia_ids=[["D1"], ["D2"], ["D3"]],
            evidence_recall=0.0,
        )

        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([top5_row], top10_predictions=[top10_row])

        clf = report.failures[0].classification
        assert clf.ranking_depth_issue is False

    def test_no_top10_means_no_ranking_depth(self):
        row = _make_row(
            gold_evidence_ids=["D42"],
            retrieved_dia_ids=[["D1"]],
            evidence_recall=0.0,
        )
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([row], top10_predictions=None)
        assert report.failures[0].classification.ranking_depth_issue is False


# ---------------------------------------------------------------------------
# Provenance mapping issue detection
# ---------------------------------------------------------------------------


class TestProvenanceMappingIssue:
    def test_partial_overlap_is_provenance_issue(self):
        """Some gold ids found, some not → provenance_mapping_issue."""
        row = _make_row(
            gold_evidence_ids=["D1", "D2", "D3"],
            retrieved_dia_ids=[["D1"], ["D4"]],  # D1 found, D2+D3 not
            evidence_recall=0.33,
        )
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([row])

        assert len(report.failures) == 1
        clf = report.failures[0].classification
        assert clf.provenance_mapping_issue is True

    def test_full_overlap_not_provenance_issue(self):
        """All gold ids found → perfect recall, no failure classified."""
        row = _make_row(
            gold_evidence_ids=["D1"],
            retrieved_dia_ids=[["D1"]],
            evidence_recall=1.0,
        )
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([row])
        assert len(report.failures) == 0

    def test_no_overlap_not_provenance_issue(self):
        """Zero overlap → NOT provenance_mapping_issue (either extraction/retrieval miss)."""
        row = _make_row(
            gold_evidence_ids=["D1", "D2"],
            retrieved_dia_ids=[["D9"]],
            evidence_recall=0.0,
        )
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([row])
        clf = report.failures[0].classification
        assert clf.provenance_mapping_issue is False


# ---------------------------------------------------------------------------
# Source coverage calculation
# ---------------------------------------------------------------------------


class TestSourceCoverage:
    def test_coverage_rate_with_store(self, tmp_path):
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.schemas import MemoryUnit

        db_path = tmp_path / "conv1.db"
        store = MemoryStore(db_path)
        mu = MemoryUnit(conversation_id="conv1", session_id="s1",
                        claim="X", original_text="X", source_dia_ids=["D1", "D2"])
        store.insert_memory_unit(mu)

        rows = [
            _make_row(conv_id="conv1", gold_evidence_ids=["D1"], evidence_recall=0.0,
                      retrieved_dia_ids=[]),  # D1 in store → covered
            _make_row(qa_id="qa2", conv_id="conv1", gold_evidence_ids=["D99"],
                      evidence_recall=0.0, retrieved_dia_ids=[]),  # D99 not in store
        ]

        analyzer = Phase2FailureAnalyzer(db_dir=tmp_path)
        report = analyzer.analyze(rows)

        cov = report.coverage
        assert cov.total_gold_dia_ids == 2
        assert cov.gold_dia_ids_in_any_mu == 1
        assert abs(cov.coverage_rate - 0.5) < 0.01

    def test_coverage_zero_without_store(self):
        rows = [_make_row(gold_evidence_ids=["D1"], evidence_recall=0.0)]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows)
        # Without a store, no coverage can be determined → 0/1 covered
        assert report.coverage.total_gold_dia_ids == 1
        assert report.coverage.gold_dia_ids_in_any_mu == 0
        assert report.coverage.coverage_rate == 0.0

    def test_coverage_by_category(self, tmp_path):
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.schemas import MemoryUnit

        db_path = tmp_path / "conv1.db"
        store = MemoryStore(db_path)
        mu = MemoryUnit(conversation_id="conv1", session_id="s1",
                        claim="X", original_text="X", source_dia_ids=["D1"])
        store.insert_memory_unit(mu)

        rows = [
            _make_row(category=1, conv_id="conv1", gold_evidence_ids=["D1"],
                      evidence_recall=0.0, retrieved_dia_ids=[]),
            _make_row(qa_id="qa2", category=3, conv_id="conv1",
                      gold_evidence_ids=["D99"], evidence_recall=0.0,
                      retrieved_dia_ids=[]),
        ]

        analyzer = Phase2FailureAnalyzer(db_dir=tmp_path)
        report = analyzer.analyze(rows)

        by_cat = report.coverage.by_category
        assert "1" in by_cat
        assert "3" in by_cat
        assert by_cat["1"].covered_dia_ids == 1
        assert by_cat["3"].covered_dia_ids == 0

    def test_no_gold_rows_are_skipped(self):
        rows = [_make_row(gold_evidence_ids=[], evidence_recall=None)]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows)
        assert report.n_with_evidence == 0
        assert len(report.failures) == 0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def test_save_analysis_creates_all_files(self, tmp_path):
        rows = [
            _make_row(gold_evidence_ids=["D1"], retrieved_dia_ids=[["D9"]],
                      evidence_recall=0.0),
        ]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows, experiment_name="test_exp")
        paths = save_analysis(report, tmp_path / "out")

        assert (paths["json"]).exists()
        assert (paths["markdown"]).exists()
        assert (paths["failures_csv"]).exists()
        assert (paths["coverage_csv"]).exists()

    def test_json_output_is_valid(self, tmp_path):
        rows = [_make_row(gold_evidence_ids=["D1"], evidence_recall=0.0)]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows, experiment_name="test_exp")
        paths = save_analysis(report, tmp_path / "out")

        data = json.loads(paths["json"].read_text())
        assert data["experiment_name"] == "test_exp"
        assert "failure_type_counts" in data
        assert "coverage" in data
        assert "failures" in data

    def test_markdown_contains_sections(self, tmp_path):
        rows = [_make_row(gold_evidence_ids=["D1"], evidence_recall=0.0)]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows, experiment_name="test_exp")
        paths = save_analysis(report, tmp_path / "out")

        md = paths["markdown"].read_text()
        assert "## Overview" in md
        assert "## Failure Type Counts" in md
        assert "## Source Coverage" in md

    def test_failures_csv_has_rows(self, tmp_path):
        rows = [_make_row(gold_evidence_ids=["D1"], evidence_recall=0.0)]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows, experiment_name="test_exp")
        paths = save_analysis(report, tmp_path / "out")

        import csv
        with paths["failures_csv"].open(encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert "primary_type" in reader[0]

    def test_coverage_csv_has_category_rows(self, tmp_path):
        rows = [
            _make_row(category=1, gold_evidence_ids=["D1"], evidence_recall=0.0),
            _make_row(qa_id="qa2", category=2, gold_evidence_ids=["D2"],
                      evidence_recall=0.0),
        ]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows, experiment_name="test_exp")
        paths = save_analysis(report, tmp_path / "out")

        import csv
        with paths["coverage_csv"].open(encoding="utf-8") as f:
            rows_out = list(csv.DictReader(f))
        categories = {r["category"] for r in rows_out}
        assert {"1", "2"} == categories

    def test_report_as_dict_serialisable(self):
        rows = [_make_row(gold_evidence_ids=["D1"], evidence_recall=0.5,
                          retrieved_dia_ids=[["D1"], ["D9"]])]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows)
        d = report.as_dict()
        # Must be fully JSON-serialisable
        json.dumps(d)

    def test_empty_predictions_produces_empty_report(self, tmp_path):
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze([])
        assert report.n_predictions == 0
        assert len(report.failures) == 0
        paths = save_analysis(report, tmp_path / "out")
        assert paths["json"].exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_perfect_recall_rows_not_in_failures(self):
        rows = [
            _make_row(gold_evidence_ids=["D1"], retrieved_dia_ids=[["D1"]],
                      evidence_recall=1.0),
            _make_row(qa_id="qa2", gold_evidence_ids=["D2"],
                      retrieved_dia_ids=[["D9"]], evidence_recall=0.0),
        ]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows)
        assert report.n_perfect_recall == 1
        assert len(report.failures) == 1

    def test_rows_without_gold_evidence_not_counted(self):
        rows = [
            _make_row(gold_evidence_ids=[], evidence_recall=None),
            _make_row(qa_id="qa2", gold_evidence_ids=None, evidence_recall=None),
        ]
        analyzer = Phase2FailureAnalyzer(db_dir=None)
        report = analyzer.analyze(rows)
        assert report.n_with_evidence == 0

    def test_multiple_conversations(self, tmp_path):
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.schemas import MemoryUnit

        for conv_id, dia_id in [("c1", "D1"), ("c2", "D2")]:
            store = MemoryStore(tmp_path / f"{conv_id}.db")
            mu = MemoryUnit(conversation_id=conv_id, session_id="s1",
                            claim="X", original_text="X", source_dia_ids=[dia_id])
            store.insert_memory_unit(mu)

        rows = [
            _make_row(conv_id="c1", gold_evidence_ids=["D1"], evidence_recall=0.0),
            _make_row(qa_id="qa2", conv_id="c2", gold_evidence_ids=["D99"],
                      evidence_recall=0.0),
        ]

        analyzer = Phase2FailureAnalyzer(db_dir=tmp_path)
        report = analyzer.analyze(rows)

        # c1/D1 is in store → retrieval_miss; c2/D99 not in store → extraction_miss
        types = {f.classification.primary_type() for f in report.failures}
        assert "retrieval_miss" in types
        assert "extraction_miss" in types
