"""Phase 2 failure analyzer — Milestone 11.

Classifies retrieval failures from Phase 2 prediction output and produces
coverage/ranking diagnostics without touching any retrieval logic.

Failure types
-------------
extraction_miss
    No MemoryUnit in the store has source_dia_ids overlapping with any gold
    evidence dia_id.  The ingestion pipeline never captured this turn.

retrieval_miss
    At least one MemoryUnit covers a gold dia_id, but the retriever did not
    surface it in top-k.

ranking_depth_issue
    Gold evidence appears in the top-10 predictions but NOT in top-5.
    Requires a second predictions set from a top-10 run.

provenance_mapping_issue
    Some (but not all) gold dia_ids appear in retrieved results.  The
    retriever found *part* of the evidence but missed the rest.

temporal_issue
    Question contains strong temporal/date cues and retrieval still failed.

unknown
    None of the above could be determined.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Temporal keyword detection
# ---------------------------------------------------------------------------

_TEMPORAL_PATTERNS = re.compile(
    r"\b(when|before|after|since|during|how long|first time|last time|"
    r"year|month|week|date|ago|recently|earliest|latest|order|sequence|"
    r"timeline|chronolog|which came first|who came first|prior to)\b",
    re.IGNORECASE,
)


def _is_temporal(question: str) -> bool:
    return bool(_TEMPORAL_PATTERNS.search(question))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureClassification:
    extraction_miss: bool = False
    retrieval_miss: bool = False
    ranking_depth_issue: bool = False
    provenance_mapping_issue: bool = False
    temporal_issue: bool = False

    def primary_type(self) -> str:
        if self.extraction_miss:
            return "extraction_miss"
        if self.ranking_depth_issue:
            return "ranking_depth_issue"
        if self.retrieval_miss:
            return "retrieval_miss"
        if self.provenance_mapping_issue:
            return "provenance_mapping_issue"
        if self.temporal_issue:
            return "temporal_issue"
        return "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_type": self.primary_type(),
            "extraction_miss": self.extraction_miss,
            "retrieval_miss": self.retrieval_miss,
            "ranking_depth_issue": self.ranking_depth_issue,
            "provenance_mapping_issue": self.provenance_mapping_issue,
            "temporal_issue": self.temporal_issue,
        }


@dataclass
class QAFailureRecord:
    conversation_id: str
    qa_id: str
    question: str
    gold_answer: str
    category: str | int
    gold_evidence_ids: list[str]
    retrieved_dia_ids_flat: list[str]
    retrieved_claims: list[str]
    evidence_recall: float | None
    gold_in_retrieved: bool
    classification: FailureClassification

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "qa_id": self.qa_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "category": self.category,
            "gold_evidence_ids": self.gold_evidence_ids,
            "retrieved_dia_ids_flat": self.retrieved_dia_ids_flat,
            "retrieved_claims": self.retrieved_claims,
            "evidence_recall": self.evidence_recall,
            "gold_in_retrieved": self.gold_in_retrieved,
            **self.classification.as_dict(),
        }


@dataclass
class CategoryCoverage:
    n_qa: int = 0
    n_with_evidence: int = 0
    gold_dia_ids: int = 0
    covered_dia_ids: int = 0
    coverage_rate: float = 0.0
    avg_evidence_recall: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_qa": self.n_qa,
            "n_with_evidence": self.n_with_evidence,
            "gold_dia_ids": self.gold_dia_ids,
            "covered_dia_ids": self.covered_dia_ids,
            "coverage_rate": round(self.coverage_rate, 4),
            "avg_evidence_recall": round(self.avg_evidence_recall, 4),
        }


@dataclass
class CoverageStats:
    total_qa: int
    total_qa_with_evidence: int
    total_gold_dia_ids: int
    gold_dia_ids_in_any_mu: int
    coverage_rate: float
    by_category: dict[str, CategoryCoverage] = field(default_factory=dict)
    missing_dia_id_examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_qa": self.total_qa,
            "total_qa_with_evidence": self.total_qa_with_evidence,
            "total_gold_dia_ids": self.total_gold_dia_ids,
            "gold_dia_ids_in_any_mu": self.gold_dia_ids_in_any_mu,
            "coverage_rate": round(self.coverage_rate, 4),
            "by_category": {k: v.as_dict() for k, v in self.by_category.items()},
            "missing_dia_id_examples": self.missing_dia_id_examples[:20],
        }


@dataclass
class AnalysisReport:
    experiment_name: str
    n_predictions: int
    n_with_evidence: int
    n_perfect_recall: int
    n_partial_recall: int
    n_zero_recall: int
    failure_type_counts: dict[str, int]
    coverage: CoverageStats
    failures: list[QAFailureRecord]

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "n_predictions": self.n_predictions,
            "n_with_evidence": self.n_with_evidence,
            "n_perfect_recall": self.n_perfect_recall,
            "n_partial_recall": self.n_partial_recall,
            "n_zero_recall": self.n_zero_recall,
            "failure_type_counts": self.failure_type_counts,
            "coverage": self.coverage.as_dict(),
            "failures": [f.as_dict() for f in self.failures],
        }


# ---------------------------------------------------------------------------
# Store-based extraction check
# ---------------------------------------------------------------------------


def _build_conv_dia_coverage(db_dir: Path) -> dict[str, set[str]]:
    """Map conversation_id → set of all dia_ids covered by any MemoryUnit.

    Returns an empty dict if db_dir is None or does not exist.
    """
    coverage: dict[str, set[str]] = {}
    if not db_dir or not db_dir.exists():
        return coverage

    for db_path in db_dir.glob("*.db"):
        conv_id = db_path.stem
        try:
            from locomo_memory.phase2.store.sqlite_store import MemoryStore
            store = MemoryStore(db_path)
            mus = store.list_all(conv_id)
            dia_set: set[str] = set()
            for mu in mus:
                dia_set.update(mu.source_dia_ids)
            coverage[conv_id] = dia_set
        except Exception:
            coverage[conv_id] = set()
    return coverage


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------


class Phase2FailureAnalyzer:
    """Classify Phase 2 retrieval failures from saved prediction rows.

    Args:
        db_dir: path to directory containing per-conversation SQLite stores
            (``{conv_id}.db``).  When provided, enables extraction_miss vs
            retrieval_miss distinction.  When None, all non-retrieved evidence
            is classified as retrieval_miss.
    """

    def __init__(self, *, db_dir: str | Path | None = None) -> None:
        self._db_dir = Path(db_dir) if db_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        predictions: list[dict[str, Any]],
        *,
        top10_predictions: list[dict[str, Any]] | None = None,
        experiment_name: str = "phase2",
    ) -> AnalysisReport:
        """Produce a full failure analysis report.

        Args:
            predictions: list of prediction row dicts (from the top-5 run).
            top10_predictions: optional list from a top-10 run; enables
                ranking_depth_issue detection.
            experiment_name: label for the report.
        """
        # Build top-10 index for ranking_depth_issue detection
        top10_index = self._build_top10_index(top10_predictions)

        # Build extraction coverage from stores
        conv_dia_coverage = _build_conv_dia_coverage(self._db_dir)

        failures: list[QAFailureRecord] = []
        n_with_evidence = 0
        n_perfect = 0
        n_partial = 0
        n_zero = 0
        failure_type_counts: dict[str, int] = {}

        # Per-category accumulators
        cat_accum: dict[str, dict[str, Any]] = {}

        for row in predictions:
            gold_ids: list[str] = row.get("gold_evidence_ids") or []
            if not gold_ids:
                continue

            n_with_evidence += 1
            er: float | None = row.get("evidence_recall")
            cat = str(row.get("category", "unknown"))

            if cat not in cat_accum:
                cat_accum[cat] = {
                    "n_qa": 0,
                    "n_with_ev": 0,
                    "gold_ids": 0,
                    "covered_ids": 0,
                    "recall_sum": 0.0,
                }
            cat_accum[cat]["n_qa"] += 1
            cat_accum[cat]["n_with_ev"] += 1
            cat_accum[cat]["gold_ids"] += len(gold_ids)
            if er is not None:
                cat_accum[cat]["recall_sum"] += er

            if er is not None and er >= 1.0:
                n_perfect += 1
            elif er is not None and er > 0.0:
                n_partial += 1
            else:
                n_zero += 1

            # Flatten retrieved dia_ids
            raw_retrieved: list[list[str]] = row.get("retrieved_dia_ids") or []
            flat_retrieved: list[str] = [d for sub in raw_retrieved for d in sub]
            gold_in_retrieved = any(g in flat_retrieved for g in gold_ids)

            # Only classify failures (evidence_recall < 1.0 or None)
            if er is not None and er >= 1.0:
                continue

            clf = self._classify(
                row=row,
                gold_ids=gold_ids,
                flat_retrieved=flat_retrieved,
                gold_in_retrieved=gold_in_retrieved,
                top10_index=top10_index,
                conv_dia_coverage=conv_dia_coverage,
            )
            ptype = clf.primary_type()
            failure_type_counts[ptype] = failure_type_counts.get(ptype, 0) + 1

            failures.append(
                QAFailureRecord(
                    conversation_id=row.get("conversation_id", ""),
                    qa_id=row.get("qa_id", ""),
                    question=row.get("question", ""),
                    gold_answer=row.get("gold_answer", ""),
                    category=cat,
                    gold_evidence_ids=gold_ids,
                    retrieved_dia_ids_flat=flat_retrieved,
                    retrieved_claims=row.get("retrieved_claims") or [],
                    evidence_recall=er,
                    gold_in_retrieved=gold_in_retrieved,
                    classification=clf,
                )
            )

        coverage = self._compute_coverage(predictions, conv_dia_coverage, cat_accum)

        return AnalysisReport(
            experiment_name=experiment_name,
            n_predictions=len(predictions),
            n_with_evidence=n_with_evidence,
            n_perfect_recall=n_perfect,
            n_partial_recall=n_partial,
            n_zero_recall=n_zero,
            failure_type_counts=failure_type_counts,
            coverage=coverage,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        *,
        row: dict[str, Any],
        gold_ids: list[str],
        flat_retrieved: list[str],
        gold_in_retrieved: bool,
        top10_index: dict[str, set[str]],
        conv_dia_coverage: dict[str, set[str]],
    ) -> FailureClassification:
        conv_id = row.get("conversation_id", "")
        question = row.get("question", "")
        qa_id = row.get("qa_id", "")

        gold_set = set(gold_ids)
        retrieved_set = set(flat_retrieved)

        # provenance_mapping_issue: some but not all gold found
        partial_overlap = bool(gold_set & retrieved_set) and not gold_set.issubset(retrieved_set)

        # ranking_depth_issue: gold in top-10 but not top-5
        ranking_depth = False
        if top10_index and qa_id in top10_index:
            top10_dias = top10_index[qa_id]
            if gold_set & top10_dias and not (gold_set & retrieved_set):
                ranking_depth = True

        # extraction_miss vs retrieval_miss
        extraction_miss = False
        retrieval_miss = False
        if not gold_in_retrieved:
            if conv_id in conv_dia_coverage:
                store_dias = conv_dia_coverage[conv_id]
                if not (gold_set & store_dias):
                    extraction_miss = True
                else:
                    retrieval_miss = True
            else:
                # No store data; cannot distinguish — mark retrieval_miss
                retrieval_miss = True

        temporal = _is_temporal(question) and not gold_in_retrieved

        return FailureClassification(
            extraction_miss=extraction_miss,
            retrieval_miss=retrieval_miss,
            ranking_depth_issue=ranking_depth,
            provenance_mapping_issue=partial_overlap,
            temporal_issue=temporal,
        )

    # ------------------------------------------------------------------
    # Coverage computation
    # ------------------------------------------------------------------

    def _compute_coverage(
        self,
        predictions: list[dict[str, Any]],
        conv_dia_coverage: dict[str, set[str]],
        cat_accum: dict[str, dict[str, Any]],
    ) -> CoverageStats:
        total_gold = 0
        covered = 0
        missing_examples: list[str] = []

        for row in predictions:
            gold_ids: list[str] = row.get("gold_evidence_ids") or []
            if not gold_ids:
                continue
            conv_id = row.get("conversation_id", "")
            store_dias = conv_dia_coverage.get(conv_id, set())
            cat = str(row.get("category", "unknown"))

            for gid in gold_ids:
                total_gold += 1
                if gid in store_dias:
                    covered += 1
                    if cat in cat_accum:
                        cat_accum[cat]["covered_ids"] = cat_accum[cat].get("covered_ids", 0) + 1
                else:
                    if len(missing_examples) < 50:
                        missing_examples.append(gid)

        coverage_rate = covered / max(total_gold, 1)

        # Build per-category coverage
        by_cat: dict[str, CategoryCoverage] = {}
        for cat, acc in cat_accum.items():
            n_ev = acc["n_with_ev"]
            g = acc["gold_ids"]
            c = acc.get("covered_ids", 0)
            r = acc["recall_sum"] / max(n_ev, 1)
            by_cat[cat] = CategoryCoverage(
                n_qa=acc["n_qa"],
                n_with_evidence=n_ev,
                gold_dia_ids=g,
                covered_dia_ids=c,
                coverage_rate=c / max(g, 1),
                avg_evidence_recall=r,
            )

        return CoverageStats(
            total_qa=len(predictions),
            total_qa_with_evidence=sum(acc["n_with_ev"] for acc in cat_accum.values()),
            total_gold_dia_ids=total_gold,
            gold_dia_ids_in_any_mu=covered,
            coverage_rate=coverage_rate,
            by_category=by_cat,
            missing_dia_id_examples=missing_examples,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_top10_index(
        top10_predictions: list[dict[str, Any]] | None,
    ) -> dict[str, set[str]]:
        """Map qa_id → set of all dia_ids in top-10 retrieved hits."""
        if not top10_predictions:
            return {}
        index: dict[str, set[str]] = {}
        for row in top10_predictions:
            qa_id = row.get("qa_id", "")
            raw: list[list[str]] = row.get("retrieved_dia_ids") or []
            flat = {d for sub in raw for d in sub}
            index[qa_id] = flat
        return index
