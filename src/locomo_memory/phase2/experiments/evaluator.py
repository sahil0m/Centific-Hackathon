"""Phase 2 evaluation harness — Milestone 10.

Collects per-prediction results, computes aggregate metrics, and saves all
output files in the same tree as Phase 1 (``results/phase2/``).

No LLM calls are made here.  All computation is pure Python arithmetic over
already-generated predictions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Per-prediction record
# ---------------------------------------------------------------------------


@dataclass
class Phase2PredictionRow:
    """One QA item's full result from the Phase 2 pipeline."""

    experiment_name: str
    conversation_id: str
    qa_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    category: str

    gold_evidence_ids: list[str]
    retrieved_mu_ids: list[str]
    retrieved_claims: list[str]
    retrieved_dia_ids: list[list[str]]
    """source_dia_ids for each retrieved MU (parallel to retrieved_mu_ids)."""

    context_sections: dict[str, list[str]]
    """Section name → list of claims in that section."""

    f1: float
    exact_match: bool
    evidence_recall: float | None

    grounding_score: float
    guard_passed: bool
    guard_warnings: list[str]

    retrieval_latency_ms: float
    generation_latency_ms: float
    end_to_end_latency_ms: float

    # --- Diagnostic fields (Milestone 14) ---
    hit_lane_sources: list[list[str]] = field(default_factory=list)
    """Per-hit list of lanes that contributed: parallel to retrieved_mu_ids."""
    hit_session_ids: list[str] = field(default_factory=list)
    """session_id of each retrieved MU: parallel to retrieved_mu_ids."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class Phase2CategoryMetrics:
    category: str
    count: int
    avg_f1: float
    exact_match_rate: float
    avg_evidence_recall: float | None
    avg_grounding_score: float
    guard_pass_rate: float


@dataclass
class Phase2Metrics:
    """Aggregate metrics for one experiment run."""

    experiment_name: str
    n_predictions: int
    n_conversations: int

    avg_f1: float
    exact_match_rate: float
    avg_evidence_recall: float | None
    avg_grounding_score: float
    guard_pass_rate: float

    retrieval_latency_p50: float
    retrieval_latency_p95: float
    retrieval_latency_mean: float

    end_to_end_latency_p50: float
    end_to_end_latency_p95: float

    avg_retrieved_mus: float
    """Average number of MUs retrieved per question."""

    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- Diagnostic fields (Milestone 14) ---
    zero_recall_count: int = 0
    """QAs with evidence_recall == 0.0."""
    perfect_recall_count: int = 0
    """QAs with evidence_recall == 1.0."""
    lane_distribution: dict[str, int] = field(default_factory=dict)
    """Total top-k hits contributed by each lane across all QA items."""
    avg_duplicate_dia_ids: float = 0.0
    """Average number of duplicate dia_ids within a single QA's top-k result."""
    avg_same_session_density: float = 0.0
    """Average fraction of top-k hits from the single most common session_id."""


# ---------------------------------------------------------------------------
# Run result container
# ---------------------------------------------------------------------------


@dataclass
class Phase2RunResult:
    """Full output of one Phase 2 experiment run."""

    experiment_name: str
    n_conversations: int
    n_qa_items: int
    predictions: list[Phase2PredictionRow]
    metrics: Phase2Metrics | None = None

    @property
    def n_predictions(self) -> int:
        return len(self.predictions)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Phase2Evaluator:
    """Computes aggregate metrics and persists all result files.

    Args:
        experiment_name: used in file names and metric records.
        output_dir: root directory for result files.
    """

    def __init__(self, experiment_name: str, output_dir: str | Path = "results/phase2") -> None:
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        predictions: list[Phase2PredictionRow],
        *,
        n_conversations: int = 0,
    ) -> Phase2Metrics:
        """Compute all aggregate metrics from a list of prediction rows."""
        if not predictions:
            return Phase2Metrics(
                experiment_name=self.experiment_name,
                n_predictions=0,
                n_conversations=n_conversations,
                avg_f1=0.0,
                exact_match_rate=0.0,
                avg_evidence_recall=None,
                avg_grounding_score=0.0,
                guard_pass_rate=0.0,
                retrieval_latency_p50=0.0,
                retrieval_latency_p95=0.0,
                retrieval_latency_mean=0.0,
                end_to_end_latency_p50=0.0,
                end_to_end_latency_p95=0.0,
                avg_retrieved_mus=0.0,
            )

        n = len(predictions)

        # --- Core QA metrics ---
        avg_f1 = round(sum(p.f1 for p in predictions) / n, 4)
        em_rate = round(sum(1 for p in predictions if p.exact_match) / n, 4)

        recalls = [p.evidence_recall for p in predictions if p.evidence_recall is not None]
        avg_recall: float | None = (
            round(sum(recalls) / len(recalls), 4) if recalls else None
        )

        avg_grounding = round(sum(p.grounding_score for p in predictions) / n, 4)
        guard_pass = round(sum(1 for p in predictions if p.guard_passed) / n, 4)

        # --- Latency ---
        ret_lats = sorted(p.retrieval_latency_ms for p in predictions)
        e2e_lats = sorted(p.end_to_end_latency_ms for p in predictions)
        ret_p = _percentiles(ret_lats)
        e2e_p = _percentiles(e2e_lats)

        # --- Retrieved MU count ---
        avg_mus = round(sum(len(p.retrieved_mu_ids) for p in predictions) / n, 2)

        # --- Per-category ---
        by_category: dict[str, dict[str, Any]] = {}
        cat_map: dict[str, list[Phase2PredictionRow]] = {}
        for pred in predictions:
            cat_map.setdefault(pred.category, []).append(pred)

        for cat, cat_preds in sorted(cat_map.items()):
            cat_n = len(cat_preds)
            cat_recalls = [p.evidence_recall for p in cat_preds if p.evidence_recall is not None]
            by_category[cat] = {
                "count": cat_n,
                "avg_f1": round(sum(p.f1 for p in cat_preds) / cat_n, 4),
                "exact_match_rate": round(sum(1 for p in cat_preds if p.exact_match) / cat_n, 4),
                "avg_evidence_recall": (
                    round(sum(cat_recalls) / len(cat_recalls), 4) if cat_recalls else None
                ),
                "avg_grounding_score": round(
                    sum(p.grounding_score for p in cat_preds) / cat_n, 4
                ),
                "guard_pass_rate": round(
                    sum(1 for p in cat_preds if p.guard_passed) / cat_n, 4
                ),
            }

        # --- Diagnostics ---
        zero_recall = sum(
            1 for p in predictions
            if p.evidence_recall is not None and p.evidence_recall == 0.0
        )
        perfect_recall = sum(
            1 for p in predictions
            if p.evidence_recall is not None and p.evidence_recall == 1.0
        )

        # Lane distribution: count how many final top-k hits each lane contributed
        lane_dist: dict[str, int] = {}
        for pred in predictions:
            for sources in pred.hit_lane_sources:
                for src in sources:
                    lane_dist[src] = lane_dist.get(src, 0) + 1

        # Duplicate dia_ids: count dia_ids appearing more than once across a QA's top-k
        dup_counts: list[float] = []
        for pred in predictions:
            all_dias: list[str] = []
            for dia_list in pred.retrieved_dia_ids:
                all_dias.extend(dia_list)
            seen: set[str] = set()
            dups = sum(1 for d in all_dias if d in seen or seen.add(d))  # type: ignore[func-returns-value]
            dup_counts.append(float(dups))
        avg_dup = round(sum(dup_counts) / len(dup_counts), 4) if dup_counts else 0.0

        # Same-session density: fraction of top-k from the most common session
        session_densities: list[float] = []
        for pred in predictions:
            if not pred.hit_session_ids:
                continue
            from collections import Counter
            cnt = Counter(pred.hit_session_ids)
            most_common = cnt.most_common(1)[0][1]
            session_densities.append(most_common / len(pred.hit_session_ids))
        avg_sess = round(
            sum(session_densities) / len(session_densities), 4
        ) if session_densities else 0.0

        return Phase2Metrics(
            experiment_name=self.experiment_name,
            n_predictions=n,
            n_conversations=n_conversations,
            avg_f1=avg_f1,
            exact_match_rate=em_rate,
            avg_evidence_recall=avg_recall,
            avg_grounding_score=avg_grounding,
            guard_pass_rate=guard_pass,
            retrieval_latency_p50=ret_p["p50"],
            retrieval_latency_p95=ret_p["p95"],
            retrieval_latency_mean=ret_p["mean"],
            end_to_end_latency_p50=e2e_p["p50"],
            end_to_end_latency_p95=e2e_p["p95"],
            avg_retrieved_mus=avg_mus,
            by_category=by_category,
            zero_recall_count=zero_recall,
            perfect_recall_count=perfect_recall,
            lane_distribution=lane_dist,
            avg_duplicate_dia_ids=avg_dup,
            avg_same_session_density=avg_sess,
        )

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, result: Phase2RunResult) -> None:
        """Persist predictions, metrics, category CSV, and failure CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = self.experiment_name

        if result.metrics is None:
            result.metrics = self.compute_metrics(
                result.predictions, n_conversations=result.n_conversations
            )

        # Predictions JSON
        preds_path = self.output_dir / "raw_predictions" / f"{name}.json"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_path.write_text(
            json.dumps([p.as_dict() for p in result.predictions], indent=2),
            encoding="utf-8",
        )
        logger.info("Saved {} predictions → {}", len(result.predictions), preds_path)

        # Metrics JSON
        metrics_path = self.output_dir / "metrics" / f"{name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(asdict(result.metrics), indent=2),
            encoding="utf-8",
        )
        logger.info("Saved metrics → {}", metrics_path)

        # Category CSV
        cat_csv_path = self.output_dir / "tables" / f"{name}_by_category.csv"
        cat_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_category_csv(result.metrics, cat_csv_path)
        logger.info("Saved category table → {}", cat_csv_path)

        # Failure cases CSV (F1 < 0.3, only when generation enabled)
        failure_preds = [p for p in result.predictions if p.f1 < 0.3 and p.predicted_answer]
        if failure_preds:
            fail_path = self.output_dir / "tables" / f"{name}_failures.csv"
            self._save_failures_csv(failure_preds, fail_path)
            logger.info("Saved {} failure cases → {}", len(failure_preds), fail_path)

    def _save_category_csv(self, metrics: Phase2Metrics, path: Path) -> None:
        rows = []
        for cat, m in metrics.by_category.items():
            rows.append({
                "experiment": self.experiment_name,
                "category": cat,
                "count": m["count"],
                "avg_f1": m["avg_f1"],
                "exact_match_rate": m["exact_match_rate"],
                "avg_evidence_recall": m.get("avg_evidence_recall", ""),
                "avg_grounding_score": m["avg_grounding_score"],
                "guard_pass_rate": m["guard_pass_rate"],
            })
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _save_failures_csv(self, preds: list[Phase2PredictionRow], path: Path) -> None:
        fields = [
            "experiment_name", "conversation_id", "qa_id", "question",
            "gold_answer", "predicted_answer", "category", "f1",
            "evidence_recall", "grounding_score",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for p in preds:
                writer.writerow(p.as_dict())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentiles(sorted_values: list[float]) -> dict[str, float]:
    if not sorted_values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    n = len(sorted_values)
    return {
        "p50": round(sorted_values[max(0, int(0.50 * n) - 1)], 2),
        "p95": round(sorted_values[max(0, int(0.95 * n) - 1)], 2),
        "mean": round(sum(sorted_values) / n, 2),
    }


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Phase2Evaluator",
    "Phase2Metrics",
    "Phase2PredictionRow",
    "Phase2RunResult",
]
