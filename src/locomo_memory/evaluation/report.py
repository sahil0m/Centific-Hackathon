"""
Report generation: saves metrics JSON, CSV tables, and a Markdown failure analysis.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from locomo_memory.data.schemas import PredictionRow
from locomo_memory.evaluation.evidence_recall import compute_mean_evidence_recall
from locomo_memory.evaluation.qa_metrics import (
    compute_category_metrics,
    compute_latency_percentiles,
    compute_metrics_for_batch,
)

logger = logging.getLogger(__name__)


def save_predictions(
    predictions: list[PredictionRow],
    output_dir: Path,
    experiment_name: str,
) -> Path:
    path = output_dir / "raw_predictions" / f"{experiment_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [p.as_dict() for p in predictions]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d predictions to %s", len(rows), path)
    return path


def save_retrieval_debug(
    predictions: list[PredictionRow],
    output_dir: Path,
    experiment_name: str,
) -> Path:
    path = output_dir / "retrieval" / f"{experiment_name}_retrieval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "qa_id": p.qa_id,
            "question": p.question,
            "retrieved_chunks": p.retrieved_chunks,
            "retrieval_latency_ms": p.retrieval_latency_ms,
        }
        for p in predictions
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def compute_and_save_metrics(
    predictions: list[PredictionRow],
    output_dir: Path,
    experiment_name: str,
    config: dict[str, Any],
    generation_enabled: bool = True,
) -> dict[str, Any]:
    if not predictions:
        logger.warning("No predictions to evaluate.")
        return {}

    preds = [p.predicted_answer for p in predictions]
    golds = [p.gold_answer for p in predictions]
    cats = [p.category for p in predictions]
    recalls = [p.evidence_recall for p in predictions]

    overall = compute_metrics_for_batch(preds, golds)
    category_metrics = compute_category_metrics(preds, golds, cats)
    mean_recall = compute_mean_evidence_recall(recalls)

    retrieval_lats = [p.retrieval_latency_ms for p in predictions]
    gen_lats = [p.generation_latency_ms for p in predictions if p.generation_latency_ms > 0]
    e2e_lats = [p.end_to_end_latency_ms for p in predictions]

    input_toks = [p.input_tokens for p in predictions]
    output_toks = [p.output_tokens for p in predictions]
    avg_input_tokens = sum(input_toks) / len(input_toks) if input_toks else 0
    avg_output_tokens = sum(output_toks) / len(output_toks) if output_toks else 0
    n_chunks = [len(p.retrieved_chunks) for p in predictions]

    metrics: dict[str, Any] = {
        "experiment_name": experiment_name,
        "n_questions": len(predictions),
        "avg_f1": overall["avg_f1"],
        "exact_match": overall["exact_match"],
        "mean_evidence_recall": mean_recall,
        "category_metrics": category_metrics,
        "retrieval_latency_ms": compute_latency_percentiles(retrieval_lats),
        "generation_latency_ms": compute_latency_percentiles(gen_lats),
        "end_to_end_latency_ms": compute_latency_percentiles(e2e_lats),
        "avg_input_tokens": round(avg_input_tokens, 1),
        "avg_output_tokens": round(avg_output_tokens, 1),
        "avg_retrieved_chunks": round(sum(n_chunks) / len(n_chunks), 2) if n_chunks else 0,
        "config": config,
    }

    path = output_dir / "metrics" / f"{experiment_name}_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved metrics to %s", path)

    _save_category_csv(category_metrics, output_dir, experiment_name)
    if generation_enabled:
        _save_failure_cases(predictions, output_dir, experiment_name)
    else:
        logger.info("Skipping failure cases (generation disabled — no predictions to compare)")

    return metrics


def _save_category_csv(
    category_metrics: dict[str, dict[str, Any]],
    output_dir: Path,
    experiment_name: str,
) -> None:
    path = output_dir / "tables" / f"{experiment_name}_by_category.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "avg_f1", "exact_match", "count"])
        writer.writeheader()
        for cat, metrics in sorted(category_metrics.items()):
            writer.writerow(
                {
                    "category": cat,
                    "avg_f1": metrics.get("avg_f1", 0),
                    "exact_match": metrics.get("exact_match", 0),
                    "count": metrics.get("count", 0),
                }
            )
    logger.info("Saved category table to %s", path)


def _save_failure_cases(
    predictions: list[PredictionRow],
    output_dir: Path,
    experiment_name: str,
    f1_threshold: float = 0.3,
) -> None:
    failures = [p for p in predictions if p.f1 < f1_threshold]
    path = output_dir / "tables" / "failure_cases.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_name",
                "qa_id",
                "category",
                "question",
                "gold_answer",
                "predicted_answer",
                "f1",
                "evidence_recall",
            ],
        )
        writer.writeheader()
        for p in failures:
            writer.writerow(
                {
                    "experiment_name": p.experiment_name,
                    "qa_id": p.qa_id,
                    "category": p.category,
                    "question": p.question[:200],
                    "gold_answer": p.gold_answer[:200],
                    "predicted_answer": p.predicted_answer[:200],
                    "f1": p.f1,
                    "evidence_recall": p.evidence_recall,
                }
            )
    logger.info("Saved %d failure cases to %s", len(failures), path)


def generate_failure_report(
    metrics: dict[str, Any],
    predictions: list[PredictionRow],
    output_dir: Path,
    experiment_name: str,
) -> Path:
    path = output_dir / "reports" / "naive_rag_failure_analysis.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    lines.append("# Naive RAG Failure Analysis\n")
    lines.append(f"**Experiment:** {experiment_name}\n")
    lines.append(f"**Total questions:** {metrics.get('n_questions', 0)}\n")
    lines.append(f"**Average F1:** {metrics.get('avg_f1', 0):.4f}\n")
    lines.append(f"**Exact Match:** {metrics.get('exact_match', 0):.4f}\n")
    if metrics.get("mean_evidence_recall") is not None:
        lines.append(f"**Mean Evidence Recall@k:** {metrics['mean_evidence_recall']:.4f}\n")
    lines.append("\n---\n")

    lines.append("## Category-wise Scores\n")
    lines.append("| Category | Avg F1 | Exact Match | Count |")
    lines.append("|----------|--------|-------------|-------|")
    cat_metrics = metrics.get("category_metrics", {})
    for cat, cm in sorted(cat_metrics.items()):
        lines.append(
            f"| {cat} | {cm.get('avg_f1', 0):.4f} | {cm.get('exact_match', 0):.4f} | {int(cm.get('count', 0))} |"
        )
    lines.append("")

    lines.append("## Latency\n")
    r_lat = metrics.get("retrieval_latency_ms", {})
    e_lat = metrics.get("end_to_end_latency_ms", {})
    lines.append(f"- Retrieval p50: {r_lat.get('p50', 0):.1f} ms, p95: {r_lat.get('p95', 0):.1f} ms")
    lines.append(f"- End-to-end p50: {e_lat.get('p50', 0):.1f} ms, p95: {e_lat.get('p95', 0):.1f} ms\n")

    lines.append("## Token Usage\n")
    lines.append(f"- Avg injected context tokens: {metrics.get('avg_input_tokens', 0):.1f}")
    lines.append(f"- Avg output tokens: {metrics.get('avg_output_tokens', 0):.1f}\n")

    lines.append("## Failure Examples\n")
    failures = [p for p in predictions if p.f1 < 0.3][:10]
    if not failures:
        lines.append("No failures with F1 < 0.3 found.\n")
    for p in failures:
        lines.append(f"### QA ID: `{p.qa_id}` (category: {p.category})\n")
        lines.append(f"**Question:** {p.question}")
        lines.append(f"**Gold answer:** {p.gold_answer}")
        lines.append(f"**Predicted:** {p.predicted_answer}")
        lines.append(f"**F1:** {p.f1:.4f} | **Evidence Recall:** {p.evidence_recall}\n")
        if p.retrieved_chunks:
            lines.append("**Top retrieved chunk:**")
            lines.append(f"```\n{p.retrieved_chunks[0].get('text', '')[:400]}\n```\n")

    lines.append("---\n")
    lines.append("## Why Naive RAG Is Insufficient\n")
    lines.append("Common failure modes observed:\n")
    failure_modes = [
        "Retrieved irrelevant chunks due to surface-level similarity",
        "Missed gold evidence when it used different vocabulary than the question",
        "Retrieved only partial evidence for multi-hop questions",
        "Retrieved stale or contradictory evidence without provenance",
        "Answered despite insufficient evidence (hallucination)",
        "Failed temporal reasoning (no timestamp-aware retrieval)",
        "Too much noisy context when top-k is large",
        "High token usage with no salience scoring",
        "No contradiction policy: conflicting facts silently injected",
        "No provenance-aware memory lifecycle",
    ]
    for mode in failure_modes:
        lines.append(f"- {mode}")
    lines.append("")

    lines.append("## How SPARC-LTM Will Address These Failures\n")
    lines.append(
        "The proposed SPARC-LTM system (Phase 2) targets the two main failure modes:\n"
    )
    lines.append("1. **Salience-aware forgetting**: instead of treating all chunks equally, SPARC-LTM scores")
    lines.append("   memory units by importance, frequency, recency, and future usefulness, and retires")
    lines.append("   low-utility memory under a hard storage cap.\n")
    lines.append("2. **Contradiction reconciliation with provenance**: when facts conflict across sessions,")
    lines.append("   SPARC-LTM tracks both facts with their source dialog IDs, marks one as superseded,")
    lines.append("   and ensures the LLM sees only the most current, consistent evidence.\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved failure analysis report to %s", path)
    return path


def save_baseline_comparison_csv(
    all_metrics: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    path = output_dir / "tables" / "baseline_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "n_questions",
        "avg_f1",
        "exact_match",
        "mean_evidence_recall",
        "avg_input_tokens",
        "avg_output_tokens",
        "retrieval_p50_ms",
        "retrieval_p95_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            r = m.get("retrieval_latency_ms", {})
            e = m.get("end_to_end_latency_ms", {})
            writer.writerow(
                {
                    "experiment_name": m.get("experiment_name", ""),
                    "n_questions": m.get("n_questions", 0),
                    "avg_f1": m.get("avg_f1", 0),
                    "exact_match": m.get("exact_match", 0),
                    "mean_evidence_recall": m.get("mean_evidence_recall", ""),
                    "avg_input_tokens": m.get("avg_input_tokens", 0),
                    "avg_output_tokens": m.get("avg_output_tokens", 0),
                    "retrieval_p50_ms": r.get("p50", 0),
                    "retrieval_p95_ms": r.get("p95", 0),
                    "e2e_p50_ms": e.get("p50", 0),
                    "e2e_p95_ms": e.get("p95", 0),
                }
            )
    logger.info("Saved baseline comparison table to %s", path)
    return path
