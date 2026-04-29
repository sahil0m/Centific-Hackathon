"""Save failure analysis outputs as Markdown, CSV, and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from locomo_memory.phase2.analysis.failure_analyzer import AnalysisReport


def save_analysis(report: AnalysisReport, output_dir: str | Path) -> dict[str, Path]:
    """Write all analysis outputs and return a dict of {label: path}."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = report.experiment_name.replace(" ", "_")
    paths: dict[str, Path] = {}

    paths["json"] = _save_json(report, out, name)
    paths["markdown"] = _save_markdown(report, out, name)
    paths["failures_csv"] = _save_failures_csv(report, out, name)
    paths["coverage_csv"] = _save_coverage_csv(report, out, name)

    return paths


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _save_json(report: AnalysisReport, out: Path, name: str) -> Path:
    p = out / f"{name}_analysis.json"
    p.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _save_markdown(report: AnalysisReport, out: Path, name: str) -> Path:
    lines: list[str] = []
    a = lines.append

    a(f"# Phase 2 Failure Analysis — {report.experiment_name}\n")

    a("## Overview\n")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Total predictions | {report.n_predictions} |")
    a(f"| QA with gold evidence | {report.n_with_evidence} |")
    a(f"| Perfect recall (=1.0) | {report.n_perfect_recall} |")
    a(f"| Partial recall (0<r<1) | {report.n_partial_recall} |")
    a(f"| Zero recall | {report.n_zero_recall} |")
    a("")

    a("## Failure Type Counts\n")
    a("| Failure Type | Count |")
    a("|--------------|-------|")
    for ft, cnt in sorted(report.failure_type_counts.items(), key=lambda x: -x[1]):
        a(f"| {ft} | {cnt} |")
    a("")

    a("## Source Coverage\n")
    cov = report.coverage
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Total gold dia_ids | {cov.total_gold_dia_ids} |")
    a(f"| Covered by any MU | {cov.gold_dia_ids_in_any_mu} |")
    a(f"| Coverage rate | {cov.coverage_rate:.4f} |")
    a("")

    a("### Coverage by Category\n")
    a("| Cat | N QA | Gold IDs | Covered | Coverage | Avg Recall |")
    a("|-----|------|----------|---------|----------|------------|")
    for cat, cc in sorted(cov.by_category.items()):
        a(
            f"| {cat} | {cc.n_qa} | {cc.gold_dia_ids} | {cc.covered_dia_ids}"
            f" | {cc.coverage_rate:.3f} | {cc.avg_evidence_recall:.3f} |"
        )
    a("")

    if cov.missing_dia_id_examples:
        a("### Examples of Missing Gold dia_ids (extraction_miss candidates)\n")
        for d in cov.missing_dia_id_examples[:10]:
            a(f"- `{d}`")
        a("")

    a("## Failure Sample (first 20)\n")
    for rec in report.failures[:20]:
        a(f"### [{rec.classification.primary_type()}] qa_id={rec.qa_id}\n")
        a(f"**Conv:** {rec.conversation_id}  |  **Cat:** {rec.category}  |  **Recall:** {rec.evidence_recall}\n")
        a(f"**Q:** {rec.question}\n")
        a(f"**Gold answer:** {rec.gold_answer}\n")
        a(f"**Gold evidence dia_ids:** {rec.gold_evidence_ids}\n")
        a(f"**Retrieved dia_ids:** {rec.retrieved_dia_ids_flat}\n")
        a(f"**Gold in retrieved:** {rec.gold_in_retrieved}\n")
        flags = [k for k, v in rec.classification.as_dict().items() if v is True]
        a(f"**Flags:** {', '.join(flags) or 'none'}\n")
        a("---\n")

    p = out / f"{name}_analysis.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _save_failures_csv(report: AnalysisReport, out: Path, name: str) -> Path:
    p = out / f"{name}_failures.csv"
    if not report.failures:
        p.write_text("", encoding="utf-8")
        return p

    fieldnames = list(report.failures[0].as_dict().keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in report.failures:
            d = rec.as_dict()
            # Serialize list fields as strings for CSV readability
            for key in ("gold_evidence_ids", "retrieved_dia_ids_flat", "retrieved_claims"):
                if isinstance(d.get(key), list):
                    d[key] = "|".join(str(x) for x in d[key])
            w.writerow(d)
    return p


def _save_coverage_csv(report: AnalysisReport, out: Path, name: str) -> Path:
    p = out / f"{name}_coverage_by_category.csv"
    cov = report.coverage
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["category", "n_qa", "n_with_evidence", "gold_dia_ids",
                        "covered_dia_ids", "coverage_rate", "avg_evidence_recall"],
        )
        w.writeheader()
        for cat, cc in sorted(cov.by_category.items()):
            w.writerow({"category": cat, **cc.as_dict()})
    return p
