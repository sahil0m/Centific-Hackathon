"""Phase 2 failure analysis CLI — Milestone 11.

Usage::

    python -m locomo_memory.phase2.analysis.run_analysis \\
        --predictions results/phase2/phase2_retrieval_only_predictions.json \\
        --db-dir data/processed/phase2_db \\
        [--top10-predictions results/phase2/phase2_retrieval_top10_predictions.json] \\
        [--output-dir results/phase2/analysis]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from locomo_memory.phase2.analysis.failure_analyzer import Phase2FailureAnalyzer
from locomo_memory.phase2.analysis.report_writer import save_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 retrieval failure analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to phase2 predictions JSON (from the top-5 run).",
    )
    parser.add_argument(
        "--db-dir",
        default=None,
        help="Directory containing per-conversation SQLite stores. "
             "Required for extraction_miss vs retrieval_miss distinction.",
    )
    parser.add_argument(
        "--top10-predictions",
        default=None,
        help="Optional path to top-10 predictions JSON for ranking_depth_issue detection.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase2/analysis",
        help="Directory to write analysis outputs.",
    )
    args = parser.parse_args(argv)

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found: {}", pred_path)
        return 1

    logger.info("Loading predictions from {}", pred_path)
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    # Handle both a bare list and the wrapped {"predictions": [...]} format
    predictions: list[dict] = data if isinstance(data, list) else data.get("predictions", [])
    logger.info("Loaded {} prediction rows", len(predictions))

    top10: list[dict] | None = None
    if args.top10_predictions:
        t10_path = Path(args.top10_predictions)
        if t10_path.exists():
            raw = json.loads(t10_path.read_text(encoding="utf-8"))
            top10 = raw if isinstance(raw, list) else raw.get("predictions", [])
            logger.info("Loaded {} top-10 prediction rows", len(top10))
        else:
            logger.warning("top10-predictions file not found, skipping: {}", t10_path)

    experiment_name = pred_path.stem.replace("_predictions", "")

    analyzer = Phase2FailureAnalyzer(db_dir=args.db_dir)
    report = analyzer.analyze(
        predictions,
        top10_predictions=top10,
        experiment_name=experiment_name,
    )

    paths = save_analysis(report, args.output_dir)

    # Console summary
    print("\n" + "=" * 60)
    print(f"  Failure Analysis — {report.experiment_name}")
    print("=" * 60)
    print(f"  Predictions          : {report.n_predictions}")
    print(f"  With gold evidence   : {report.n_with_evidence}")
    print(f"  Perfect recall (1.0) : {report.n_perfect_recall}")
    print(f"  Partial recall       : {report.n_partial_recall}")
    print(f"  Zero recall          : {report.n_zero_recall}")
    print(f"  Source coverage      : {report.coverage.coverage_rate:.4f}")
    print()
    print("  Failure type breakdown:")
    for ft, cnt in sorted(report.failure_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ft:<30} {cnt:>5}")
    print()
    print("  Coverage by category:")
    for cat, cc in sorted(report.coverage.by_category.items()):
        print(
            f"    cat {cat:>3}  coverage={cc.coverage_rate:.3f}"
            f"  avg_recall={cc.avg_evidence_recall:.3f}"
        )
    print()
    print("  Output files:")
    for label, p in paths.items():
        print(f"    {label:<20} {p}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
