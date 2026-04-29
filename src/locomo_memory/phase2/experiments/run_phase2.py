"""Phase 2 CLI entry point — Milestone 10.

Usage::

    python -m locomo_memory.phase2.experiments.run_phase2 \\
        --config configs/phase2_retrieval_only.yaml

The script loads the config, creates a Phase2LoCoMoRunner (with the real
sentence-transformers embed function), and calls runner.run().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from locomo_memory.phase2.experiments.config import Phase2RunnerConfig
from locomo_memory.phase2.experiments.runner import Phase2LoCoMoRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 SPARC-LTM LoCoMo evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/phase2_retrieval_only.yaml)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Skip saving result files (useful for quick sanity checks)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: {}", config_path)
        return 1

    logger.info("Loading config from {}", config_path)
    config = Phase2RunnerConfig.from_yaml(config_path)
    logger.info("Experiment: {}", config.experiment_name)

    runner = Phase2LoCoMoRunner(config)
    result = runner.run(save=not args.no_save)

    # Print summary
    m = result.metrics
    if m:
        print("\n" + "=" * 60)
        print(f"  Phase 2 — {config.experiment_name}")
        print("=" * 60)
        print(f"  Conversations : {result.n_conversations}")
        print(f"  Predictions   : {result.n_predictions}")
        print(f"  Avg F1        : {m.avg_f1:.4f}")
        print(f"  Exact Match   : {m.exact_match_rate:.4f}")
        if m.avg_evidence_recall is not None:
            print(f"  Evidence R@k  : {m.avg_evidence_recall:.4f}")
        print(f"  Grounding     : {m.avg_grounding_score:.4f}")
        print(f"  Guard pass    : {m.guard_pass_rate:.4f}")
        print(f"  Retrieval p50 : {m.retrieval_latency_p50:.1f}ms")
        print(f"  Retrieval p95 : {m.retrieval_latency_p95:.1f}ms")
        print("=" * 60)

        if m.by_category:
            print("\n  By category:")
            for cat, cm in sorted(m.by_category.items()):
                print(
                    f"    cat {cat:>3}  n={cm['count']:>4}  "
                    f"F1={cm['avg_f1']:.3f}  "
                    f"R@k={cm.get('avg_evidence_recall') or 'N/A'}"
                )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
