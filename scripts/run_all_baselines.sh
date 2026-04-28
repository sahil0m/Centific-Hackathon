#!/usr/bin/env bash
# Run all Phase 1 baseline experiments in sequence.
# Embeddings are cached after the first run, so subsequent runs are faster.
set -euo pipefail

CONFIGS=(
  "configs/naive_rag_turn_top5.yaml"
  "configs/naive_rag_turn_top10.yaml"
  "configs/naive_rag_turn_top20.yaml"
  "configs/naive_rag_window3_top5.yaml"
  "configs/naive_rag_window3_top10.yaml"
  "configs/naive_rag_session_summary_top5.yaml"
)

FAILED=()

for cfg in "${CONFIGS[@]}"; do
  echo "========================================"
  echo "Running: $cfg"
  echo "========================================"
  if python -m locomo_memory.experiments.run_rag_qa --config "$cfg"; then
    echo "SUCCESS: $cfg"
  else
    echo "FAILED: $cfg"
    FAILED+=("$cfg")
  fi
  echo ""
done

echo "========================================"
echo "All baselines complete."
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED configs:"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
  exit 1
else
  echo "All experiments succeeded."
fi

echo ""
echo "Outputs saved to:"
echo "  results/raw_predictions/"
echo "  results/retrieval/"
echo "  results/metrics/"
echo "  results/tables/"
echo "  results/reports/"
