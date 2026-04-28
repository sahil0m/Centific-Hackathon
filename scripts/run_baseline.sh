#!/usr/bin/env bash
# Run a single experiment baseline.
# Usage: bash scripts/run_baseline.sh configs/naive_rag_turn_top5.yaml
set -euo pipefail

CONFIG="${1:-configs/naive_rag_turn_top5.yaml}"

echo "Running experiment: $CONFIG"
python -m locomo_memory.experiments.run_rag_qa --config "$CONFIG"
echo "Done."
