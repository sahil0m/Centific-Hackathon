#!/usr/bin/env bash
# Download LoCoMo dataset to data/raw/
# The official dataset is available from the LoCoMo paper authors.
# Replace the URL below with the actual download link when available.
set -euo pipefail

OUTDIR="data/raw"
mkdir -p "$OUTDIR"

echo "LoCoMo dataset download instructions:"
echo "======================================="
echo ""
echo "Option A — Hugging Face (if published):"
echo "  pip install huggingface_hub"
echo "  python -c \""
echo "    from huggingface_hub import hf_hub_download"
echo "    hf_hub_download(repo_id='<author>/locomo', filename='locomo10.json', local_dir='data/raw')"
echo "  \""
echo ""
echo "Option B — Manual download:"
echo "  1. Request access from the LoCoMo paper authors."
echo "  2. Place the file at: data/raw/locomo10.json"
echo ""
echo "Option C — Use the synthetic fixture for testing (no download needed):"
echo "  python -c \""
echo "    from locomo_memory.data.load_locomo import make_synthetic_locomo"
echo "    import json"
echo "    convs = make_synthetic_locomo(n_conversations=10)"
echo "    # The fixture is used directly in tests without a JSON file."
echo "  \""
echo ""
echo "After placing the file, verify with:"
echo "  python -c \""
echo "    from locomo_memory.data.load_locomo import load_locomo"
echo "    convs = load_locomo('data/raw/locomo10.json')"
echo "    print(f'Loaded {len(convs)} conversations')"
echo "  \""
