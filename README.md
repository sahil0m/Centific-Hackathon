# Long-Horizon Conversational Memory — Phase 2: SPARC-LTM

This repository implements **SPARC-LTM** (Salience and Provenance Aware Reconciliation and Compression for Long-Term Memory) — a production-grade memory layer for LLM agents — alongside the **Phase 1 naive vector-RAG baseline** that anchors its evaluation.

For full design details, see [`METHODOLOGY.md`](METHODOLOGY.md). For the original build brief, see [`CLAUDE.md`](CLAUDE.md).

---

## Quick Start (Phase 2 — SPARC-LTM Streamlit demo)

```bash
# 1. Install
pip install -e .

# 2. Provide an OpenRouter API key in .env
cp .env.example .env
# edit .env to add OPENROUTER_API_KEY=sk-or-...

# 3. Launch the interactive memory demo
streamlit run app.py
```

In the Streamlit UI:
- **Click "🚀 Demo (cap=10)"** to enable demo-mode lifecycle thresholds (visible in ~2 minutes)
- Send statements ("I work at Centific", "I live in Hyderabad") → they enter Active memory
- Watch the memory bar populate Active / Compressed / Archived / Forgotten layers
- Send updates ("I moved to Mumbai") → see automatic supersession with provenance
- Open **Memory Inspector** to browse every fact + its full provenance trail

## Quick Start (Phase 2 — full LoCoMo evaluation)

```bash
python -m locomo_memory.experiments.run_full_system_eval \
  --config configs/phase2_full_system_eval.yaml
```

Outputs land in `results/phase2/`. See [§14 of METHODOLOGY.md](METHODOLOGY.md#14-comparison-with-the-phase-1-naive-rag-baseline) for what this measures.

## Quick Start (Phase 1 — naive RAG baseline)

```bash
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/naive_rag_turn_top5.yaml
```

The Phase 1 baseline is what Phase 2 is measured against. Both ship in this repo.

---

## What Phase 1 Does



## What Phase 1 Does

| Component | Details |
|-----------|---------|
| **Data loader** | Normalizes LoCoMo JSON into typed `Conversation`, `Turn`, `QAItem` objects — handles multiple JSON shapes |
| **Chunking** | Three strategies: `turn` (1 turn = 1 chunk), `window3` (sliding window of 3), `session_summary` (if present) |
| **Embeddings** | `sentence-transformers` with disk cache — default `BAAI/bge-small-en-v1.5` |
| **Vector index** | FAISS per conversation — retrieval always scoped to the same conversation |
| **Retriever** | Top-k dense retrieval with latency measurement |
| **Prompt builder** | Structured prompt with retrieved evidence — no gold answers or evidence injected |
| **LLM client** | Anthropic / OpenAI / Ollama — cached, retried, temperature=0 |
| **Evaluator** | Token F1, Exact Match, Evidence Recall@k, per-category breakdown, latency percentiles |
| **Reports** | Predictions JSON, retrieval debug JSON, metrics JSON, category CSV, failure analysis Markdown |

## What Phase 1 Does NOT Do

- Full-transcript stuffing into context (invalid baseline)
- Use gold evidence or gold answers during retrieval or generation
- Import memory frameworks (Mem0, LangMem, Letta, etc.)
- Implement SPARC-LTM (Phase 2 — salience scoring, contradiction reconciliation, compression, memory UI)

---

## Repository Structure

```
Centific-Hackathon/
  configs/
    naive_rag_turn_top5.yaml
    naive_rag_turn_top10.yaml
    naive_rag_turn_top20.yaml
    naive_rag_window3_top5.yaml
    naive_rag_window3_top10.yaml
    naive_rag_session_summary_top5.yaml
    retrieval_only_turn_top5.yaml      # no LLM needed — for pipeline testing
  data/
    raw/                               # place locomo10.json here
    processed/
      embedding_cache/                 # auto-filled after first run
      llm_cache/                       # auto-filled after first run
  src/
    locomo_memory/
      data/
        schemas.py                     # typed dataclasses
        load_locomo.py                 # robust loader + synthetic fixture
      indexing/
        chunkers.py                    # turn / window3 / session_summary
        embeddings.py                  # EmbeddingGenerator with cache
        vector_index.py                # FAISS per-conversation index
      retrieval/
        dense_retriever.py             # DenseRetriever
      generation/
        prompts.py                     # prompt templates
        llm_client.py                  # LLMClient (Anthropic / OpenAI / Ollama)
      evaluation/
        qa_metrics.py                  # F1, EM, latency
        evidence_recall.py             # Evidence Recall@k
        report.py                      # save predictions, metrics, CSV, Markdown
      experiments/
        run_rag_qa.py                  # main CLI entry point
  scripts/
    download_locomo.sh
    run_baseline.sh
    run_all_baselines.sh
  tests/
    test_loader.py
    test_chunker.py
    test_metrics.py
    test_evidence_recall.py
    test_retriever.py
  results/
    raw_predictions/
    retrieval/
    metrics/
    tables/
    reports/
  pyproject.toml
  .env.example
```

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Place the LoCoMo dataset

```bash
# See download instructions:
bash scripts/download_locomo.sh

# After downloading, verify:
python -c "
from locomo_memory.data.load_locomo import load_locomo
convs = load_locomo('data/raw/locomo10.json')
print(f'Loaded {len(convs)} conversations')
"
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY for OpenAI)
```

API keys are read from `.env` via `python-dotenv`. They are **never** hardcoded or saved to result files.

---

## Running Experiments

### Run one baseline

```bash
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/naive_rag_turn_top5.yaml
```

### Test the pipeline without LLM calls (no API key needed)

```bash
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/retrieval_only_turn_top5.yaml
```

### Run all baselines

```bash
bash scripts/run_all_baselines.sh
```

Embeddings are cached after the first run. Subsequent experiments using the same model and chunks reuse the cache.

---

## Experiments

| Config | Chunking | Top-k | LLM |
|--------|----------|-------|-----|
| `naive_rag_turn_top5` | turn | 5 | Claude |
| `naive_rag_turn_top10` | turn | 10 | Claude |
| `naive_rag_turn_top20` | turn | 20 | Claude |
| `naive_rag_window3_top5` | window (3 turns) | 5 | Claude |
| `naive_rag_window3_top10` | window (3 turns) | 10 | Claude |
| `naive_rag_session_summary_top5` | session summary | 5 | Claude |
| `retrieval_only_turn_top5` | turn | 5 | None |

> **Session summary note**: LoCoMo may not contain session summaries. If absent, the experiment logs a warning and saves an empty run record — no crash.

---

## Output Files

```
results/
  raw_predictions/naive_rag_turn_top5.json        # full prediction rows
  retrieval/naive_rag_turn_top5_retrieval.json    # retrieved chunks per question
  metrics/naive_rag_turn_top5_metrics.json        # all metrics + config hash
  tables/naive_rag_turn_top5_by_category.csv      # per-category F1
  tables/baseline_comparison.csv                  # cross-experiment summary
  tables/failure_cases.csv                        # low-F1 cases
  reports/naive_rag_failure_analysis.md           # failure analysis report
  logs/naive_rag_turn_top5.log                    # full debug log
```

Each prediction row includes:
- `f1`, `exact_match`, `evidence_recall`
- `retrieved_chunks` with chunk text and scores
- `input_tokens`, `output_tokens`
- `retrieval_latency_ms`, `generation_latency_ms`, `end_to_end_latency_ms`

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Avg Token F1** | Token-level overlap between prediction and gold answer (SQuAD-style) |
| **Exact Match** | Fraction of predictions that exactly match the normalized gold answer |
| **Evidence Recall@k** | Fraction of gold evidence dialog IDs found in the top-k retrieved chunks |
| **Retrieval latency p50/p95** | Retrieval speed percentiles in milliseconds |
| **End-to-end latency p50/p95** | Total per-question latency |
| **Avg input/output tokens** | Context window usage and generation length |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run without any LLM calls or external downloads — they use the synthetic LoCoMo fixture built into the loader.

---

## Benchmark Table (to be filled after running)

| Method | Chunking | Top-k | Avg F1 | Evidence Recall@k | Avg Tokens | p95 Latency |
|--------|----------|-------|--------|-------------------|------------|-------------|
| Naive RAG | turn | 5 | TBD | TBD | TBD | TBD |
| Naive RAG | turn | 10 | TBD | TBD | TBD | TBD |
| Naive RAG | turn | 20 | TBD | TBD | TBD | TBD |
| Naive RAG | window3 | 5 | TBD | TBD | TBD | TBD |
| Naive RAG | window3 | 10 | TBD | TBD | TBD | TBD |
| SPARC-LTM (Phase 2) | — | — | — | — | — | — |

---

## Roadmap

**Phase 1 (this repo):** Naive vector-RAG baseline — establish benchmark scores, identify failure modes.

**Phase 2 — SPARC-LTM** *(not yet implemented)*:

- **S** — Salience scoring: rank memory by importance, recency, frequency, future usefulness
- **P** — Provenance tracking: every fact is tied to its source dialog ID and session
- **A** — Adaptive compression: compact low-frequency memories to summaries with pointers
- **R** — Reconciliation: detect and resolve contradictions with full provenance
- **C** — Context-controlled retrieval: retrieve only salience-approved evidence

The Phase 2 system will reuse this repo's data loader, evaluator, metrics, and experiment runner, allowing a direct apples-to-apples comparison on the same LoCoMo benchmark.
