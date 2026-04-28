# Claude Build Brief: LoCoMo Naive RAG Benchmark First, SPARC-LTM Later

Use this Markdown file as the **single source of truth** for the project. The immediate task is to build the **naive LoCoMo RAG benchmark system**. The future task is to build the advanced memory system. Do not confuse the two.

---

## 0. How to Use This File With Limited Claude Tokens

Preferred method:

1. Add this file to the repository root as `CLAUDE.md` or `docs/CLAUDE_PROJECT_SPEC.md`.
2. In Claude, paste only the short command at the end of this file.
3. Ask Claude to read this file and implement **Phase 1 only**.
4. For future sessions, do not paste the full project again. Paste only:

```text
Read CLAUDE.md in the repo. Continue Phase 1: naive LoCoMo vector-RAG benchmark. Do not implement SPARC-LTM yet. Preserve all constraints and tests.
```

This saves tokens while keeping quality high.

---

## 1. Project Name

**Long-Horizon Conversational Memory for LLM Agents**

Repository:

```text
https://github.com/sahil0m/Centific-Hackathon.git
```

If the repository is empty, initialize the full project skeleton described below.

---

## 2. Big Picture Objective

We are building a memory layer for LLM agents that can persist useful information across many conversations. The final system should:

- remember what matters,
- forget or compress what does not matter,
- reconcile contradictions with provenance,
- retrieve the right evidence at the right time,
- stay within tight storage, token, latency, and API cost limits,
- expose what it remembered, forgot, and reconciled to the user.

The problem statement says naive approaches fail in two opposite ways:

1. Full-history prompting is expensive and hits token limits.
2. Plain vector retrieval misses nuance and breaks on multi-session reasoning.

The final advanced system will target only these two selected failure modes:

1. **Salience-aware forgetting under a hard storage cap**
2. **Contradiction reconciliation with provenance**

Failure modes 3 and 4 from the problem statement, namely intent-aware retrieval and temporal reasoning, are important background but are **not the main target of the advanced method for now**. They may appear in LoCoMo evaluation, but do not make them the central proposed contribution yet.

---

## 3. Immediate Task: Build the Naive RAG Benchmark First

The mentor specifically wants the project to start with a benchmark baseline.

Immediate goal:

> Build a professional, reproducible **naive vector-RAG baseline over LoCoMo message chunks**, run it end-to-end on LoCoMo QA, and produce benchmark scores.

Do **not** build the advanced memory system yet.

The baseline is required because later we need to prove that the proposed SPARC-LTM memory method improves over naive RAG.

The first research story is:

```text
Step 1: Build naive RAG baseline over LoCoMo.
Step 2: Measure benchmark scores.
Step 3: Analyze failures.
Step 4: Build advanced memory system targeting salience-aware forgetting and contradiction reconciliation.
Step 5: Run proposed method on the same benchmark slice.
Step 6: Compare baseline vs proposed method using metrics, probes, ablations, and external published numbers.
```

---

## 4. Non-Negotiable Problem Constraints

Follow these strictly.

### 4.1 Invalid Solution

A single LLM call with the full transcript stuffed into context is **not valid**.

Do not do:

```text
full conversation transcript + question -> LLM answer
```

### 4.2 Required Baseline

The baseline must be:

```text
vector-RAG over message chunks
```

### 4.3 Gold Evidence Rule

Do not use gold answers or gold evidence during retrieval or generation.

Valid:

```text
question -> retriever -> retrieved chunks -> LLM answer -> compare with gold answer/evidence
```

Invalid:

```text
question + gold evidence + gold answer -> LLM
```

### 4.4 Framework Restriction

Do not import, wrap, or depend on existing conversational-memory frameworks such as:

- Mem0
- LangMem
- Letta
- MemGPT
- A-MEM
- SimpleMem
- Zep
- Cognee
- similar memory frameworks

It is allowed to read/cite these systems later for literature review and published benchmark anchors.

### 4.5 Allowed Tools

Allowed:

- Python
- vector stores
- embedding models
- LLM clients
- FAISS
- sentence-transformers
- NumPy
- Pandas
- Pydantic/dataclasses
- Typer/argparse
- YAML configs
- pytest
- general orchestration libraries

---

## 5. What LoCoMo Is

LoCoMo is a long-horizon conversational memory benchmark. It contains multi-session conversations and QA annotations. The dataset usually includes:

- conversations,
- sessions,
- dialogue turns,
- speakers,
- timestamps,
- dialog IDs,
- questions,
- ground-truth answers,
- QA category,
- evidence dialog IDs when available.

Question categories can include:

- single-hop,
- multi-hop,
- temporal,
- commonsense / world knowledge,
- adversarial.

For Phase 1, focus on **LoCoMo QA** only.

Expected dataset path:

```text
data/raw/locomo10.json
```

The loader must be robust because the exact JSON shape may vary. It should inspect and normalize common fields instead of relying on one fragile schema.

---

## 6. Phase 1 System: Naive RAG Baseline

Pipeline:

```text
LoCoMo JSON
  -> Data Loader
  -> Chunk Builder
  -> Embedding Generator
  -> Vector Index
  -> Retriever
  -> Prompt Builder
  -> LLM Answer Generator
  -> Evaluator
  -> Metrics Report
```

### 6.1 Data Loader

The loader must normalize each conversation into:

```text
conversation_id / sample_id
sessions
messages / turns
QA examples
```

Each turn should include when available:

```text
sample_id
conversation_id
session_id
turn_index
dia_id / dialog_id
speaker
text
timestamp
```

Each QA item should include:

```text
qa_id
conversation_id
question
answer
category
gold_evidence_ids
```

The loader should:

- handle missing optional fields,
- produce clear error messages,
- log dataset statistics,
- support a small synthetic test fixture,
- never use gold evidence during retrieval/generation.

### 6.2 Chunk Builder

Implement these chunking strategies:

#### Strategy A: `turn`

One dialogue turn equals one chunk.

#### Strategy B: `window3`

Sliding window of 3 adjacent turns. Each chunk may contain multiple dialog IDs.

#### Strategy C: `session_summary`

Use session summaries if LoCoMo provides them. If unavailable, skip gracefully.

Each chunk must include:

```text
chunk_id
conversation_id
sample_id
session_id
turn_index_start
turn_index_end
dia_ids
speaker(s)
timestamp(s)
text
chunk_strategy
```

Recommended chunk text format:

```text
[Conversation: {conversation_id} | Session: {session_id} | Dialog IDs: {dia_ids}]
{speaker}: {text}
```

For window chunks:

```text
[Conversation: {conversation_id} | Session: {session_id} | Dialog IDs: D12,D13,D14]
SpeakerA: ...
SpeakerB: ...
SpeakerA: ...
```

### 6.3 Embeddings

Use a configurable embedding model.

Default:

```text
BAAI/bge-small-en-v1.5
```

Also support:

```text
BAAI/bge-base-en-v1.5
intfloat/e5-base-v2
```

Use `sentence-transformers` for local embeddings.

Requirements:

- normalize embeddings,
- batch embeddings,
- cache embeddings,
- cache key should depend on model name, chunk text, chunk strategy, and dataset/config hash when possible.

### 6.4 Vector Index

Use FAISS for Phase 1.

Requirements:

- build index per experiment,
- retrieve only from the same conversation/sample,
- use cosine similarity or inner product over normalized vectors,
- preserve chunk metadata,
- design abstraction so Qdrant/Milvus/pgvector can be added later.

### 6.5 Retriever

For each QA question:

1. Embed the question.
2. Search top-k chunks.
3. Restrict retrieval to the same conversation only.
4. Return chunks with metadata and scores.
5. Save retrieval output for debugging.

Run top-k experiments:

```text
top_k = 5
top_k = 10
top_k = 20
```

### 6.6 Prompt Builder

Use this prompt for answer generation:

```text
You are answering a question about a long multi-session conversation.

Rules:
1. Use only the retrieved conversation evidence.
2. If the evidence does not contain the answer, reply exactly: "No information available."
3. Give a short direct answer.
4. Do not explain your reasoning.
5. Do not mention evidence IDs in the final answer unless asked.

Retrieved evidence:
{retrieved_context}

Question:
{question}

Answer:
```

Optional debug JSON prompt:

```text
Return JSON only:
{
  "answer": "...",
  "used_evidence_ids": ["..."],
  "confidence": "low|medium|high"
}
```

For official scoring, save and evaluate only the answer string.

### 6.7 LLM Client

Use a configurable LLM client.

Support at least one of:

- Anthropic Claude API,
- OpenAI-compatible API,
- local Ollama/OpenAI-compatible endpoint.

Requirements:

- no hardcoded API keys,
- read secrets from `.env`,
- temperature 0,
- max output tokens around 100-150,
- retries with exponential backoff,
- request/response caching,
- save generation latency,
- save approximate token usage.

### 6.8 Evaluator

Required metrics:

1. Average token-level F1
2. Exact match
3. Category-wise F1
4. Evidence Recall@k when gold evidence IDs exist
5. Retrieval latency p50/p95
6. End-to-end latency p50/p95
7. Generation latency p50/p95
8. Average injected context tokens
9. Average output tokens
10. Cost estimate if API model is used
11. Number of retrieved chunks
12. Failure cases

F1 normalization:

- lowercase,
- remove punctuation,
- remove articles if appropriate,
- normalize whitespace,
- compute token overlap precision/recall/F1.

Evidence Recall@k:

```text
fraction of gold evidence IDs that appear in retrieved chunk metadata
```

If no gold evidence is available, skip evidence recall for that QA item.

---

## 7. Required Phase 1 Experiments

Create YAML configs for:

```text
naive_rag_turn_top5.yaml
naive_rag_turn_top10.yaml
naive_rag_turn_top20.yaml
naive_rag_window3_top5.yaml
naive_rag_window3_top10.yaml
naive_rag_session_summary_top5.yaml
```

If session summaries are not available, the session-summary experiment should skip gracefully and record why.

Optional but useful:

```text
last_session_only_baseline.yaml
```

---

## 8. Output Files

Save outputs as:

```text
results/
  raw_predictions/
    naive_rag_turn_top5.json
  retrieval/
    naive_rag_turn_top5_retrieval.json
  metrics/
    naive_rag_turn_top5_metrics.json
  tables/
    naive_rag_turn_top5_by_category.csv
    baseline_comparison.csv
    failure_cases.csv
  reports/
    naive_rag_failure_analysis.md
```

Each prediction row should include:

```json
{
  "experiment_name": "naive_rag_turn_top5",
  "conversation_id": "...",
  "qa_id": "...",
  "question": "...",
  "gold_answer": "...",
  "predicted_answer": "...",
  "category": "...",
  "gold_evidence_ids": [],
  "retrieved_chunks": [
    {
      "chunk_id": "...",
      "dia_ids": [],
      "session_id": "...",
      "speaker": "...",
      "text": "...",
      "score": 0.0
    }
  ],
  "f1": 0.0,
  "exact_match": false,
  "evidence_recall": null,
  "input_tokens": 0,
  "output_tokens": 0,
  "retrieval_latency_ms": 0.0,
  "generation_latency_ms": 0.0,
  "end_to_end_latency_ms": 0.0
}
```

---

## 9. Required Repository Structure

Create or update the repository to this structure:

```text
long-horizon-memory/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  configs/
    naive_rag_turn_top5.yaml
    naive_rag_turn_top10.yaml
    naive_rag_turn_top20.yaml
    naive_rag_window3_top5.yaml
    naive_rag_window3_top10.yaml
    naive_rag_session_summary_top5.yaml
  data/
    raw/
    processed/
  src/
    locomo_memory/
      __init__.py
      data/
        __init__.py
        load_locomo.py
        schemas.py
      indexing/
        __init__.py
        chunkers.py
        embeddings.py
        vector_index.py
      retrieval/
        __init__.py
        dense_retriever.py
      generation/
        __init__.py
        prompts.py
        llm_client.py
      evaluation/
        __init__.py
        qa_metrics.py
        evidence_recall.py
        report.py
      experiments/
        __init__.py
        run_rag_qa.py
  scripts/
    download_locomo.sh
    run_baseline.sh
    run_all_baselines.sh
  results/
    raw_predictions/
    retrieval/
    metrics/
    tables/
    reports/
  tests/
    test_loader.py
    test_chunker.py
    test_metrics.py
    test_evidence_recall.py
```

If the existing GitHub repo root is named differently, keep the existing repo name but use this internal structure.

---

## 10. Tech Stack

Use:

```text
Python 3.11+
pydantic or dataclasses
PyYAML
typer or argparse
sentence-transformers
faiss-cpu
numpy
pandas
tqdm
python-dotenv
tiktoken or fallback tokenizer
pytest
rich or standard logging
```

Optional:

```text
anthropic
openai
httpx
tenacity
```

---

## 11. CLI Requirement

This command must run the full pipeline:

```bash
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/naive_rag_turn_top5.yaml
```

It should perform:

```text
load data
-> chunk conversations
-> embed chunks
-> build/load FAISS index
-> retrieve top-k chunks
-> generate answers
-> evaluate
-> save predictions, retrievals, metrics, reports
```

Also create:

```bash
bash scripts/run_all_baselines.sh
```

---

## 12. Example YAML Config

```yaml
experiment:
  name: naive_rag_turn_top5
  seed: 42

dataset:
  name: locomo
  path: data/raw/locomo10.json
  split: all

chunking:
  strategy: turn
  window_size: 1
  include_speaker: true
  include_timestamp: true
  include_session_id: true

embedding:
  provider: sentence_transformers
  model_name: BAAI/bge-small-en-v1.5
  batch_size: 64
  normalize_embeddings: true
  cache_dir: data/processed/embedding_cache

retrieval:
  index_type: faiss
  top_k: 5
  same_conversation_only: true

generation:
  provider: anthropic
  model_name: claude-3-5-sonnet-latest
  temperature: 0
  max_output_tokens: 120
  cache_dir: data/processed/llm_cache
  enabled: true

evaluation:
  compute_f1: true
  compute_exact_match: true
  compute_evidence_recall: true
  compute_latency: true
  compute_token_usage: true

output:
  dir: results
```

Also support a retrieval-only mode for testing without paid LLM calls:

```yaml
generation:
  enabled: false
```

In retrieval-only mode, still save retrieved chunks and evidence recall.

---

## 13. Robustness Requirements

The implementation must be stable and professional:

1. Helpful error if dataset file is missing.
2. Helpful schema debug if expected fields are missing.
3. No hardcoded API keys.
4. Retry logic for LLM calls.
5. Embedding cache.
6. LLM output cache.
7. Deterministic seeds where possible.
8. Logs for every major pipeline stage.
9. Avoid duplicate chunk IDs.
10. Validate prediction rows before saving.
11. Save partial outputs safely.
12. Resume support where reasonable.
13. Do not crash if gold evidence IDs are missing.
14. Do not crash if session summaries are unavailable.
15. Keep modules small and easy to extend.
16. Do not leak secrets into result files.
17. Store config and config hash with every run.
18. Save run timestamp and git commit if available.

---

## 14. Tests Required

Add pytest tests for:

1. Dataset loader using a tiny synthetic LoCoMo-like JSON.
2. Turn chunker.
3. Window3 chunker.
4. F1 metric.
5. Exact match metric.
6. Evidence Recall@k.
7. Same-conversation-only retrieval logic.
8. Missing optional fields.

Tests must run without calling an external LLM.

---

## 15. README Requirements

Write a strong README explaining:

1. Project purpose.
2. What Phase 1 baseline does.
3. What Phase 1 intentionally does not do yet.
4. Setup instructions.
5. How to place/download LoCoMo.
6. How to configure API keys.
7. How to run one baseline.
8. How to run all baselines.
9. Output file explanations.
10. Metrics explanations.
11. How this baseline will later compare to the advanced memory system.

Include this statement in README:

```text
This is the naive vector-RAG baseline required before implementing the advanced long-horizon memory method. It treats all message chunks equally and retrieves by vector similarity only. Later, the proposed method will add salience-aware forgetting, compression under a hard storage cap, contradiction reconciliation with provenance, and a memory inspection UI.
```

---

## 16. Failure Analysis Report

Generate:

```text
results/reports/naive_rag_failure_analysis.md
```

The report should include:

1. Overall score.
2. Category-wise score.
3. Evidence Recall@k.
4. Examples where retrieval failed.
5. Examples where retrieval found evidence but generation failed.
6. Examples where temporal questions failed.
7. Examples where multi-hop questions failed.
8. Examples where adversarial questions failed.
9. Why naive RAG is insufficient.
10. How future SPARC-LTM will address failures.

Expected naive RAG failure categories:

- retrieved irrelevant chunks,
- missed gold evidence,
- retrieved only one part of multi-hop evidence,
- retrieved stale or contradictory evidence,
- answered despite insufficient evidence,
- failed temporal reasoning,
- too much noisy context,
- high token usage,
- no salience scoring,
- no contradiction policy,
- no provenance-aware memory lifecycle.

---

## 17. Future Advanced System Context: SPARC-LTM

This section is context only. Do not implement it in Phase 1, but keep the code extensible for it.

The final system will be called:

```text
SPARC-LTM: Salience and Provenance Aware Reconciliation and Compression for Long-Term Memory
```

SPARC means:

```text
S = Salience scoring
P = Provenance tracking
A = Adaptive compression / archiving
R = Reconciliation of contradictions
C = Context-controlled retrieval
```

### 17.1 Final Advanced Goal

Build a memory manager for an AI assistant that decides:

- what to keep active,
- what to compress,
- what to archive,
- what to mark as forgotten/inactive,
- what to reconcile when facts conflict,
- what evidence to show the user.

### 17.2 Memory Lifecycle

The proposed memory states are:

#### Active Memory

Recent or highly important facts that can be retrieved quickly and injected into the assistant context.

#### Compressed Memory

A compact label/summary plus pointer. Important note: compressed memory should behave like a **label**, not the full evidence. If a query needs the exact original content, the system should follow the pointer into archive/evidence store and promote the exact data back to active context.

#### Archive Memory

Exact recovery layer containing raw chunks, metadata, and provenance. This is for evidence, audit, and recovery. Under a true hard storage cap, archive may also require retention policies, summarization, or deletion/tombstones.

#### Forgotten / Inactive Memory

Memory that is not used in active retrieval. It may be expired, low-value, stale, superseded, or user-hidden. Depending on storage policy, it may be archived, tombstoned, or deleted.

### 17.3 Future Modules

The advanced system should eventually include:

1. Message Receiver
   - receives user messages and assistant responses.

2. Memory Router
   - decides whether to read from active, compressed, archived, or forgotten/inactive memory.

3. Salience Scorer
   - ranks memory by importance, frequency, recency, safety, stability, future usefulness, user pinning, and retrieval history.
   - Important: call it **Salience Scorer**, not Sentiment Scorer.

4. Compression Service
   - turns long, low-frequency memory into compact summaries with pointers to exact evidence.

5. Conflict Resolver
   - detects contradictions and changing facts.
   - keeps both fact and source.
   - marks status as active, superseded, conflicted, or historical.

6. Evidence Store
   - stores raw chunks, metadata, timestamps, dialog IDs, and provenance.

7. Response Guard
   - ensures the LLM answers only from approved evidence when memory claims are made.

8. Memory Inspection UI
   - shows what the system remembers, compressed, forgot, and reconciled.
   - lets users Keep, Compress, Forget, Delete, Pin, or Restore memories.

### 17.4 Future Salience Scoring

Future salience score should combine:

```text
importance
safety relevance
future usefulness
recency
frequency
retrieval history
user pinning
confidence
stability
uniqueness
storage cost
privacy sensitivity
staleness
redundancy
```

Potential formula:

```text
utility = salience_score / storage_cost
```

When memory reaches 80% capacity, future system should suggest compression/forgetting of low-utility memories.

### 17.5 Future Contradiction Reconciliation

Future system should distinguish:

```text
same fact
updated fact
contradiction
temporal change
related but not contradictory
ambiguous
```

Example:

```text
Session 1: I work at Google.
Session 10: I joined Microsoft.
```

Policy:

```text
Microsoft = current employer
Google = previous/superseded employer
Both keep provenance
```

Important correction:

```text
Session 1: I have surgery next week.
Session 5: I have a cold, suggest cold medicine.
```

This is **not automatically a contradiction**. It is related health context. The future memory system should retrieve surgery information for safety when answering about cold medicine, but it should not mark cold and surgery as contradictory.

### 17.6 Future UI

The UI should include:

- chat interface,
- memory inspection panel,
- active memories,
- compressed memories,
- archived memories,
- forgotten/inactive memories,
- conflict cards,
- evidence/provenance links,
- storage usage indicator,
- 80% storage warning,
- actions: Keep Full, Compress, Forget, Delete, Pin, Restore.

---

## 18. How Phase 1 Must Prepare for SPARC-LTM Without Implementing It

The baseline code should be modular enough that later we can replace naive chunks with structured memory units.

Design interfaces around:

```text
Document / Chunk
Retriever
Generator
Evaluator
Experiment Runner
```

Do not hardcode the assumption that all retrieval units are raw dialogue turns.

Later, SPARC-LTM will reuse:

- dataset loader,
- evaluator,
- metrics,
- experiment runner,
- result reporting,
- failure analysis,
- maybe vector index abstractions.

But Phase 1 must stay honest as naive RAG.

---

## 19. Benchmark Tables Needed Later

The final project will need tables like:

```text
Method | Memory Unit | Retrieval | Top-k | Avg F1 | Evidence Recall@k | Avg Tokens | p95 Latency
```

Start Phase 1 by producing rows for:

```text
Last-session only
Naive RAG turn top5
Naive RAG turn top10
Naive RAG turn top20
Naive RAG window3 top5
Naive RAG window3 top10
Naive RAG session summary top5, if available
```

Later rows:

```text
SPARC-LTM full
SPARC-LTM minus salience
SPARC-LTM minus conflict resolver
SPARC-LTM minus compression
SPARC-LTM under smaller storage caps
```

---

## 20. Implementation Quality Bar

Build like this will be judged by a mentor/reviewer.

Quality expectations:

- clean code,
- typed schemas,
- robust configs,
- repeatable runs,
- no hidden magic,
- clear logs,
- complete outputs,
- useful failure reports,
- tests,
- graceful handling of missing fields,
- easy extension to proposed method,
- no benchmark cheating,
- no full-transcript stuffing,
- no memory-framework imports.

---

## 21. First Claude Task

Implement Phase 1 only.

Deliver:

1. Full repository skeleton.
2. Code files.
3. Config files.
4. README.
5. Tests.
6. Shell scripts.
7. Example command.
8. Explanation of how to run.
9. Explanation of outputs.

The first working command must be:

```bash
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/naive_rag_turn_top5.yaml
```

If implementing in steps, start with:

1. schemas,
2. dataset loader,
3. chunkers,
4. metrics,
5. retrieval-only FAISS pipeline,
6. LLM generation,
7. reports,
8. tests,
9. README.

---

## 22. Short Prompt to Paste Into Claude

Paste this short prompt into Claude after adding this Markdown file to the repo:

```text
Read the project spec in CLAUDE.md / docs/CLAUDE_PROJECT_SPEC.md. Implement Phase 1 only: the professional naive LoCoMo vector-RAG benchmark system. Do not implement SPARC-LTM yet. Follow all constraints: no full-transcript stuffing, no use of gold evidence during generation, no memory frameworks, same-conversation-only retrieval, FAISS + sentence-transformers baseline, YAML configs, tests, metrics, reports, and the exact CLI command. Start by creating the repository skeleton and core modules, then implement the end-to-end retrieval-only pipeline before adding LLM generation.
```

---

## 23. Even Shorter Follow-Up Prompt for Later Claude Sessions

```text
Continue the repo using CLAUDE.md. Stay in Phase 1 naive LoCoMo RAG baseline. Run/fix tests, preserve constraints, and do not implement advanced memory yet.
```
