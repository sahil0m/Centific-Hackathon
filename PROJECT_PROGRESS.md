# Project Progress — Long-Horizon Conversational Memory
## Everything we have built, fixed, tested, and measured so far

---

## What Is This Project?

We are building a **memory system for AI assistants** that can remember things across many conversations over a long time.

The problem: current AI assistants forget everything after a conversation ends. If you tell the assistant "I work at Google" on Monday, it won't know that on Friday. We want to fix this.

**The full vision (Phase 2, not yet built):**
A smart memory that has three states:
- **Active** — facts the assistant knows right now and uses quickly
- **Compressed** — older facts summarized to save space
- **Forgotten/Archived** — very old or unimportant facts put away, can be recovered if needed

**What we are doing right now (Phase 1):**
Before building that smart memory, we need to prove the problem is real. So we are building the **simplest possible memory approach** — called Naive RAG — and measuring exactly where it fails. Then Phase 2 will fix those failures.

---

## What Is LoCoMo?

**LoCoMo** is a research dataset we use to test memory systems.

It contains:
- 10 long conversations between two people (e.g. Caroline and Melanie)
- Each conversation spans many sessions over many months
- Up to 35 sessions per conversation
- 1,986 questions about those conversations with gold (correct) answers
- 5 question types (called categories)

| Category | Type | Count |
|----------|------|-------|
| 1 | Single-hop (simple fact) | 282 |
| 2 | Multi-hop (needs connecting two facts) | 321 |
| 3 | Temporal (about dates/times) | 96 |
| 4 | Adversarial (tricky questions) | 841 |
| 5 | Adversarial open-ended | 446 |

**Dataset file location:** `data/raw/locomo10.json`

---

## Phase 1: What We Built — The Naive RAG System

### What is "RAG"?

RAG stands for **Retrieval-Augmented Generation**. The idea is simple:

```
Question comes in
    → Search the conversation for relevant pieces (retrieval)
    → Give those pieces + question to an LLM
    → LLM produces an answer
    → Compare answer with gold answer
```

This is "naive" because it treats every conversation turn equally — no smart memory, no forgetting, no salience scoring. Just basic search.

### The Rule We Follow (No Cheating)

We **never** give the LLM the gold answer or gold evidence during retrieval or generation. The system must find the answer on its own. This is the only valid way to measure performance.

---

## Full Codebase Structure

```
Centific-Hackathon/
│
├── src/locomo_memory/
│   ├── data/
│   │   ├── schemas.py          — Data types: Turn, QAItem, Chunk, PredictionRow
│   │   └── load_locomo.py      — Loads LoCoMo JSON into Python objects
│   │
│   ├── indexing/
│   │   ├── chunkers.py         — Splits conversation into searchable chunks
│   │   ├── embeddings.py       — Turns text into vectors using AI model
│   │   └── vector_index.py     — FAISS vector database, one per conversation
│   │
│   ├── retrieval/
│   │   ├── dense_retriever.py  — Basic vector search
│   │   ├── bm25_retriever.py   — Keyword search (BM25)
│   │   └── hybrid_retriever.py — Combines both with RRF fusion
│   │
│   ├── generation/
│   │   ├── prompts.py          — Prompt templates for the LLM
│   │   └── llm_client.py       — Calls Anthropic / OpenAI / Ollama
│   │
│   ├── evaluation/
│   │   ├── qa_metrics.py       — F1 score, Exact Match
│   │   ├── evidence_recall.py  — Evidence Recall@k metric
│   │   └── report.py           — Saves all results to files
│   │
│   └── experiments/
│       └── run_rag_qa.py       — Main pipeline, ties everything together
│
├── configs/                    — YAML files, one per experiment
├── tests/                      — 61 automated tests
├── scripts/                    — Shell scripts to run experiments
├── results/                    — All output files (gitignored)
└── data/raw/                   — LoCoMo dataset (gitignored)
```

---

## How the Pipeline Works (Step by Step)

```
Step 1: LOAD
  Read locomo10.json → parse into Conversation objects
  Each Conversation has: turns (dialogue) + QA items (questions + answers)

Step 2: CHUNK
  Split each conversation into small searchable pieces called "chunks"
  Strategy A — turn:    1 dialogue turn = 1 chunk
  Strategy B — window3: 3 adjacent turns = 1 chunk (overlapping)
  Strategy C — session_summary: use the pre-written session summaries

Step 3: EMBED
  Convert each chunk's text into a 384-dimensional vector
  Model: BAAI/bge-small-en-v1.5 (free, runs locally)
  Embeddings are cached on disk so we don't recompute every run

Step 4: INDEX
  Build one FAISS index per conversation
  IMPORTANT: retrieval only searches within the SAME conversation
  (we never mix up different people's conversations)

Step 5: RETRIEVE
  For each question: embed the question → search FAISS → get top-5 chunks

Step 6: GENERATE (optional, needs API key)
  Build a prompt with retrieved chunks + question → send to LLM → get answer

Step 7: EVALUATE
  Compare predicted answer vs gold answer
  Compute F1, Exact Match, Evidence Recall@k, latency

Step 8: SAVE
  Save all predictions, metrics, tables, and failure analysis to results/
```

---

## How to Run

```bash
# Install everything
pip install -e ".[dev]"

# Test pipeline without any API key (free)
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/retrieval_only_turn_top5.yaml

# Run with LLM generation (needs API key in .env)
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/naive_rag_turn_top5.yaml

# Run the improved hybrid pipeline
python -m locomo_memory.experiments.run_rag_qa \
  --config configs/hybrid_bm25_turn_context2_top5.yaml

# Run all baselines
bash scripts/run_all_baselines.sh

# Run all tests (no API key needed)
python -m pytest tests/ -v
```

---

## Metrics We Measure

| Metric | What It Means | Good Score |
|--------|--------------|-----------|
| **Evidence Recall@k** | Out of all gold evidence turns, what fraction did we find in top-k? | Higher = better |
| **Token F1** | Word overlap between our answer and gold answer | Higher = better |
| **Exact Match** | Does our answer exactly match the gold answer? | Higher = better |
| **Retrieval Latency p50/p95** | How fast is retrieval (milliseconds)? | Lower = better |

**Evidence Recall is the most important metric right now** because it tells us whether the right evidence was even retrieved. If recall is low, no LLM can fix it.

---

## Bugs We Found and Fixed

### Bug 1 — Unicode crash on Windows
**What happened:** Writing result files crashed with `UnicodeEncodeError` because Windows uses cp1252 encoding by default, which can't handle emoji characters in the LoCoMo dataset.

**Fix:** Added `encoding="utf-8"` to every `write_text()` call in `report.py`, `llm_client.py`, and `run_rag_qa.py`.

---

### Bug 2 — Wrong session IDs (all turns showed `session_id = "S0"`)
**What happened:** The LoCoMo JSON stores sessions as `session_1`, `session_2`, ... keys in a flat dictionary. Our generic loader didn't understand this format and assigned "S0" to all turns.

**What the actual format looks like:**
```json
{
  "conversation": {
    "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "..."}],
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_2": [...],
    ...
  }
}
```

**Fix:** Wrote a dedicated LoCoMo parser that detects the `session_N` key pattern and extracts session IDs and timestamps correctly.

---

### Bug 3 — Category 5 gold answers were empty
**What happened:** Category 5 questions use a field called `adversarial_answer` instead of `answer`. Our loader only looked for `answer`, so all 446 category-5 questions had empty gold answers. This made F1 = 1.0 for category 5 (both prediction and gold were empty → perfect match — completely wrong).

**Fix:** Added `adversarial_answer` as a fallback in the QA loader.

---

### Bug 4 — Some gold evidence IDs had semicolons inside one string
**What happened:** Some LoCoMo QA items stored two evidence IDs as one string: `"D8:6; D9:17"` instead of `["D8:6", "D9:17"]`. Our code treated this as one ID that could never match anything, making evidence recall = 0.

**Fix:** Added a `_split_evidence_ids()` function that splits on semicolons.

---

### Bug 5 — Session summary turns polluting the turn index
**What happened:** When we added session summaries as special turns (speaker = "summary"), they were accidentally included in the regular turn chunker. Summary chunks were getting retrieved instead of real dialogue turns, hurting recall.

**Example of what was happening:**
```
Q: "What did Caroline research?"
Gold: D2:8 (a real dialogue turn)
Retrieved: session_6_summary (a summary turn — wrong!)
```

**Fix:** Added `if turn.speaker.lower() == "summary": continue` to the turn and window3 chunkers. Summary turns are only used in the `session_summary` strategy.

---

### Bug 6 — Failure cases CSV saved even in retrieval-only mode
**What happened:** When generation is disabled, all predicted answers are empty strings, so F1 = 0.0 for everything. This caused the failure cases CSV to contain all 1,986 questions, which was misleading.

**Fix:** Skip saving failure cases when `generation_enabled = False`.

---

## All Experiments Run and Results

### Experiment 1 — Dense-Only Retrieval Baseline
**Config:** `retrieval_only_turn_top5.yaml`
**What it does:** Pure FAISS vector search, turn chunking, top-5, no LLM

**Results:**

| Category | Evidence Recall@5 |
|----------|------------------|
| single-hop (cat 1) | 0.307 |
| multi-hop (cat 2) | 0.608 |
| temporal (cat 3) | 0.271 |
| adversarial (cat 4) | 0.573 |
| adv-open-ended (cat 5) | 0.386 |
| **OVERALL** | **0.485** |

**Key finding — bimodal distribution:**
```
Perfect recall (found everything) : 875 questions = 44.1%
Zero recall    (found nothing)    : 910 questions = 45.9%
Partial recall                    :  ~10%
```
Almost no partial matches. The retriever either completely succeeds or completely fails. This tells us the problem is a **vocabulary mismatch**: the question uses abstract words, the answer uses specific words.

**Example failure:**
```
Question : "What did Caroline research?"       ← uses word "research"
Gold turn : "I've been looking into adoption agencies"  ← uses "looking into"
BGE-small can't bridge this gap → zero recall
```

**Latency:** p50 = 0.85ms, p95 = 1.24ms (very fast)

---

### Experiment 2 — Hybrid BM25 + Dense RRF + Context Window
**Config:** `hybrid_bm25_turn_context2_top5.yaml`
**What it adds:**
1. **BM25 keyword search** — finds exact word matches that dense search misses
2. **RRF fusion** — combines BM25 and dense rankings intelligently
3. **Context window = 2** — each chunk includes ±2 surrounding turns as `[Prior]`/`[Next]` context

**How RRF works (simple explanation):**
```
Dense search returns:   [D14:5, D2:8, D7:3, D9:1, ...]  ← ranked by similarity
BM25 search returns:    [D2:8, D14:5, D1:2, ...]         ← ranked by keyword match
RRF says: score = 1/(60 + rank) for each list
D2:8 appears at rank 2 in dense + rank 1 in BM25 → gets high combined score
Final list ranked by combined score → top-5 returned
```

**How contextual enrichment works:**
```
Without context (old):
  [Dialog ID: D2:8]
  Caroline: I've been looking into adoption agencies.

With context window=2 (new):
  [Dialog ID: D2:8]
  [Prior] Melanie: So what have you been up to?
  [Prior] Caroline: I've been doing a lot of research lately.
  Caroline: I've been looking into adoption agencies.    ← main turn
  [Next] Melanie: Oh wow, are you thinking of adopting?
  [Next] Caroline: Yes, I've always wanted to.
```
The embedding now sees the word "research" in the prior context → can match the question "What did Caroline research?"

**Results:**

| Category | Baseline | Hybrid | Change |
|----------|---------|--------|--------|
| single-hop (cat 1) | 0.307 | 0.246 | -0.062 |
| multi-hop (cat 2) | 0.608 | 0.565 | -0.043 |
| temporal (cat 3) | 0.271 | 0.233 | -0.038 |
| adversarial (cat 4) | 0.573 | 0.714 | **+0.140** |
| adv-open-ended (cat 5) | 0.386 | 0.669 | **+0.284** |
| **OVERALL** | **0.485** | **0.591** | **+0.106 (+21.8%)** |

| | Baseline | Hybrid | Change |
|--|---------|--------|--------|
| Perfect recall | 875 (44.1%) | 1090 (55.0%) | **+215 questions** |
| Zero recall | 910 (45.9%) | 708 (35.7%) | **-202 failures** |
| Latency p50 | 0.85ms | 4.69ms | +3.8ms |

**Why cat 1/2/3 got slightly worse:**
The context window adds ±2 turns around each chunk. For abstract/narrative questions (cat 4, 5) this helps a lot because the answer requires understanding context. For simple factual questions (cat 1) the context sometimes adds noise from adjacent turns about different topics.

The regression is **small** (only 14 questions in cat 1+3 went from perfect to zero) vs the **large gain** (215 questions now have perfect recall overall). Net: clearly better.

**Latency:** Still under 5ms p50 — fast enough for real use.

---

## Summary Table — All Results

| Experiment | Recall@5 | Perfect | Zero | Latency p50 |
|-----------|---------|---------|------|-------------|
| Dense only (baseline) | 0.485 | 44.1% | 45.9% | 0.85ms |
| Hybrid BM25 + ctx2 | **0.591** | **55.0%** | **35.7%** | 4.69ms |

---

## Tests

We have **61 automated tests** covering every module. They run without any LLM API key or internet connection.

```
tests/test_loader.py       — 8 tests  — data loading, missing fields, formats
tests/test_chunker.py      — 11 tests — turn, window3, session_summary strategies
tests/test_metrics.py      — 15 tests — F1, exact match, normalization, latency
tests/test_evidence_recall.py — 11 tests — recall calculation edge cases
tests/test_retriever.py    — 7 tests  — same-conversation isolation, top-k, batch
```

Run with: `python -m pytest tests/ -v`

---

## Config Files

| Config | Strategy | Top-k | Hybrid | Context | Generation |
|--------|----------|-------|--------|---------|-----------|
| `naive_rag_turn_top5.yaml` | turn | 5 | No | 0 | Yes (Claude) |
| `naive_rag_turn_top10.yaml` | turn | 10 | No | 0 | Yes |
| `naive_rag_turn_top20.yaml` | turn | 20 | No | 0 | Yes |
| `naive_rag_window3_top5.yaml` | window3 | 5 | No | 0 | Yes |
| `naive_rag_window3_top10.yaml` | window3 | 10 | No | 0 | Yes |
| `naive_rag_session_summary_top5.yaml` | session_summary | 5 | No | 0 | Yes |
| `retrieval_only_turn_top5.yaml` | turn | 5 | No | 0 | **No** |
| `hybrid_bm25_turn_context2_top5.yaml` | turn | 5 | **Yes** | **2** | **No** |

---

## Output Files Explained

After running an experiment, results are saved to `results/`:

```
results/
  raw_predictions/
    {experiment}.json          ← Every question: gold answer, predicted answer,
                                  retrieved chunks, F1, recall, latency, tokens

  retrieval/
    {experiment}_retrieval.json ← What chunks were retrieved for each question

  metrics/
    {experiment}_metrics.json  ← Summary: overall F1, recall, latency percentiles,
                                  per-category breakdown, config used, git commit

  tables/
    {experiment}_by_category.csv  ← Per-category scores in spreadsheet format
    baseline_comparison.csv       ← All experiments side by side
    failure_cases.csv             ← Questions where F1 < 0.3

  reports/
    naive_rag_failure_analysis.md ← Human-readable failure analysis with examples
```

---

## Why the Baseline Fails (Root Cause Analysis)

**The core problem in one sentence:**
> Naive RAG treats every old conversation turn equally and retrieves by surface similarity only — it has no concept of importance, no memory of what was reliable before, and no way to handle contradictions.

**Specific failure modes observed:**

1. **Vocabulary mismatch** — Questions use abstract words ("research"), answers use specific words ("looking into adoption agencies"). BGE-small can't always bridge this gap.

2. **Temporal blindness** — Temporal questions (cat 3, recall=0.27) fail because the retriever has no concept of "when". It can't find "the most recent time X happened" or order facts by date.

3. **Multi-hop limitations** — For questions that need two facts from different sessions, the retriever sometimes finds only one of them.

4. **No salience** — A turn from 2 years ago about a trivial topic gets the same weight as a turn from yesterday about something important.

5. **No contradiction handling** — If Caroline says "I work at Google" in session 1 and "I joined Microsoft" in session 10, both facts get retrieved equally. The LLM sees conflicting evidence with no guidance.

6. **Bimodal recall cliff** — 44% perfect, 46% zero. Almost no middle ground. This means even small improvements in the retriever design have large impact.

---

## What Comes Next

### Remaining Phase 1 improvements (retrieval upgrades)

**3. Cross-encoder reranker** (`BAAI/bge-reranker-base`)
- Retrieve top-30 candidates via hybrid
- Cross-encoder reads each (question, chunk) pair together → much more accurate relevance score
- Re-rank to top-5
- Expected gain: +5–8% recall
- Free, runs locally

**4. Multi-query retrieval**
- Generate 3 rephrased versions of each question
- Retrieve with each, union results, rerank
- Targets vocabulary mismatch directly
- Expected gain: +3–5% recall

**5. Better embedding model** (`BAAI/bge-large-en-v1.5`)
- 10× more parameters than bge-small
- Much better semantic understanding
- Expected gain: +5–8% recall

---

### Phase 2 — SPARC-LTM (The Real Goal)

SPARC-LTM = **S**alience and **P**rovenance **A**ware **R**econciliation and **C**ompression for **L**ong-**T**erm **M**emory

This is the advanced system we will build after the baseline is fully measured.

**The three memory states:**

| State | What it contains | When to use |
|-------|-----------------|-------------|
| **Active** | Recent facts, frequently accessed, high importance | Retrieve immediately |
| **Compressed** | Older facts summarized to short labels with pointers to full evidence | Retrieve summary; promote to active if needed |
| **Forgotten/Archived** | Low-salience, stale, superseded facts | Not retrieved by default; recoverable on demand |

**Key capabilities Phase 2 will add:**

| Capability | What it does |
|-----------|-------------|
| Salience Scorer | Rates every memory by importance, recency, frequency, safety, future usefulness |
| Compression Service | Shrinks low-frequency memories to summaries, saves storage |
| Conflict Resolver | Detects when facts contradict across sessions; keeps both with provenance |
| Provenance Tracker | Every answer traces back to the exact dialog ID that supports it |
| Memory Inspector UI | Shows what the system remembers, compressed, and forgot; lets user edit |

**Example of contradiction handling (Phase 2 target):**
```
Session 1:  "I work at Google."
Session 10: "I just joined Microsoft."

Phase 1 (naive): Both facts retrieved, LLM confused, wrong answer likely.

Phase 2 (SPARC-LTM):
  Microsoft = ACTIVE (current employer, high salience, recent)
  Google    = COMPRESSED (previous employer, superseded, with provenance)
  Answer uses Microsoft, notes Google as previous with source session IDs.
```

---

## Key Numbers to Remember

| What | Number |
|------|--------|
| Conversations in LoCoMo | 10 |
| Total QA questions | 1,986 |
| Total dialogue turns | ~5,882 (excluding summaries) |
| Session summaries available | 5 per conversation |
| Tests passing | 61 / 61 |
| Baseline recall@5 | 0.485 |
| Best recall@5 so far | **0.591** (hybrid BM25 + ctx2) |
| Retrieval latency (hybrid) | 4.69ms p50 |
| Questions fixed by hybrid | +215 perfect recall |
| Questions still failing | 708 zero-recall |

---

## Git History

| Commit | What it contains |
|--------|-----------------|
| `cb71301` | Full Phase 1 implementation — all modules, configs, 61 tests |

**Remote:** `https://github.com/sahil0m/Centific-Hackathon.git` (branch: `main`)

---

*Last updated: after Experiment 2 (hybrid BM25 + contextual enrichment)*
