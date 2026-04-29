# SPARC-LTM Phase 2 Methodology

**SPARC-LTM** — Salience and Provenance Aware Reconciliation and Compression for Long-Term Memory

This document describes exactly what is implemented in the Phase 2 codebase.
Every detail maps directly to actual code — nothing is assumed or extrapolated.

---

## 1. Architecture Overview

Every user message passes through an ingestion pipeline that extracts atomic facts,
scores them, stores them, detects contradictions, indexes them, and manages capacity.
Every question passes through a parallel retrieval pipeline that searches three memory
states simultaneously, merges results, promotes retrieved forgotten memories, then
generates a grounded answer.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                           │
│                                                                       │
│  raw text → FactExtractor → SalienceScorer → SQLite store            │
│                → ContradictionResolver → FAISS + BM25 index          │
│                → LifecycleEngine (auto compress / forget at 90%)     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL PIPELINE                           │
│                                                                       │
│  question ─┬─ Worker 1: FAISS dense over ACTIVE                      │
│             ├─ Worker 2: BM25 sparse over ACTIVE                     │
│             ├─ Worker 3: FAISS over COMPRESSED labels                │
│             │            → pointer follow → ARCHIVE raw text         │
│             └─ Worker 4: BM25 over FORGOTTEN                         │
│                          → auto-promote hits to ACTIVE               │
│                                                                       │
│  RRF fusion → top-k → ContextBuilder → LLM answer                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Memory States

Five states are defined in `MemoryStatus` (str enum):

| State | Meaning | In retrieval indexes |
|-------|---------|----------------------|
| `active` | Working memory. All new facts start here. | FAISS dense + BM25 |
| `compressed` | Low-salience. LLM summary label stored. Raw data in archive. | Label FAISS only |
| `archived` | Backend storage for raw compressed data. Not a MU status — stored in `archived_entries` table. | Not directly searched |
| `forgotten` | Very low salience. Removed from all hot indexes. | Searched by Worker 4 on demand |
| `deleted` | Terminal. Content nulled out. Audit row written. | Never returned |

---

## 3. Core Data Schemas

### 3.1 MemoryUnit

The atomic unit of memory. Every extracted fact becomes one `MemoryUnit`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mu_id` | str | `mu_{12-hex}` | Primary key |
| `conversation_id` | str | required | Scope key — all retrieval is scoped to this |
| `session_id` | str | required | Session the fact came from |
| `claim` | str | required | The extracted atomic fact |
| `original_text` | str | `""` | Raw source text before extraction |
| `source_dia_ids` | list[str] | `[]` | Source dialogue turn IDs |
| `source_speaker` | str | `""` | Speaker name |
| `timestamp` | str\|None | None | Source turn timestamp |
| `salience_score` | float [0,1] | 0.5 | Current salience (recomputed by scorer) |
| `importance` | float [0,1] | 0.5 | Topic importance (set at extraction time) |
| `recency_weight` | float [0,1] | 1.0 | Recency decay factor |
| `uniqueness` | float [0,1] | 1.0 | Distinctiveness |
| `retrieval_count` | int | 0 | Times this fact was retrieved |
| `prompt_frequency` | float [0,1] | 0.0 | Frequency in prompts |
| `last_accessed` | datetime\|None | None | Last retrieval timestamp |
| `status` | MemoryStatus | `active` | Current lifecycle state |
| `confidence` | float [0,1] | 0.9 | Extraction confidence |
| `needs_reindex` | bool | False | Flag for FAISS sync after restore |
| `compressed_label_id` | str\|None | None | Pointer to CompressedLabel (set when compressed) |
| `archived_entry_id` | str\|None | None | Pointer to ArchivedEntry (set when compressed) |
| `user_pinned` | bool | False | Pinned MUs are never auto-transitioned by lifecycle |

### 3.2 CompressedLabel

Created when a MemoryUnit is compressed. Stored in the `compressed_labels` table.
This is what gets embedded and searched by Worker 3.

| Field | Type | Description |
|-------|------|-------------|
| `label_id` | str | `lbl_{12-hex}` |
| `archived_pointer` | str | Points to the ArchivedEntry |
| `mu_id` | str | Which MemoryUnit this compresses |
| `conversation_id` | str | Scope key |
| `topic` | str | Rule-detected topic category |
| `short_summary` | str | LLM-generated dense summary sentence (≤130 tokens) |
| `key_entities` | list[str] | Regex-extracted capitalized named entities |
| `time_range` | str\|None | Timestamp carried from source MU |
| `original_dia_ids` | list[str] | Source dialogue IDs |
| `retrieval_count` | int | Times this label matched a query |

### 3.3 ArchivedEntry

Created atomically alongside every CompressedLabel. Stored in `archived_entries` table.
This is the lossless raw data recovery layer.

| Field | Type | Description |
|-------|------|-------------|
| `archived_entry_id` | str | `arc_{12-hex}` |
| `label_pointer` | str | Points back to the CompressedLabel |
| `mu_id` | str | Which MemoryUnit was archived |
| `conversation_id` | str | Scope key |
| `full_memory_unit_json` | str | Complete serialized MemoryUnit (Pydantic model_dump_json) |
| `full_original_text` | str | Verbatim source text |
| `restoration_count` | int | Times full content was loaded from archive |

### 3.4 EdgeRecord

Provenance edges between MemoryUnits. Stored in `edges` table.
Unique constraint on `(source_mu_id, target_mu_id, edge_type)`.

| Edge Type | Meaning |
|-----------|---------|
| `superseded_by` | old fact → new fact (job changed, location changed, fact updated) |
| `conflicts_with` | bidirectional contradiction between two facts |
| `related_to` | topically related but not contradictory |
| `derived_from` | fact derived or inferred from another |

---

## 4. Ingestion Pipeline

Entry point: `SystemEngine.process_message(text, speaker, session_id)`

### Step 1 — Wrap into Chunk

`_make_chunk` formats the raw text as:

```
[Conversation: {conversation_id} | Session: {session_id}]
{speaker}: {text}
```

A random `dia_id` (`D` + 6 hex chars) and UTC timestamp are generated.
`chunk_strategy` is set to `"live"`.

### Step 2 — Fact Extraction (`FactExtractor`)

**Model:** `anthropic/claude-3-haiku` (via OpenRouter)
**Parameters:** `temperature=0.0`, `max_output_tokens=512`, `max_facts_per_chunk=7`, `confidence=0.9`

The LLM is prompted to extract atomic claims and return strict JSON:

```json
{"facts": [{"claim": "...", "speaker": "..." | null, "source_dia_id": "..." | null}]}
```

Instructions given to the LLM:
- Resolve pronouns to proper nouns
- Skip questions and opinion hedges ("I think", "maybe", "perhaps")
- One atomic claim per fact entry

**Fallback (if LLM fails):** Heuristic sentence splitter drops questions, opinions, and fragments shorter than 10 characters. Returns up to 7 facts with `confidence=0.5`.

**Provenance resolution:**
If the LLM-suggested `source_dia_id` matches a turn in the chunk, the fact is attributed to that specific turn (speaker and timestamp narrowed). Otherwise the full chunk dia_id list and joined speakers are used.

**Output:** `ExtractionResult` with a list of `MemoryUnit` objects, all `status=ACTIVE`.

### Step 3 — Salience Scoring (`SalienceScorer`)

Called immediately after extraction: `scorer.score_and_update(mu)` writes `mu.salience_score`.

**6 sub-scores** combined via weighted sum:

| Dimension | Weight | Formula |
|-----------|--------|---------|
| importance | 0.30 | `mu.importance` (set by TopicImportanceEstimator during extraction) |
| confidence | 0.15 | `mu.confidence` |
| recency | 0.20 | `exp(-k × days_since_last_access_or_creation)` where `k = ln(2) / 30` (half-life = 30 days) |
| retrieval_frequency | 0.15 | `retrieval_count / (retrieval_count + 10)` saturation curve |
| user_pinned | 0.10 | 1.0 if pinned, 0.0 otherwise |
| uniqueness | 0.10 | `mu.uniqueness` |

**Final salience:** `clamp(weighted_sum / sum_of_weights)` in `[0.0, 1.0]`

**Utility score** (used internally by LifecycleEngine): `salience / max(1.0, len(claim) / 100)`

### Step 4 — Persist to SQLite

`store.insert_memory_unit(mu)` — WAL mode, single atomic write per MU.

### Step 5 — Contradiction Resolution (`ContradictionResolver`)

Each new MU is compared against all existing active MUs in the same conversation.

**Comparison rules (applied in priority order):**

| Relationship | Detection rule | Edge confidence |
|-------------|---------------|-----------------|
| `SAME_FACT` | Jaccard token overlap ≥ 0.70 | `min(1.0, jaccard)` |
| `CONTRADICTION` | Negation word in new MU AND (jaccard ≥ 0.25 OR entity overlap ≥ 2) AND (entity overlap ≥ 1 OR same topic) | `0.6 + 0.1×min(entity_overlap,3) + 0.1×same_topic` |
| `UPDATED_FACT` | Update verb in new MU ("joined", "moved to", "is now", "started at", "got married", "graduated from"…) AND (same topic OR entity overlap ≥ 1) | 0.75 |
| `TEMPORAL_CHANGE` | Temporal marker in new MU ("used to", "previously", "formerly", "back when", "until recently"…) AND (same topic OR entity overlap ≥ 1) | 0.70 |
| `RELATED` | Same topic OR jaccard ≥ 0.10 | `max(jaccard, 0.3 if same_topic else 0.0)` |
| `UNRELATED` | None of the above | `max(0.0, 1.0 − jaccard)` |

**Tokenization:** lowercase, strip punctuation, remove 60+ stop words.

**Edge creation policy:**

| Relationship | Edge written |
|-------------|-------------|
| SAME_FACT | old_mu → new_mu: `SUPERSEDED_BY` |
| UPDATED_FACT | old_mu → new_mu: `SUPERSEDED_BY` |
| TEMPORAL_CHANGE | old_mu → new_mu: `SUPERSEDED_BY` |
| CONTRADICTION | both directions: `CONFLICTS_WITH` |
| RELATED | old_mu → new_mu: `RELATED_TO` |
| UNRELATED | no edge written |

Duplicate edges (same source, target, type) are silently skipped.

### Step 6 — Index Update

```python
faiss_index.add_mu(mu)   # adds embedding to dense FAISS
bm25_index.add_mu(mu)    # adds tokens to BM25
```

### Step 7 — Lifecycle (LifecycleEngine)

Called once per `process_message` call via `lifecycle.maybe_run(conversation_id)`.

**Trigger:** pressure = `active_count / active_cap ≥ 0.90`
**Active cap:** 100 (configured in SystemEngine)
**Target:** run until pressure drops below 0.70

**Selection:** ranks all active non-pinned MUs by salience ascending (lowest first).
Transitions the required number to bring pressure below target:

| Salience | Transition |
|----------|-----------|
| `< 0.15` | MU → `FORGOTTEN` (via `forget_atomic`) |
| `0.15 – 0.40` | MU → `COMPRESSED` (via `compress_atomic`, see Section 5) |
| `≥ 0.40` or `user_pinned=True` | No change |

**After any transitions:** all indexes fully rebuilt:
```python
faiss_index.rebuild_from_store(store, conversation_id)
bm25_index.rebuild_from_store(store, conversation_id)
label_index.rebuild_from_store(store)
graph.rebuild_from_store(store)
```

---

## 5. Compression Pipeline

When a MU is compressed, two records are written inside a single SQLite transaction (`compress_atomic`). The transaction inserts the archive row, inserts the label row, then updates the MU status with both pointers. All three writes succeed together or none.

### 5.1 LLM Label Generation (`LLMLabeler`)

**Model:** `anthropic/claude-3-haiku`, `temperature=0.0`, `max_tokens=130`
**Cache key:** `SHA256(mu_id + claim)[:20]`
**Prompt template version:** `compress_label_v1`

The LLM receives the original claim and source text (truncated at 400 chars) and is instructed to:
- Write a single dense sentence
- Preserve all key info: full names, locations, dates, numbers, roles, relationships
- State the type of fact (employment, preference, health, event, personal info, etc.)
- Make it searchable for future queries
- Return only the sentence — no prefix, no explanation

**Fallback:** if LLM call fails or returns fewer than 10 characters → `claim[:120]`

### 5.2 What Gets Written Atomically

**`compressed_labels` row:**
- `short_summary` = LLM-generated label sentence
- `topic` = rule-detected topic from TopicImportanceEstimator
- `key_entities` = regex-extracted capitalized names from original claim
- `archived_pointer` = ID of the corresponding ArchivedEntry
- `label_id` = points back from MemoryUnit

**`archived_entries` row:**
- `full_memory_unit_json` = complete Pydantic JSON of the MemoryUnit at compression time
- `full_original_text` = verbatim source text
- `label_pointer` = ID of the CompressedLabel

**`memory_units` row update:**
- `status` → `compressed`
- `compressed_label_id` → `label_id`
- `archived_entry_id` → `archived_entry_id`
- `claim` field is **not changed** — only status and pointer fields are updated

### 5.3 Restoration (Compressed → Active)

`store.restore_atomic(mu_id)`:
1. Verifies current status is `COMPRESSED`
2. Updates MU: `status=ACTIVE`, clears `compressed_label_id` and `archived_entry_id`, sets `needs_reindex=True`
3. Increments `restoration_count` on the archive entry
4. Deletes the `compressed_labels` row
5. Deletes the `archived_entries` row
6. Returns the updated MemoryUnit

---

## 6. Retrieval Pipeline

Entry point: `SystemEngine.ask(question, session_id, generate)`

### 6.1 Four Parallel Workers

All four workers are submitted simultaneously via `ThreadPoolExecutor(max_workers=4, thread_name_prefix="mem_retrieval")`. Each has a 15-second timeout. Failures are caught per-worker — remaining workers continue unaffected.

**Worker 1 — Dense FAISS over ACTIVE**
- Embedding model: `BAAI/bge-small-en-v1.5` (384-dim, L2-normalized float32)
- Searches `MemoryFAISSIndex` which contains embeddings of all active MU `claim` texts
- Candidate pool: 20 results

**Worker 2 — BM25 sparse over ACTIVE**
- Index: `rank_bm25` (`MemoryBM25Index`) over active MU `claim` texts
- Candidate pool: 20 results

**Worker 3 — FAISS over COMPRESSED labels → pointer follow → archive**
- Searches `CompressedLabelFAISSIndex` which contains embeddings of all `short_summary` texts
- Candidate pool: 10 results
- For each label hit: verifies MU exists and has `status=COMPRESSED`
- **Pointer follow:** calls `CompressionService.peek_archive(mu_id)` which:
  - Reads `archived_entries` row
  - Deserializes `full_memory_unit_json` back into a `MemoryUnit` object (non-destructive, no status change)
- Replaces `hit.mu` with the deserialized archived MU — the LLM receives the full original claim text
- `is_from_label=True` is set on the hit (used for RESTORED section in ContextBuilder)

**Worker 4 — BM25 over FORGOTTEN**
- Builds a temporary in-memory BM25 index from all `FORGOTTEN` MUs in the conversation
- Candidate pool: 20 results
- Returns `HybridHit` objects with `sources=["forgotten"]`
- FORGOTTEN hits only pass the hydration filter if `"forgotten"` is in the MU's sources list

### 6.2 RRF Fusion

All workers feed into a shared `rrf_map: dict[mu_id → {rrf_score, sources, label_summary}]`.

**Contribution formula:** `1.0 / (60 + rank)` (rrf_k = 60)

Multiple workers scoring the same `mu_id` accumulate their contributions additively.

### 6.3 Graph Traversal (Lane 5, sequential — requires FAISS results first)

After FAISS results are available, 1-hop expansion follows edges from the top FAISS seed IDs:
- Edge types traversed: `RELATED_TO`, `SUPERSEDED_BY`, `CONFLICTS_WITH`
- Neighbor score: `seed_rrf_score × 0.80`
- Only `ACTIVE` MUs from the same conversation are expanded
- New neighbors added to `rrf_map` with `sources=["graph"]`

### 6.4 Top-k Selection

All candidates sorted by accumulated RRF score descending. Default `top_k=5`.

### 6.5 Auto-Promotion of Forgotten Hits

After final top-k selection, every hit with `hit.mu.status == FORGOTTEN` is immediately promoted:

```python
restored_mu = store.restore_from_forgotten(hit.mu.mu_id)
# — SQLite: status → ACTIVE, needs_reindex=True
hit.mu = restored_mu
faiss_index.add_mu(restored_mu)   # immediately back in dense search index
bm25_index.add_mu(restored_mu)   # immediately back in sparse search index
```

The fact is live in both SQLite and in-memory indexes without waiting for the next rebuild.
The lifecycle engine will compress or forget it again on the next pressure check if its salience remains low.

---

## 7. Context Building (`ContextBuilder`)

Takes the final ranked `HybridHit` list and builds a structured evidence block for the LLM.

### Section Assignment (first match wins)

| Condition on hit | Section |
|-----------------|---------|
| `hit.is_from_label == True` | RESTORED |
| `relation_meta.superseded_by` non-empty | HISTORICAL CONTEXT (SUPERSEDED) |
| `relation_meta.conflicts_with` non-empty | CONFLICTING |
| none of the above | ACTIVE MEMORIES |

### Rendered Evidence Format

Sections rendered in order: ACTIVE MEMORIES → RESTORED → HISTORICAL CONTEXT → CONFLICTING.

Each entry:
```
[{index}] {claim}
  Source: {speaker} | Session {session_id} | {timestamp} | conf={confidence:.2f}
```

SUPERSEDED entries append: `  SUPERSEDED BY: {newer claim text}`
CONFLICTED entries append: `  CONFLICTS WITH: {conflicting claim text}`
RESTORED entries append: `  Label matched: {label_summary}`

### System Prompt (exact text sent to LLM)

```
You are answering a question about a long multi-session conversation using structured memory evidence.

Rules:
1. Use ONLY the evidence provided below.
2. Trust ACTIVE MEMORIES first.
3. For HISTORICAL CONTEXT entries marked SUPERSEDED, prefer the newer fact they were replaced by.
4. For CONFLICTING memories, acknowledge the uncertainty explicitly.
5. For RESTORED entries, use the full claim text provided.
6. If no evidence supports the answer, reply exactly: "No information available."
7. Give a short, direct answer. Do not explain your reasoning or cite evidence IDs.
```

---

## 8. Answer Generation

**Model:** `anthropic/claude-3-haiku` (via OpenRouter)
**Parameters:** `temperature=0.0`, `max_tokens=200`
**Cache key:** `SHA256(question + rendered_context_text)[:20]`
**Prompt template version:** `answer_v1`

Messages sent to the LLM:
- `system`: ContextBuilder system prompt (above)
- `user`: rendered evidence block + `\n\nQuestion:\n{question}\n\nAnswer:`

If `generate=False` or no hits: answer returned is `hit[0].mu.claim` or `"No relevant memories found."` without any LLM call.

---

## 9. Persistent Storage

### SQLite Database (`data/system/memory.db`)

WAL journal mode, NORMAL synchronous mode. One connection per operation (thread-safe).

| Table | Contents |
|-------|---------|
| `memory_units` | All MUs regardless of status |
| `compressed_labels` | One row per compressed MU (LLM summary + entity data) |
| `archived_entries` | One row per compressed MU (full JSON snapshot + raw text) |
| `edges` | Typed provenance edges between MUs |
| `deletion_audit` | Tombstone log for every deleted MU |
| `schema_version` | Migration version tracking |

### LLM Cache (`data/system/llm_cache/`)

Disk-based cache (`diskcache`) keyed by `prompt_template_version + cache_input`.
Shared across all three LLM use cases: fact extraction, compression labelling, and answer generation.

### In-Memory Indexes (rebuilt from SQLite on startup)

| Index | Contents |
|-------|---------|
| `MemoryFAISSIndex` | Float32 embeddings of ACTIVE MU `claim` texts |
| `MemoryBM25Index` | BM25 token index of ACTIVE MU `claim` texts |
| `CompressedLabelFAISSIndex` | Float32 embeddings of `compressed_labels.short_summary` texts |
| `MemoryGraphIndex` | NetworkX directed graph of all EdgeRecord relationships |

All indexes are derived caches. SQLite is the only durable source of truth. Indexes can be rebuilt from SQLite at any time.

---

## 10. SystemEngine Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conversation_id` | `"user_default"` | All memories scoped to this ID |
| `db_path` | `"data/system/memory.db"` | SQLite file path |
| `model_extract` | `"anthropic/claude-3-haiku"` | Used for fact extraction and compression labelling |
| `model_answer` | `"anthropic/claude-3-haiku"` | Used for answer generation |
| `embedding_model` | `"BAAI/bge-small-en-v1.5"` | 384-dim normalized embeddings |
| `active_cap` | `100` | Max active MUs before lifecycle fires |

**Retrieval default config:**

| Config field | Value |
|-------------|-------|
| `top_k` | 5 |
| `dense_candidates` | 20 |
| `bm25_candidates` | 20 |
| `label_candidates` | 10 |
| `rrf_k` | 60 |
| `enable_bm25` | True |
| `enable_label_search` | True |
| `enable_graph_traversal` | True |
| `enable_forgotten_worker` | True |

---

## 11. Lifecycle Thresholds Reference

| Threshold | Value | Meaning |
|-----------|-------|---------|
| Lifecycle trigger | 90% of `active_cap` | Compression pass starts |
| Lifecycle target | 70% of `active_cap` | Compression pass stops |
| Forget boundary | salience < 0.15 | MU → FORGOTTEN |
| Compress boundary | 0.15 ≤ salience < 0.40 | MU → COMPRESSED |
| Recency half-life | 30 days | Time for recency score to halve |
| Retrieval saturation | `n / (n + 10)` | Retrieval frequency sub-score |
| Salience weights sum | 1.00 | importance(0.30) + confidence(0.15) + recency(0.20) + retrieval(0.15) + pinned(0.10) + uniqueness(0.10) |

---

## 12. What Phase 2 Does Not Include

- No user-facing memory management actions from the chat UI (Memory Inspector view is read-only)
- No manual promote / compress / delete from chat interface
- No ingestion of historical LoCoMo conversations (only live chat messages are ingested)
- No LoCoMo benchmark evaluation in Phase 2 (Phase 1 pipeline handles benchmark runs separately)
- No graph traversal initiated from FORGOTTEN or COMPRESSED seeds (only ACTIVE seeds expand)
- No cross-conversation memory (all retrieval strictly scoped to `conversation_id`)
- No multi-user memory separation beyond `conversation_id`
