# Phase 2 — SPARC-LTM Final Methodology

**Smart Context Management for Long-Horizon Conversational Memory**

---

## 1. Executive Summary

Phase 2 transforms the naive RAG baseline into an enterprise-grade memory system that behaves like a smart memory manager — not just a search engine. Instead of storing raw conversation turns and treating them all equally, it extracts atomic facts, tracks their importance, manages four distinct memory tiers, automatically moves data between tiers when storage pressure builds, and lets the user override any decision.

The system answers questions using structured memory (not raw text), detects and reconciles contradictions, compresses old context into smart labels while preserving the original data safely, and runs parallel search workers across all memory tiers when a question arrives.

---

## 2. The Core Problem (Why Phase 1 Fails)

| Failure Mode | Phase 1 Best Score | Why It Fails |
|---|---|---|
| Single-hop questions | 0.246 | Vocabulary mismatch — *"researching"* vs *"looking into"* |
| Temporal questions | 0.233 | No concept of "most recent" or temporal ordering |
| Multi-hop questions | 0.565 | Misses one of two required facts |
| **Overall recall** | **0.591** | Bimodal: 35% of questions get zero recall |

Phase 1 stores raw turns. Phase 2 stores normalized facts with importance scores, contradictions tracked, and tiered storage that compresses without losing information.

---

## 3. Memory Architecture — 4 Tiers

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1: ACTIVE                                                     │
│  - Full Memory Unit text (atomic claims)                            │
│  - FAISS indexed for fast semantic search                           │
│  - BM25 indexed for keyword search                                  │
│  - Used directly in answer generation                               │
│  - High salience, frequently accessed                               │
└─────────────────────────────────────────────────────────────────────┘
              ↓  Auto-move at 90% capacity (lowest utility first)
              ↑  User prompt matches → restore full text
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2: COMPRESSED (LABEL ONLY)                                    │
│  - Short summary label (~10 words) + metadata                       │
│  - Pointer to archived full data                                    │
│  - FAISS indexed (label embedding only — light)                     │
│  - When matched → fetch full data from Archived → promote to Active │
└─────────────────────────────────────────────────────────────────────┘
              ↓  Compressed label barely accessed for very long time
              ↑  User prompt explicitly references → restore
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3: ARCHIVED (FULL DATA)                                       │
│  - Stores the EXACT original full data of compressed MUs            │
│  - Accessed only via Compressed-tier pointer                        │
│  - NOT directly searched                                            │
│  - Acts as the exact recovery layer                                 │
└─────────────────────────────────────────────────────────────────────┘
              ↓  Long unused + low salience + low frequency
              ↑  User explicit restore action
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 4: FORGOTTEN                                                  │
│  - Memory deemed not useful — out of search                         │
│  - Full text still preserved in DB (never deleted)                  │
│  - Restorable: user override OR explicit AI search                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Compressed = label only (the metadata). Archived = full original data. The label points to the archive. When a user query matches the label, the full data is fetched from archive and promoted back to active.

### Source-of-Truth Discipline

Not all storage layers are equal. They have explicit roles:

| Layer | Role | Rebuildable? |
|---|---|---|
| **SQLite** | **Single source of truth.** Holds every MU, label, archive entry, status, salience, edge metadata. All writes go here first, transactionally. | No — this IS the data |
| FAISS (active + compressed) | Derived semantic search index | **Yes** — rebuild from SQLite anytime |
| NetworkX graph | Derived relationship index | **Yes** — rebuild from SQLite edge tables anytime |
| Disk archive files | Optional binary archive payloads | Yes — re-derived from SQLite full-text columns |

If FAISS gets corrupted, the index is dropped and rebuilt from SQLite. If the NetworkX graph crashes mid-process, it is rebuilt from the SQLite edge tables on next startup. SQLite is the only layer that must survive — the rest are caches.

### Status Lifecycle (5 States)

Memory Units carry one of these statuses at any time:

```
active      → in use, retrievable, used in answer generation
compressed  → label only in search; full data parked in archive
archived    → full data preserved, only reached via label pointer
forgotten   → removed from retrieval and model use; data preserved
deleted     → permanently removed; only an audit tombstone remains
```

**Forgotten ≠ Deleted.** Forgotten data is still in SQLite and can be restored anytime. Deleted data is gone. Only the user can trigger delete — the system never auto-deletes.

---

## 4. The Memory Unit and Label Structure

### Memory Unit (lives in Active tier)

```
MemoryUnit:
  mu_id:             unique identifier
  conversation_id:   which conversation this came from
  session_id:        which session
  
  # Content
  claim:             "Caroline is researching adoption agencies"
  original_text:     full raw turn text (for provenance)
  
  # Provenance
  source_dia_ids:    [D2:5]
  source_speaker:    "Caroline"
  timestamp:         "2024-03-15"
  extracted_at:      ISO timestamp of MU creation
  
  # Salience tracking
  salience_score:        0.82
  importance:            0.85   (LLM-judged)
  recency_weight:        0.91
  uniqueness:            0.78
  retrieval_count:       12     (how often retrieved/used)
  prompt_frequency:      0.34   (user prompt matches per total queries)
  last_accessed:         ISO timestamp
  
  # State (one of: active | compressed | archived | forgotten | deleted)
  status:                "active"
  confidence:            0.92
  
  # Relationships (Graph DB)
  superseded_by:         null or mu_id
  conflicts_with:        []
  related_to:            [mu_id, mu_id, ...]
  
  # User control
  user_pinned:           false
```

### Compressed Label (lives in Compressed tier)

```
CompressedLabel:
  label_id:              unique identifier
  archived_pointer:      points to archived_mu_id
  
  # The label = the metadata
  topic:                 "Career change"
  short_summary:         "Caroline: Google → Microsoft (March 2024)"
  key_entities:          ["Caroline", "Google", "Microsoft"]
  time_range:            "2024-01-10 to 2024-03-15"
  
  # Embedding for search
  label_embedding:       vector for FAISS
  
  # Provenance preserved
  original_dia_ids:      [D1:3, D8:7]
  
  # Tracking
  compressed_at:         ISO timestamp
  retrieval_count:       0      (since compression)
  last_label_match:      null
```

### Archived Entry (lives in Archive tier)

```
ArchivedEntry:
  archived_mu_id:        unique identifier
  label_pointer:         points back to label_id
  
  # The full original data — preserved exactly
  full_memory_unit:      complete original MemoryUnit
  full_original_text:    complete raw turn text
  
  # Restoration tracking
  archived_at:           ISO timestamp
  restoration_count:     0
```

This three-piece structure is the heart of the design:
- **Label** is small, searchable, lives in compressed tier
- **Archive** holds the heavy full data, only fetched when needed
- **Pointer** chain ensures we never lose anything

---

## 5. Ingestion Pipeline (How Memory Gets Built)

```
LoCoMo Conversation
       ↓
[1] Trivial Filter             (rule-based, removes greetings/laughs)
       ↓
[2] Semantic Chunking          (group consecutive turns by topic similarity)
       ↓
[3] Memory Candidate Detector  (lightweight scoring — should we even call the LLM?)
       ↓
[4] Agentic Chunking           (LLM extracts atomic facts from candidate chunks)
       ↓
[5] Embedding Generation       (BGE-small encodes each claim)
       ↓
[6] Salience Scoring           (multi-factor scoring, no LLM)
       ↓
[7] Contradiction Detection    (FAISS similarity → LLM classifier)
       ↓
[8] Graph Linking              (NetworkX records relationships)
       ↓
[9] Memory Store Write         (SQLite first → derived FAISS + graph)
```

### Step 1 — Trivial Filter (Free, Fast)

Filters out turns with no extractable facts:
- Less than 5 words
- Pure greetings/affirmations: *"hi"*, *"ok"*, *"haha"*, *"yeah sure"*
- Cuts ~15% of LLM extraction calls with zero accuracy loss

### Step 2 — Semantic Chunking

**What it does:** Groups consecutive turns that talk about the same topic into one semantic unit, instead of fixed-size windows.

**How it works:**
1. Embed every turn individually with BGE-small
2. Walk through turns sequentially
3. If `cosine(turn_i, turn_i-1) > 0.65` → same topic, extend current chunk
4. Otherwise → close current chunk, start a new one

**Example:**
```
Turn 1: "I quit my job at Google"           → chunk_1 starts
Turn 2: "Yeah I was burned out there"       → similar (0.71) → chunk_1 extends
Turn 3: "Anyway, weather is nice today"     → low sim (0.12) → chunk_2 starts
Turn 4: "Going for a walk later"            → similar (0.68) → chunk_2 extends
```

**Why it beats fixed window3:** Topic boundaries are respected. A 3-turn fixed window can split a topic across chunks. Semantic chunking keeps it whole.

### Step 3 — Memory Candidate Detector (Cheap Filter Before LLM)

**Purpose:** Avoid calling the extraction LLM on chunks that obviously contain no extractable facts. The trivial filter only catches greetings; this step catches subtler low-value chunks (e.g., long emotional venting with no concrete facts, pure speculation, hypothetical "what if" debates).

**How it works (rule-based, no LLM, runs in microseconds):**

```
candidate_score =
      0.30 * has_named_entity        # spaCy / regex catches Persons, Orgs, Dates
    + 0.20 * verb_density             # action verbs vs filler
    + 0.15 * is_factual_statement     # statement vs question/opinion
    + 0.15 * has_concrete_topic_marker  # work, family, health, location, time
    + 0.10 * length_normalized        # very short = lower; 30-150 words = ideal
    + 0.10 * has_specific_number_or_date

if candidate_score >= 0.35:
    → SEND to LLM extractor (Step 4)
else:
    → SKIP LLM; archive raw turn for provenance only
```

**Expected savings:**
- Trivial filter alone: ~15% of turns skipped
- Trivial filter + Candidate Detector: **~35–45%** of turns skipped before LLM
- Direct LLM-cost reduction with no measurable accuracy loss

**Tunable threshold:** the 0.35 cutoff is a config value. For an ablation that disables the detector, set it to 0.0 — every chunk goes to the LLM.

### Step 4 — Agentic Chunking (Fact Extraction)

**What it does:** An LLM agent reads each semantic chunk and decides what atomic facts to extract. It is "agentic" because it makes decisions — what counts as a complete fact, what to merge, what to skip.

**LLM Call:** llama-3.1-8b-instruct via OpenRouter (~$0.07/M tokens — very cheap)

**Prompt:**
```
You are a memory extraction agent. Read this conversation chunk and extract 
atomic facts. Make decisions:
- One fact per line, complete and standalone
- Normalize entity names to full form (resolve pronouns)
- Merge facts that say the same thing differently
- Skip opinions, questions, uncertain statements
- Maximum 7 facts per chunk

Chunk [Session 2 | 2024-03-15]:
"Caroline: I quit Google. Starting Microsoft Monday. Super nervous though."
"Jake: That's huge! What was the final straw at Google?"
"Caroline: My team got reorged for the third time."

Output JSON:
{
  "facts": [
    "Caroline left Google in March 2024",
    "Caroline starts at Microsoft on Monday after quitting Google",
    "Caroline left Google because her team was reorganized three times",
    "Caroline feels nervous about starting at Microsoft"
  ]
}
```

Each extracted fact becomes one Memory Unit. **Every LLM call is cached by diskcache keyed on `model_name + prompt_template_hash + input_text_hash`** — exact same inputs always produce a cache hit, no second LLM call.

### Step 5 — Embedding Generation

Each MU's claim text is embedded with `BAAI/bge-small-en-v1.5` (384-dim, normalized). Embeddings are cached by diskcache keyed on `model_name + text_hash`.

### Step 6 — Salience Scoring (No LLM, Pure Math)

```
salience = 0.25 * entity_density       # named entities / word count
         + 0.20 * recency_weight        # exp decay from session timestamp
         + 0.20 * topic_importance      # rule-based (life events > chitchat)
         + 0.15 * uniqueness            # 1 - max similarity to existing MUs
         + 0.10 * prompt_frequency      # how often user has asked about this topic
         + 0.10 * user_pin_bonus        # if pinned by user

utility = salience / storage_cost
```

Frequency tracking is critical here — it answers the requirement that *"user prompting frequency"* matters, not just timestamps.

### Step 7 — Contradiction Detection (LLM, Rarely Triggered)

Two-pass pipeline:

**Pass 1 (cheap):** FAISS similarity search. Only proceed to LLM if `cosine_sim > 0.85`.

**Pass 2 (LLM):** llama-3.3-70b-instruct via OpenRouter classifies the relationship:

```
Claim A (2024-01-10): "Caroline works at Google"
Claim B (2024-03-15): "Caroline joined Microsoft"

Classify: same / updated / contradiction / temporal_change / related / unrelated
```

Output → `updated` → A's status flag changes (still `active` but with `superseded_by` populated), both kept.

### Step 8 — Graph Linking (NetworkX)

Every relationship is recorded in an in-memory graph:
- `superseded_by` edges
- `conflicts_with` edges  
- `related_to` edges (e.g., surgery + cold medicine case)
- `derived_from` edges (which raw turn produced which MU)

Graph is used later for:
- Relevance propagation during retrieval
- State transition decisions (well-connected MUs stay active longer)
- Conflict resolution display in UI

> **Note:** NetworkX is a prototype-grade in-memory graph layer. It is rebuilt from SQLite edge tables on startup. For production scale (millions of MUs across many users), this layer would migrate to a real graph DB (Neo4j, Memgraph). For LoCoMo benchmark + demo scale, NetworkX is more than enough and avoids running another server.

### Step 9 — Memory Store Write (SQLite-First)

Writes follow strict source-of-truth discipline:

```
1. INSERT row into SQLite (transactional, atomic)
   ├ memory_units table        (the canonical MU record)
   ├ edges table               (graph relationships)
   └ provenance table          (raw turn → MU mapping)
   COMMIT or ROLLBACK

2. After SQLite COMMIT succeeds:
   ├ Add vector to FAISS active index   (best-effort)
   └ Add node + edges to NetworkX graph (best-effort)

3. If any derived index update fails:
   ├ Log the failure (loguru WARNING)
   ├ Mark the MU with needs_reindex = true
   └ Background sweeper rebuilds derived indexes from SQLite later
```

**Why this matters:** the system can crash anywhere — between SQLite commit and FAISS write — and recover cleanly. On next startup, any MU with `needs_reindex = true` is re-added to FAISS from its SQLite row. The graph rebuilds from the edges table. No data loss is possible because SQLite is the only thing that has to be right.

---

## 6. State Transition Engine (The 90% Capacity Trigger)

This is the most important design decision. **Automatic transitions only fire when active memory hits ~90% of its capacity.** Below that threshold, the system does nothing automatic — it only responds to user overrides. This is the simplification you asked for.

### When Auto-Transitions Run

```
if active_count < 0.90 * STORAGE_CAP:
    do nothing automatic
    only honor user override actions
    
if active_count >= 0.90 * STORAGE_CAP:
    run TransitionEngine.compute()
        ↓
    decide what to compress, archive, forget
```

### Decision Factors (Multi-Factor, Not Just Time)

For each candidate MU at the 90% trigger, compute:

```
demotion_score = (
      w1 * (1 - salience_score)        # low salience = demote
    + w2 * (1 - prompt_frequency)      # rarely prompted about
    + w3 * time_decay                  # old + unused = demote
    + w4 * (1 - graph_centrality)      # isolated in graph = demote
    + w5 * redundancy                   # similar MUs exist = demote
    - w6 * user_pinned * 100           # never demote pinned
)
```

The **GraphDB role** here: an MU that is heavily connected (related to many active MUs, references in conflicts, source of multiple derived facts) is more important than an isolated one. NetworkX computes betweenness centrality and degree centrality cheaply.

### Decision Outcomes (At 90% Trigger)

The engine ranks all active MUs by `demotion_score` (descending) and processes the top 30%:

```
For each candidate (highest demotion_score first):
    if salience_score < 0.15 AND retrieval_count == 0:
        → FORGOTTEN
        (never been useful, not worth keeping searchable)
    
    elif salience_score < 0.40 OR (age > 30 days AND retrieval_count < 2):
        → COMPRESSED + ARCHIVED
        (worth keeping, but as label only)
    
    else:
        → keep ACTIVE
```

After processing, active count should drop to ~70% of cap, leaving headroom.

### Compression Process (Active → Compressed + Archived)

```
1. LLM generates short label/summary (cheap model):
     Input: full MU text + related MUs (up to 3)
     Output: ~10-word summary + key entities
     
2. CREATE archived_entry:
     archived_mu_id = new_id()
     full_memory_unit = current MU (complete)
     full_original_text = current MU.original_text
     label_pointer = (set after step 3)
     
3. CREATE compressed_label:
     label_id = new_id()
     archived_pointer = archived_mu_id
     short_summary = LLM output
     key_entities = LLM output
     label_embedding = embed(short_summary)
     
4. UPDATE: link label.archived_pointer ↔ archive.label_pointer
     
5. REMOVE original MU embedding from Active FAISS index
     INSERT label_embedding into Compressed FAISS index
     
6. UPDATE SQLite: status = "compressed"
```

Now the MU exists in TWO places: a tiny searchable label, and the full original safely archived.

### Forgotten Process (Active or Compressed → Forgotten)

```
1. SQLite: UPDATE status = "forgotten" (committed first)
2. Remove embedding from any FAISS index (Active or Compressed)
3. Mark NetworkX node as inactive (kept for relationship history)
4. Full text still preserved in SQLite — NOT deleted
5. Can be restored by user override or explicit AI fallback search
```

### Deleted Process (User-Triggered Only — Permanent)

`forgotten` and `deleted` are different states. Forgotten data is dormant but recoverable. Deleted data is gone for good.

```
Deletion is NEVER automatic. Only the user can delete via UI override.

When user clicks "Delete":
  1. SQLite: UPDATE status = "deleted", null out content fields
  2. Keep mu_id and a small audit row for traceability
  3. Remove from all derived indexes (FAISS, graph)
  4. Cannot be restored — content is gone

Audit row preserves:
  - mu_id, deletion timestamp, who triggered it
  - Original conversation_id and source_dia_ids
  - No claim text, no original text
```

This guarantees:
- Auto-transitions never destroy data (only forget it)
- The user has the only path to permanent deletion
- A deletion audit trail exists for compliance / debugging

### Restoration Process (User Prompt Matches Compressed Label)

This is the magic part — exactly what you asked for:

```
Query arrives → search runs in parallel across all tiers (see §7)
       ↓
A search worker hits a Compressed label match
       ↓
1. Follow label.archived_pointer → fetch ArchivedEntry
2. Reconstruct full MemoryUnit from ArchivedEntry.full_memory_unit
3. INSERT MU back into Active FAISS index
4. UPDATE SQLite: status = "active", restoration_count += 1
5. Use the full data (not just the label) in answer generation
```

So when the user asks something matching a compressed label → full original data is restored from archive → used for answering. The label was essentially a "cache pointer" to the archive.

---

## 7. Query Pipeline With Parallel Workers

```
                 User Question
                       ↓
              [Question Embedder]
                       ↓
     ┌────────────────┼────────────────┬───────────────┐
     ↓                ↓                ↓               ↓
[Worker 1]       [Worker 2]       [Worker 3]      [Worker 4]
FAISS Dense      BM25 Sparse      Compressed       Graph
on Active        on Active        Label Search    Traversal
                                  (FAISS)         (NetworkX)
     ↓                ↓                ↓               ↓
  top-30           top-30          top-10           neighbors
candidates       candidates      label hits       of high MUs
     └────────────────┼────────────────┴───────────────┘
                      ↓
              [RRF Fusion + Dedup]
                      ↓
              top-30 candidate pool
                      ↓
        [Restoration Step if labels matched]
        - For each compressed label hit:
        - fetch full data from archive
        - promote to active
                      ↓
         [Cross-Encoder Reranker]
         BAAI/bge-reranker-base
         scores (question, claim) jointly
                      ↓
                  Top-5 MUs
                      ↓
        [Confidence Threshold Check]
        if mean_score < 0.5:
          search Forgotten tier as fallback
                      ↓
              [Context Builder]
        - ACTIVE section
        - SUPERSEDED section
        - CONFLICTED section
        - RESTORED section
                      ↓
        [Answer LLM via OpenRouter]
        Claude-3.5-Sonnet or GPT-4o
                      ↓
                  Answer
```

### Why Parallel Workers Matter

Each retrieval lane has different latency characteristics:
- FAISS dense search: ~2ms
- BM25 sparse search: ~5ms
- Compressed label FAISS: ~1ms (smaller index)
- Graph traversal: ~3ms

Running them sequentially: ~11ms total.  
Running them in parallel via `concurrent.futures.ThreadPoolExecutor`: ~5ms total (gated by slowest worker).

For 1,986 questions × 4 lanes, parallelism saves ~12 seconds on retrieval alone. More importantly, when we add the cross-encoder reranker (slower), parallelism keeps total query latency under 100ms.

### The 4 Worker Lanes

**Worker 1 — Dense FAISS over Active MUs**  
Question embedded → cosine search in active index → top-30 candidates with scores.

**Worker 2 — BM25 over Active MU claims**  
Question tokenized → BM25 ranking over active MU claims → top-30 with scores. Catches keyword matches dense search misses.

**Worker 3 — Compressed Label FAISS**  
Question embedded → cosine search in compressed label index → top-10 label hits. Each hit triggers archive lookup if selected.

**Worker 4 — Graph Traversal**  
For top-5 candidates from Worker 1, traverse NetworkX graph 1-hop to find related MUs that might also be relevant. Good for multi-hop questions.

### Reranking (Cross-Encoder)

After fusion, top-30 candidates go through `BAAI/bge-reranker-base`:
- Joint encoding of (question, claim) — much stronger than bi-encoder cosine
- Re-sorts the candidate pool
- Top-5 rerank winners proceed to context building

Expected gain: +5-8% recall on top of hybrid retrieval.

### Forgotten Tier Fallback

If after reranking the top-5 MUs have mean score < 0.5, the system runs an additional search against the forgotten tier (lazily — only when needed). This handles your requirement that *"if user asks something related to forgotten context, AI should restore it."*

### Context Building

The retrieved MUs are organized into a structured prompt:

```
ACTIVE MEMORIES (use these first):
[1] Caroline is researching adoption agencies
    Source: Session 2, March 15 2024 | Confidence: 0.92

[2] Caroline met with adoption agency on March 20th
    Source: Session 3, March 20 2024 | Confidence: 0.88

HISTORICAL CONTEXT (superseded, kept for reference):
[3] Caroline worked at Google
    SUPERSEDED by: Caroline now works at Microsoft (since March 2024)

CONFLICTING (treat with caution):
[4] Caroline owns a cat — CONFLICTS WITH: Caroline is allergic to cats

RESTORED FROM COMPRESSED (label match → full data fetched):
[5] Caroline mentioned hiking 12 times across early sessions
    Restored because query matched label "outdoor activities"
    Full original: [text]
```

### Answer Generation

LLM Call #3 — Claude-3.5-Sonnet or GPT-4o via OpenRouter:

```
System: You answer questions using structured memory evidence.
Rules:
  1. Use only evidence above
  2. Trust newer SUPERSEDED facts over older ones
  3. Acknowledge uncertainty for CONFLICTED memories
  4. If no evidence supports the answer, say "No information available"

[Structured context from above]

Question: What did Caroline research?
Answer:
```

---

## 8. User Override System (Secondary Path, Saves LLM Calls)

The Streamlit UI exposes manual controls. Every override is instant — no LLM call needed:

| User Action | What Happens | LLM Calls Saved |
|-------------|--------------|-----------------|
| Click "Restore to Active" on Forgotten | Direct SQLite update + FAISS insert | None during query |
| Click "Restore" on Compressed label | Archive → Active in one step | Skip restoration logic |
| Click "Compress" on Active MU | Forces compression even before 90% | Skip transition engine |
| Click "Pin" on Active MU | Salience locked at 1.0 | Permanent active |
| Click "Forget" on any MU | Direct status change to forgotten | None |
| Click "Delete" on any MU | Permanent removal (only true delete) | None |

**Why this matters for cost:** When the user knows certain context will be relevant (e.g., "I'm about to ask about my career history"), they can pre-restore those memories instead of letting the AI find them through retrieval. This:
- Saves the cross-encoder reranker call
- Skips potential forgotten-tier search
- Reduces hallucination risk (right context guaranteed)

---

## 9. Tech Stack — Component-by-Component

| # | System Component | Technology Used | Alternatives Considered | Why This One |
|---|---|---|---|---|
| 1 | Language | **Python 3.11** | Python 3.10, 3.12 | Best balance: faster than 3.10, more stable than 3.12, broad library compatibility |
| 2 | Backend API | **FastAPI** | Flask, Django, Express | Auto OpenAPI docs, async support, Pydantic-native, type-safe |
| 3 | Demo UI | **Streamlit** | Gradio, Dash, React | 50 lines = full UI, real-time updates, perfect for live demos |
| 4 | Vector DB (Active + Compressed) | **FAISS** | Qdrant, Weaviate, Chroma, Milvus | No server, file-based, fastest in-memory exact search, proven at scale |
| 5 | Sparse Retrieval | **rank-bm25** | Elasticsearch, Whoosh, Pyserini | Pure Python, no server, perfect for our scale (~10k MUs) |
| 6 | Graph layer (prototype) | **NetworkX** | Neo4j, ArangoDB, Memgraph | In-memory, no server, full Python integration. **Note:** prototype-grade — for production scale, swap to Neo4j. Rebuilt from SQLite at startup, not a transactional store. |
| 7 | Metadata Store (SQL) | **SQLite** | PostgreSQL, MySQL | File-based, zero config, ACID, more than enough for benchmark + demo |
| 8 | Embeddings | **BAAI/bge-small-en-v1.5** | bge-large, e5-base-v2, OpenAI ada-002 | Local (free), 384-dim, top-tier on MTEB for its size class |
| 9 | Reranker | **BAAI/bge-reranker-base** | Cohere Rerank, ms-marco MiniLM | Local (free), strong cross-encoder, runs on CPU |
| 10 | LLM Provider | **OpenRouter** | Direct OpenAI, direct Anthropic | Single API for many models, cost optimization, fallback if one provider down |
| 11 | LLM (Extraction) | **llama-3.1-8b-instruct** | mistral-7b, gpt-4o-mini | Cheapest capable extractor — $0.07/M tokens via OpenRouter |
| 12 | LLM (Contradiction) | **llama-3.3-70b-instruct** | claude-haiku, gpt-4o-mini | Mid-tier reasoning, $0.59/M tokens, called rarely |
| 13 | LLM (Answer) | **claude-3.5-sonnet** OR **gpt-4o** | gpt-4-turbo, claude-3-opus | Top-tier accuracy, both available via OpenRouter |
| 14 | Cache Layer | **diskcache** | Redis, manual SHA256 files, joblib | Thread-safe, file-based, automatic LRU, no server |
| 15 | Logging | **loguru** | logging stdlib, structlog | Colors, automatic rotation, single-line setup, structured output |
| 16 | Data Validation | **Pydantic v2** | dataclasses, attrs, marshmallow | Type-safe, fast (Rust core), seamless FastAPI integration |
| 17 | Config Loading | **YAML + Pydantic** | TOML, JSON, ENV-based | Human-readable, supports comments, validated by Pydantic |
| 18 | Testing | **pytest** | unittest, nose2 | Fixtures, parametrization, plugin ecosystem |
| 19 | Parallelism | **concurrent.futures.ThreadPoolExecutor** | asyncio, multiprocessing | Simple API, ideal for I/O-bound work (LLM calls + FAISS reads) |
| 20 | DataFrames | **pandas** | polars, csv stdlib | Mature, handles all formats, well-known, results saving |
| 21 | Numerical | **numpy** | torch, jax | Standard for vector math, FAISS-compatible |
| 22 | Progress | **tqdm** | rich.progress, progress | Universal, works in terminal + Jupyter + scripts |

---

## 9.1 Config Flags for Expensive Components

Every expensive subsystem has a config flag so we can run cheap variants for ablation studies and cost control. All flags live in the experiment YAML file.

```yaml
phase2:
  # Memory extraction
  enable_llm_extraction: true        # false → rule-based fact extraction (heuristic)
  candidate_detector_threshold: 0.35 # lower = more LLM calls; 0.0 = always call LLM

  # Contradiction detection
  enable_contradiction_llm: true     # false → embedding similarity only, no LLM classifier
  contradiction_similarity_threshold: 0.85  # lower = more LLM contradiction calls

  # Retrieval pipeline
  enable_reranker: true              # false → hybrid RRF top-5 directly
  enable_compressed_label_search: true   # false → search active tier only
  enable_graph_traversal_worker: true    # false → 3 workers instead of 4
  enable_forgotten_tier_fallback: true   # false → never search forgotten

  # Lifecycle
  storage_cap: 500                   # max active MUs per conversation
  transition_trigger_pct: 0.90       # when to fire auto-transitions
  enable_compression_llm: true       # false → use first 10 words as label

  # Caching (always on, but configurable directories)
  cache_dir: data/processed/phase2_cache
```

**Cache key construction (for every LLM call):**

```
cache_key = sha256(
    model_name + "|" +
    prompt_template_version + "|" +
    sha256(input_text)
)[:16]
```

This ensures: changing the prompt template invalidates cache; changing the model invalidates cache; same model + same template + same input always hits cache.

**Why these flags matter for the benchmark:**

- The ablation table the project requires (SPARC-LTM full / minus salience / minus reranker / etc.) is just a sweep over these YAML flags
- Cost control: run a fast cheap pass first, then a high-quality pass for the final demo
- Each flag is independently testable, no code changes between ablations

---

## 10. Architecture Diagram (Full System)

```
                        ╔════════════════════════════════╗
                        ║      STREAMLIT DEMO UI         ║
                        ║  Chat panel | Memory inspector ║
                        ╚═══════════════╤════════════════╝
                                        │
                        ╔═══════════════╧════════════════╗
                        ║       FASTAPI BACKEND          ║
                        ║  /chat /memory /override       ║
                        ╚═══════════════╤════════════════╝
                                        │
       ┌────────────────────────────────┼────────────────────────────────┐
       │                                │                                │
       ↓                                ↓                                ↓
╔═══════════════╗              ╔═══════════════╗              ╔═══════════════╗
║   INGESTION   ║              ║   QUERY       ║              ║   TRANSITION  ║
║   PIPELINE    ║              ║   PIPELINE    ║              ║   ENGINE      ║
║               ║              ║               ║              ║               ║
║ Semantic      ║              ║ 4 Parallel    ║              ║ Triggers @90% ║
║  Chunking     ║              ║  Workers      ║              ║  capacity     ║
║      ↓        ║              ║      ↓        ║              ║      ↓        ║
║ Agentic       ║              ║ RRF Fusion    ║              ║ GraphDB +     ║
║  Chunking     ║              ║      ↓        ║              ║  Salience +   ║
║      ↓        ║              ║ Restoration   ║              ║  Frequency    ║
║ Salience +    ║              ║ (label hits)  ║              ║      ↓        ║
║  Contradict   ║              ║      ↓        ║              ║ Compress +    ║
║      ↓        ║              ║ Reranker      ║              ║  Archive +    ║
║ Graph Link    ║              ║      ↓        ║              ║  Forget       ║
║      ↓        ║              ║ Context Build ║              ║               ║
║ Store         ║              ║      ↓        ║              ║               ║
║               ║              ║ Answer LLM    ║              ║               ║
╚═══════╤═══════╝              ╚═══════╤═══════╝              ╚═══════╤═══════╝
        │                              │                              │
        └──────────────┬───────────────┴──────────────┬───────────────┘
                       ↓                              ↓
            ╔══════════════════════╗      ╔═══════════════════════╗
            ║  STORAGE LAYER       ║      ║  CACHE LAYER          ║
            ║  ─────────────       ║      ║  ─────────────        ║
            ║  SQLite (metadata)   ║      ║  diskcache            ║
            ║  FAISS (active)      ║      ║   ├ embeddings        ║
            ║  FAISS (compressed)  ║      ║   ├ LLM extractions   ║
            ║  NetworkX (graph)    ║      ║   ├ LLM contradicts   ║
            ║  Files (archived)    ║      ║   └ LLM answers       ║
            ╚══════════════════════╝      ╚═══════════════════════╝
```

---

## 11. End-to-End Test Cases

| # | Scenario | Phase 2 Behavior | Expected Result |
|---|---|---|---|
| 1 | Simple factual question | Search active → direct answer | Recall ≥ 0.55 |
| 2 | Multi-hop question | Both facts as separate MUs, both retrieved | Recall ≥ 0.72 |
| 3 | Temporal question (most recent) | Recency weight + SUPERSEDED ordering | Recall ≥ 0.55 |
| 4 | Outdated fact (Google→Microsoft) | SUPERSEDED label, newer fact preferred | Correct current answer |
| 5 | Genuine contradiction | CONFLICTED flag → LLM acknowledges | Honest answer with uncertainty |
| 6 | Related but not conflicting (surgery+cold) | Both retrieved as related | Safe answer with full context |
| 7 | Storage at 90% | Auto-compress + archive lowest utility | Active count drops to ~70% |
| 8 | Question matches compressed label | Restore full data from archive → active | Full text used in answer |
| 9 | Question matches forgotten MU | Confidence < 0.5 → search forgotten | Restored if relevant |
| 10 | User pins a fact | Salience = 1.0, never auto-compressed | Stays active forever |
| 11 | User manually restores forgotten MU | Direct DB + FAISS update | No LLM call needed |
| 12 | Trivial turns (greetings) | Pre-filter skips them | Zero LLM cost |
| 13 | API failure mid-ingestion | diskcache + SQLite resume | No data loss on retry |
| 14 | Same fact different wording | Contradiction LLM = "same" → dedup | Single MU |
| 15 | Long conversation (1000+ turns) | Incremental ingestion + 90% cap management | Stable memory size |
| 16 | Cross-conversation isolation | Per-conversation SQLite + FAISS | No leakage |
| 17 | Concurrent queries | ThreadPoolExecutor parallelism | <100ms latency p95 |
| 18 | Reranker model unavailable | Falls back to RRF top-5 | Graceful degradation |
| 19 | OpenRouter rate limit | Tenacity retry with exponential backoff | Auto-recovery |
| 20 | Memory inspection from UI | FastAPI returns SQLite snapshot | Real-time view |

---

## 12. Implementation Plan (Build Order)

> **Important:** Phase 1 baseline code is **frozen** and must not be modified. Phase 2 is built as a separate `phase2/` module tree. Phase 1 results (`mean_evidence_recall = 0.485` and `0.591`) are the comparison anchors.

### Week 1 — Core Backend (No UI)

Strict build order — each step depends on the previous:

1. **Schemas** (Pydantic v2) — `MemoryUnit`, `CompressedLabel`, `ArchivedEntry`, `EdgeRecord`, configs
2. **Memory Store (SQLite — source of truth)** — full schema, migrations, transaction wrappers
3. **Memory Candidate Detector** — rule-based extractability scoring
4. **Semantic Chunker** — pure Python topic-boundary chunking
5. **Fact Extractor (Agentic Chunker, LLM Call #1)** — OpenRouter + diskcache + retries + cache-key discipline
6. **Salience Scorer** — multi-factor scoring with frequency tracking
7. **Lifecycle Engine (State Transition)** — 90% trigger + demotion scoring + 5-state lifecycle
8. **Contradiction Resolver (LLM Call #2)** — FAISS similarity gate + LLM classifier
9. **Derived Indexes** — FAISS (active + compressed) + NetworkX graph, both rebuildable from SQLite
10. **Retriever** — 4-worker parallel pipeline, RRF fusion, restoration step, cross-encoder reranker
11. **Context Builder** — structured prompt sections (active / superseded / conflicted / restored)
12. **Answer LLM (LLM Call #3)** — OpenRouter + caching with prompt-hash discipline
13. **Evaluator (Phase 2 runner)** — extends Phase 1 runner, same metrics, separate output dir
14. **Run on LoCoMo** — produce ablation grid + final benchmark numbers

### Week 2 — Demo & UI

15. **FastAPI backend** — `/chat`, `/memory/{conv_id}`, `/override/{action}`
16. **Streamlit chat UI** — left panel
17. **Streamlit memory inspector** — right panel with tabs per status
18. **Storage gauge + transition animations**
19. **Conflict resolution cards**
20. **Manual override controls** (pin, compress, forget, delete, restore)
21. **Provenance trail viewer**
22. **Comparison report** (Phase 1 vs Phase 2 with confidence intervals)
23. **Final tests + documentation + demo recording**

---

## 13. Expected Performance

> **These are design targets, not guaranteed outcomes.** Real results depend on extraction quality, contradiction-detector precision, embedding model behavior on LoCoMo's specific vocabulary, and LLM availability via OpenRouter. The numbers below are based on published cross-domain results for similar architectures and our own Phase 1 ablation deltas — actual measured results will be reported honestly after the LoCoMo run, including any underperforming categories.

| Metric | Phase 1 Best | Phase 2 Design Target | Reason |
|---|---|---|---|
| Single-hop recall | 0.246 | ~0.55 | Claim normalization removes vocabulary gap |
| Multi-hop recall | 0.565 | ~0.72 | Each hop is a separate MU, both can be retrieved |
| Temporal recall | 0.233 | ~0.55 | Timestamps + SUPERSEDED + recency weight |
| Adversarial recall | 0.714 | ~0.80 | Maintains hybrid gains + better context structure |
| Adv open-ended recall | 0.669 | ~0.76 | Compression + restoration + multi-MU context |
| **Overall recall** | **0.591** | **~0.72–0.78 (target range)** | Compounding improvements; not a guarantee |
| **Retrieval latency p95** (before LLM answer) | ~3ms | **<100ms** | Parallel workers + reranker — applies only to retrieval, not the full LLM-answer path which depends on OpenRouter |
| End-to-end latency p95 (incl. LLM answer) | n/a | 1–4s typical | Bounded by upstream LLM provider |
| API cost per QA item | $0 | ~$0.005 (with caching) | Most calls cached + cheap models for ingestion |

**Guardrails for honest reporting:**
- Run final benchmark with all caches cleared once, then with caches warm — report both
- Report per-category recall, not just overall
- If a category regresses vs Phase 1, document and explain why
- Include latency breakdown: retrieval-only vs full pipeline
- Include API cost breakdown: extraction / contradiction / answer

---

## 14. Why This Methodology Wins (Hackathon Judging Criteria)

| Criterion | How We Win |
|---|---|
| **Speed** | Parallel 4-worker retrieval (<100ms before LLM), diskcache, FAISS exact search |
| **Accuracy** | Claim normalization, contradiction handling, reranker, restoration |
| **Implementation simplicity** | File-based stores (SQLite, FAISS, NetworkX) — no servers needed |
| **Demo quality** | Live Streamlit chat + real-time memory state visualization |
| **Reproducibility** | YAML configs + diskcache = same input = same output forever |
| **Professional appearance** | Type-safe Pydantic, structured loguru logs, FastAPI auto docs |
| **Robustness** | Atomic writes, retry logic, graceful degradation, resumable |
| **Correctness** | No use of gold evidence at retrieval time, same eval as baseline |

---

## 15. The One-Sentence Summary

> Phase 2 is an AI memory manager that extracts atomic facts from conversations, stores them in 4 tiers (active, compressed labels, archived full data, forgotten), automatically rebalances the tiers when storage hits 90% capacity using salience and prompt frequency, lets the user override any decision, searches all tiers in parallel when a question arrives, restores compressed memories from archive on demand, and answers questions with structured provenance-grounded context — using OpenRouter for cost-optimized LLM calls across extraction, contradiction detection, and answer generation.
