"""SystemEngine — real SPARC-LTM pipeline wired end-to-end.

Every user message goes through:
  1. FactExtractor  (LLM)  — extract atomic claims
  2. SalienceScorer         — score each claim
  3. MemoryStore            — persist to SQLite
  4. ContradictionResolver  — detect superseded / conflicting facts, write edges
  5. FAISS + BM25 index     — make facts searchable immediately
  6. LifecycleEngine        — compress / forget if at capacity (automatic)

Every question goes through:
  1. HybridMemoryRetriever  — top-k relevant memories
  2. ContextBuilder         — structure evidence into sections
  3. OpenRouterClient (LLM) — generate a grounded answer
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from loguru import logger

from locomo_memory.data.schemas import Chunk
from locomo_memory.phase2.compression.llm_labeler import LLMLabeler
from locomo_memory.phase2.compression.service import CompressionService
from locomo_memory.phase2.context.builder import BuiltContext, ContextBuilder
from locomo_memory.phase2.contradiction.resolver import (
    ContradictionResolver,
    RelationshipType,
)
from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
from locomo_memory.phase2.indexes.label_index import CompressedLabelFAISSIndex
from locomo_memory.phase2.ingestion.fact_extractor import ExtractionResult, FactExtractor
from locomo_memory.phase2.lifecycle.engine import LifecycleBatch, LifecycleConfig, LifecycleEngine
from locomo_memory.phase2.llm.cache import LLMCache
from locomo_memory.phase2.llm.client import OpenRouterClient
from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
from locomo_memory.phase2.retrieval.hybrid_retriever import (
    HybridMemoryRetriever,
    HybridRetrieverConfig,
)
from locomo_memory.phase2.salience.scorer import SalienceScorer
from locomo_memory.phase2.schemas import EdgeType, MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.graph_index import MemoryGraphIndex
from locomo_memory.phase2.store.sqlite_store import MemoryStore
from locomo_memory.security.validators import APIKeyValidator, ConversationIDValidator, ValidationError

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ProcessResult:
    """What happened after processing one user message."""

    raw_text: str
    extracted_mus: list[MemoryUnit] = field(default_factory=list)
    extraction: ExtractionResult | None = None
    lifecycle: LifecycleBatch | None = None
    contradictions_found: int = 0
    superseded_ids: list[str] = field(default_factory=list)


@dataclass
class AskResult:
    """Full result of answering one question."""

    question: str
    answer: str
    hits: list = field(default_factory=list)
    context: BuiltContext | None = None
    from_cache: bool = False
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Named constants  (replaces magic numbers scattered through the engine)
# ---------------------------------------------------------------------------

# Per-batch dedup: claims with Jaccard token overlap >= this are merged.
# Lower than the cross-session threshold because the LLM extractor often emits
# minor paraphrases of the same fact within a single turn.
_BATCH_DEDUP_JACCARD_THRESHOLD: float = 0.60

# Cross-session dedup: an incoming MU is dropped if Jaccard overlap with any
# existing active or compressed-archived MU is at or above this threshold.
# Higher (stricter) to avoid silently dropping subtly-different facts.
_CROSS_SESSION_DEDUP_JACCARD_THRESHOLD: float = 0.85

# Confidence-guard delta for supersession.  A new fact whose confidence is
# more than this many points below the existing fact's confidence cannot
# supersede it (the resolver edge is still written for audit, but no status
# change happens).  Prevents speculative input from wiping confident facts.
_SUPERSEDE_CONFIDENCE_DELTA: float = 0.20

# LLMCache size limit (bytes).
_LLM_CACHE_SIZE_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB

# Token budget for grounded answer generation.
_ANSWER_MAX_TOKENS: int = 200

# Truncation lengths for log statements (UUID prefixes and claim previews).
_LOG_MU_ID_LEN: int = 12
_LOG_CLAIM_LEN: int = 60


_DEDUP_STOP = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "as", "is", "are", "be", "was", "were", "user", "speaker",
    "i", "my", "me", "he", "she", "it", "they", "we",
})


def _jaccard_tokens(a: str, b: str) -> float:
    def toks(s: str) -> frozenset:
        cleaned = re.sub(r"[^\w\s]", "", s.lower())
        return frozenset(t for t in cleaned.split() if t not in _DEDUP_STOP)
    ta, tb = toks(a), toks(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _dedup_batch(mus: list) -> list:
    """Remove near-duplicate MUs from a single extraction batch."""
    result: list = []
    for mu in mus:
        is_dup = any(
            _jaccard_tokens(mu.claim, existing.claim) >= _BATCH_DEDUP_JACCARD_THRESHOLD
            for existing in result
        )
        if not is_dup:
            result.append(mu)
    return result


def _load_embed_fn(model_name: str = "BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    import numpy as np

    model = SentenceTransformer(model_name)
    # get_embedding_dimension() is the current API; fall back for older library versions
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )

    def embed(texts: list[str]):
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    logger.info("Loaded embedding model '{}' (dim={})", model_name, dim)
    return embed, dim


# ---------------------------------------------------------------------------
# SystemEngine
# ---------------------------------------------------------------------------


class SystemEngine:
    """Ties all Phase 2 components together into one deployable system.

    Args:
        conversation_id: logical user/conversation scope — all memories for one user.
        db_path: path to the SQLite database (persists across restarts).
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        model_extract: model used for fact extraction (fast/cheap).
        model_answer: model used for answer generation (can be stronger).
        embedding_model: sentence-transformers model name.
        active_cap: max active memories before lifecycle compression fires.
    """

    def __init__(
        self,
        conversation_id: str = "user_default",
        db_path: str | Path = "data/system/memory.db",
        api_key: str | None = None,
        model_extract: str = "anthropic/claude-3-haiku",
        model_answer: str = "anthropic/claude-3-haiku",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        active_cap: int = 500,
    ) -> None:
        # Validate conversation_id
        try:
            self.conversation_id = ConversationIDValidator.validate(conversation_id)
        except ValidationError as e:
            raise ValueError(f"Invalid conversation_id: {e}")
        
        self._model_extract = model_extract
        self._model_answer = model_answer

        # Resolve and validate API key
        _key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        try:
            _key = APIKeyValidator.validate_openrouter_key(_key)
        except ValidationError as e:
            raise ValueError(f"Invalid API key: {e}")

        # Thread safety for index updates
        self._index_lock = Lock()
        
        # Persistent storage
        _db = Path(db_path)
        _db.parent.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(_db)
        logger.info("SystemEngine: store at {}", _db)

        # LLM client + cache
        _cache_dir = Path(db_path).parent / "llm_cache"
        _cache = LLMCache(str(_cache_dir), size_limit_bytes=_LLM_CACHE_SIZE_BYTES)
        self._llm = OpenRouterClient(api_key=_key, cache=_cache)

        # Embedding + indexes
        self._embed_fn, self._dim = _load_embed_fn(embedding_model)
        self.faiss_index = MemoryFAISSIndex(embed_fn=self._embed_fn, dim=self._dim)
        self.bm25_index = MemoryBM25Index()
        self.label_index = CompressedLabelFAISSIndex(embed_fn=self._embed_fn, dim=self._dim)
        self.graph = MemoryGraphIndex()

        # Load existing memories into indexes
        n = self.faiss_index.rebuild_from_store(self.store, conversation_id=conversation_id)
        self.bm25_index.rebuild_from_store(self.store, conversation_id=conversation_id)
        self.label_index.rebuild_from_store(self.store)
        self.graph.rebuild_from_store(self.store)
        logger.info("SystemEngine: loaded {} existing memories from store", n)

        # Phase 2 pipeline components
        self.extractor = FactExtractor(client=self._llm, model=model_extract)
        self.scorer = SalienceScorer()
        self.resolver = ContradictionResolver(store=self.store)

        # LLM-powered label builder — shared by CompressionService + LifecycleEngine
        # so every compression path (manual or automatic at capacity) uses the same
        # high-quality summarisation rather than plain truncation.
        _llm_labeler = LLMLabeler(client=self._llm, model=model_extract)

        # Auto-tune lifecycle thresholds based on cap.
        #
        # In demo mode (cap ≤ 20), the user wants to see all three lifecycle
        # destinations — COMPRESSED, ARCHIVED, FORGOTTEN — fire within a single
        # short session.  With the conservative production default
        # (forget_threshold=0.15) every fresh fact has salience ≥ 0.72, so
        # nothing ever drops into FORGOTTEN until many days have passed.
        # Raising the forget threshold to 0.80 routes low-importance opinions
        # and general statements (salience 0.72–0.78) straight to FORGOTTEN
        # while still keeping real life facts (employment 0.94, location 0.94,
        # education 0.92, lifestyle 0.82) in COMPRESSED.
        #
        # In production (large cap), keep the original thresholds so a
        # high-importance fact can never be forgotten by accident — a real
        # 30-message-a-day user would otherwise hit unexpected forget events.
        if active_cap <= 20:
            lifecycle_config = LifecycleConfig(
                active_cap=active_cap,
                salience_forget_threshold=0.80,
                salience_compress_threshold=0.95,
            )
            logger.info(
                "SystemEngine: demo-mode lifecycle thresholds active "
                "(forget=0.80, compress=0.95) for cap={}",
                active_cap,
            )
        else:
            lifecycle_config = LifecycleConfig(active_cap=active_cap)
        compression_service = CompressionService(
            store=self.store,
            label_builder=_llm_labeler,
        )
        self.lifecycle = LifecycleEngine(
            store=self.store,
            scorer=self.scorer,
            config=lifecycle_config,
            label_builder=_llm_labeler,
        )
        self.context_builder = ContextBuilder(store=self.store)
        self.retriever = HybridMemoryRetriever(
            store=self.store,
            faiss_index=self.faiss_index,
            bm25_index=self.bm25_index,
            label_index=self.label_index,
            graph=self.graph,
            compression_service=compression_service,
            default_config=HybridRetrieverConfig(
                top_k=5,
                dense_candidates=20,
                bm25_candidates=20,
                enable_bm25=True,
                enable_label_search=True,
                enable_graph_traversal=True,
                enable_forgotten_worker=True,
            ),
        )

        self._session_counter: int = 0
        # Compressed→Forgotten decay: run at most once per day per engine instance.
        self._last_decay_check: datetime = datetime.now(timezone.utc)
        self._decay_interval_days: int = 30   # idle threshold for label decay
        self._decay_check_interval_msgs: int = 20  # run check every N messages
        self._msg_since_decay: int = 0

    # ------------------------------------------------------------------
    # Index rebuild helper
    # ------------------------------------------------------------------

    def _safe_rebuild_all_indexes(self, *, log_label: str = "rebuild") -> None:
        """Rebuild FAISS / BM25 / label indexes, tolerating partial failures.

        A single index failure (e.g. corrupted FAISS file, OOM during embed)
        must not propagate up and abort the user's request — the SQLite store
        remains the source of truth, and the failed index will simply be
        rebuilt on the next call.  We log but never re-raise.

        Caller is responsible for holding ``self._index_lock`` if concurrent
        readers exist.
        """
        try:
            self.faiss_index.rebuild_from_store(
                self.store, conversation_id=self.conversation_id
            )
        except Exception as exc:
            logger.warning("SystemEngine[{}]: faiss rebuild failed: {}", log_label, exc)
        try:
            self.bm25_index.rebuild_from_store(
                self.store, conversation_id=self.conversation_id
            )
        except Exception as exc:
            logger.warning("SystemEngine[{}]: bm25 rebuild failed: {}", log_label, exc)
        try:
            self.label_index.rebuild_from_store(
                self.store, conversation_id=self.conversation_id
            )
        except Exception as exc:
            logger.warning("SystemEngine[{}]: label rebuild failed: {}", log_label, exc)

    # ------------------------------------------------------------------
    # New session
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        """Return a fresh session_id for a new chat."""
        self._session_counter += 1
        return f"s{self._session_counter}"

    # ------------------------------------------------------------------
    # Process a user message — full ingestion pipeline
    # ------------------------------------------------------------------

    def process_message(
        self,
        text: str,
        speaker: str = "User",
        session_id: str | None = None,
        dia_id: str | None = None,
    ) -> ProcessResult:
        """Run the full ingestion pipeline for one user message.

        Steps: extract → score → store → resolve contradictions → index → lifecycle.
        """
        if not text.strip():
            return ProcessResult(raw_text=text)

        _sid = session_id or self.new_session()
        chunk = self._make_chunk(text, speaker, _sid, dia_id=dia_id)

        # Step 1 — fact extraction (LLM call)
        extraction = self.extractor.extract_from_chunk(chunk)
        if not extraction.success:
            logger.warning("Extraction failed for chunk {}: {}", chunk.chunk_id, extraction.failure_reason)

        new_mus: list[MemoryUnit] = []
        superseded: list[str] = []
        n_contradictions = 0

        # Deduplicate within the extraction batch (prevents duplicate facts from same message)
        deduped_mus = _dedup_batch(extraction.memory_units)
        if len(deduped_mus) < len(extraction.memory_units):
            logger.info(
                "engine: deduped {}/{} MUs from extraction batch",
                len(deduped_mus), len(extraction.memory_units),
            )

        # Pre-fetch active MUs once for cross-session dedup and resolution
        existing_active = self.store.list_active(self.conversation_id)

        # Also fetch lifecycle-compressed ARCHIVED MUs so the contradiction
        # resolver can detect when a new fact supersedes a compressed-but-not-
        # yet-forgotten memory.  Superseded-by-update ARCHIVED MUs (no label)
        # are excluded — they are already archived for provenance and need no
        # further action.
        compressed_archived = [
            m for m in self.store.list_by_status(self.conversation_id, MemoryStatus.ARCHIVED)
            if m.compressed_label_id is not None
        ]

        for mu in deduped_mus:
            # Cross-session dedup — two-tier check:
            #   Tier 1: Fast Jaccard >= _CROSS_SESSION_DEDUP_JACCARD_THRESHOLD
            #           (catches exact/near-exact wording)
            #   Tier 2: NLI SAME_FACT via resolver.compare() (catches paraphrases)
            # Both tiers check active AND compressed_archived so that a paraphrase
            # of an already-compressed fact is not re-admitted as a new active MU.
            _dup_reason: str | None = None
            for ex in existing_active + compressed_archived:
                if _jaccard_tokens(mu.claim, ex.claim) >= _CROSS_SESSION_DEDUP_JACCARD_THRESHOLD:
                    _dup_reason = (
                        f"jaccard>={_CROSS_SESSION_DEDUP_JACCARD_THRESHOLD} "
                        f"with '{ex.claim[:50]}'"
                    )
                    break
                comparison = self.resolver.compare(ex, mu)
                if comparison.relationship == RelationshipType.SAME_FACT:
                    _dup_reason = (
                        f"SAME_FACT(conf={comparison.confidence:.2f}) "
                        f"with '{ex.claim[:50]}'"
                    )
                    break
            if _dup_reason:
                logger.info(
                    "engine: cross-session dedup — skipping '{}' | {}",
                    mu.claim[:60], _dup_reason,
                )
                continue

            # Step 2 — salience scoring
            self.scorer.score_and_update(mu)

            # Step 3 — persist
            self.store.insert_memory_unit(mu)

            # Step 4 — contradiction / supersession detection
            # Candidates: all current active MUs + newly ingested MUs this batch
            # + lifecycle-compressed ARCHIVED MUs (so a new fact can supersede
            #   a memory that was compressed due to capacity pressure).
            all_candidate_ids = (
                [ex.mu_id for ex in existing_active]
                + [m.mu_id for m in new_mus]
                + [m.mu_id for m in compressed_archived]
            )
            resolution = self.resolver.resolve_incoming(
                mu.mu_id,
                candidate_mu_ids=[mid for mid in all_candidate_ids if mid != mu.mu_id],
            )
            n_contradictions += resolution.edges_created

            # Archive every fact that this new claim supersedes so it no longer
            # appears in active retrieval. The original data is preserved with
            # ARCHIVED status and provenance edges for full auditability.
            rebuild_needed = False
            # Confidence floor for supersession is the module-level
            # _SUPERSEDE_CONFIDENCE_DELTA constant; see its definition.
            for action in resolution.actions:
                if (
                    action.action == "edge_created"
                    and action.edge is not None
                    and action.edge.edge_type == EdgeType.SUPERSEDED_BY
                ):
                    old_mu_id = action.edge.source_mu_id
                    old_mu = self.store.get_memory_unit(old_mu_id)

                    # Confidence guard: reject the supersession if the new fact
                    # is materially less confident than the old one.  We still
                    # keep the SUPERSEDED_BY edge in the graph for audit, but
                    # the older fact stays ACTIVE.
                    if (
                        old_mu is not None
                        and mu.confidence + _SUPERSEDE_CONFIDENCE_DELTA < old_mu.confidence
                    ):
                        logger.info(
                            "engine: REJECT supersession of {} (conf={:.2f}) by {} "
                            "(conf={:.2f}); new fact is too speculative",
                            old_mu_id[:12], old_mu.confidence,
                            mu.mu_id[:12], mu.confidence,
                        )
                        continue

                    superseded.append(old_mu_id)
                    # If the superseded MU is already ARCHIVED (lifecycle compression),
                    # skip the status update but still record the provenance edge.
                    if old_mu is not None and old_mu.status == MemoryStatus.ARCHIVED:
                        logger.info(
                            "engine: new fact supersedes compressed MU {} (already archived)",
                            old_mu_id[:12],
                        )
                        continue  # Already off the active retrieval stack
                    try:
                        self.store.update_status(old_mu_id, MemoryStatus.ARCHIVED)
                        rebuild_needed = True
                        logger.info(
                            "engine: archived superseded MU {} (superseded by {})",
                            old_mu_id[:12], mu.mu_id[:12],
                        )
                    except Exception as exc:
                        logger.warning("engine: could not archive {}: {}", old_mu_id, exc)

            # Remove archived MUs from existing_active so subsequent MUs in this
            # batch don't see stale candidates
            if rebuild_needed:
                existing_active = [
                    ex for ex in existing_active if ex.mu_id not in superseded
                ]

            # Step 5 — update indexes
            with self._index_lock:
                try:
                    self.faiss_index.add_mu(mu)
                except Exception as exc:
                    logger.warning("SystemEngine[ingest]: faiss add_mu failed: {}", exc)
                try:
                    self.bm25_index.add_mu(mu)
                except Exception as exc:
                    logger.warning("SystemEngine[ingest]: bm25 add_mu failed: {}", exc)
                if rebuild_needed:
                    # Remove archived MUs from FAISS/BM25/label_index so they
                    # stop being retrieved.  label_index is also rebuilt here
                    # because a superseded MU may have been lifecycle-compressed
                    # (ARCHIVED with a label), and that label must be evicted.
                    self._safe_rebuild_all_indexes(log_label="post-supersession")

            existing_active.append(mu)  # new MU is now a candidate for subsequent MUs
            new_mus.append(mu)
            logger.info("Ingested: [{:.2f}] '{}'", mu.salience_score, mu.claim[:60])
            # Remove any compressed_archived MUs that were just superseded from future checks
            if superseded:
                compressed_archived = [m for m in compressed_archived if m.mu_id not in superseded]

        # Step 6 — lifecycle check (auto compress/forget at capacity)
        batch = self.lifecycle.maybe_run(self.conversation_id)
        if batch.n_compressed > 0 or batch.n_forgotten > 0:
            logger.info(
                "Lifecycle: compressed={}, forgotten={}",
                batch.n_compressed, batch.n_forgotten,
            )
            # Rebuild indexes after lifecycle changes (thread-safe).
            # Failures here must not fail the user's request — the store is
            # the source of truth and a stale index recovers on next call.
            with self._index_lock:
                self._safe_rebuild_all_indexes(log_label="post-lifecycle")

        # Periodic compressed→forgotten decay (runs every N messages)
        self._msg_since_decay += 1
        if self._msg_since_decay >= self._decay_check_interval_msgs:
            self._run_compressed_decay()
            self._msg_since_decay = 0

        return ProcessResult(
            raw_text=text,
            extracted_mus=new_mus,
            extraction=extraction,
            lifecycle=batch,
            contradictions_found=n_contradictions,
            superseded_ids=superseded,
        )

    # ------------------------------------------------------------------
    # Ask a question — retrieval + answer generation
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        generate: bool = True,
    ) -> AskResult:
        """Retrieve relevant memories and optionally generate a grounded answer."""
        import time

        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=question,
            conversation_id=self.conversation_id,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not retrieval.hits:
            return AskResult(
                question=question,
                answer="No relevant memories found.",
                hits=[],
                retrieval_latency_ms=retrieval_ms,
            )
        if not generate:
            self._track_hit_usage(retrieval.hits)
            # Still build context so callers (eval scripts, guard checks) have a
            # BuiltContext object instead of None — avoids NoneType errors downstream.
            _, _, _ctx = self.context_builder.build_prompt(question, retrieval.hits)
            return AskResult(
                question=question,
                answer=retrieval.hits[0].mu.claim,
                hits=retrieval.hits,
                context=_ctx,
                retrieval_latency_ms=retrieval_ms,
            )

        # Build structured prompt context
        system_prompt, user_msg, context = self.context_builder.build_prompt(
            question, retrieval.hits
        )

        cache_input = hashlib.sha256(
            (question + context.rendered_text).encode()
        ).hexdigest()[:20]

        t1 = time.perf_counter()
        response = self._llm.chat_completion(
            model=self._model_answer,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            prompt_template_version="answer_v1",
            cache_input=cache_input,
            max_tokens=_ANSWER_MAX_TOKENS,
            temperature=0.0,
        )
        gen_ms = (time.perf_counter() - t1) * 1000

        result = AskResult(
            question=question,
            answer=response.content.strip(),
            hits=retrieval.hits,
            context=context,
            from_cache=response.from_cache,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
        )
        # Track usage: increment retrieval_count + last_accessed for every MU
        # that reached the final top-k.  This keeps salience scores fresh and
        # feeds the compressed-decay idle clock.
        self._track_hit_usage(retrieval.hits)
        return result

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_memories(self, status: MemoryStatus | None = None) -> list[MemoryUnit]:
        """Return all memories for this conversation, optionally filtered by status.

        Args:
            status: if provided, only MUs with this exact status are returned;
                if ``None``, all statuses are returned (Active + Compressed +
                Archived + Forgotten).
        Returns:
            A fresh list (not a generator) — caller may iterate or index freely.
        """
        if status is None:
            return self.store.list_all(self.conversation_id)
        return self.store.list_by_status(self.conversation_id, status)

    def status_counts(self) -> dict[str, int]:
        """Return counts keyed by the 4-layer display model.

        Keys returned:
          "active"     — MUs in working memory (default retrieval pool)
          "compressed" — lifecycle-compressed MUs (ARCHIVED with a label) +
                         legacy COMPRESSED rows. These have a searchable label
                         and auto-restore to ACTIVE when the label matches a
                         retrieval query.
          "archived"   — ARCHIVED MUs without a label.  Two sources flow into
                         this layer: (a) facts replaced by a newer same-topic
                         fact (provenance preserved via SUPERSEDED_BY edges),
                         and (b) compressed MUs that decayed past their idle
                         window without any retrieval.  Not retrieved by
                         default; restorable on demand.
          "forgotten"  — MUs excluded from retrieval entirely (audit-only).

        The 4-layer model makes the architecture human-readable: ACTIVE is
        working memory, COMPRESSED is a smart pointer to a summarized form,
        ARCHIVED is the historical record, and FORGOTTEN is the dead letter
        office.  Distinct sub-types of ARCHIVED are not exposed because the
        provenance reason ("superseded by X" vs "decayed from compressed") is
        already captured in the graph edges and can be displayed per-card on
        demand without needing to split counts.
        """
        raw = self.store.count_by_status(self.conversation_id)
        archived_total = raw.get(MemoryStatus.ARCHIVED, 0)

        archived_with_label = 0
        archived_no_label = 0
        if archived_total > 0:
            archived_with_label, archived_no_label = (
                self.store.count_archived_by_type(self.conversation_id)
            )

        out: dict[str, int] = {
            "active": 0,
            "compressed": 0,
            "archived": 0,
            "forgotten": 0,
        }
        for s, n in raw.items():
            if s == MemoryStatus.ARCHIVED:
                # ARCHIVED with label = lifecycle-compressed (counts under "compressed")
                # ARCHIVED without label = historical/replaced (counts under "archived")
                out["compressed"] += archived_with_label
                out["archived"]   += archived_no_label
            elif s == MemoryStatus.COMPRESSED:
                out["compressed"] += n
            elif s == MemoryStatus.ACTIVE:
                out["active"] += n
            elif s == MemoryStatus.FORGOTTEN:
                out["forgotten"] += n
        return out

    def lifecycle_pressure(self) -> float:
        """Return current capacity pressure as ``active_count / active_cap``.

        Value is in [0.0, 1.0+] (can briefly exceed 1.0 between an insert and
        the next lifecycle pass).  The Streamlit memory bar reads this value
        every render to colour itself green/amber/red.
        """
        return self.lifecycle.pressure(self.conversation_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_hit_usage(self, hits: list) -> None:
        """Increment retrieval_count + last_accessed for every MU in a top-k result.

        Skips transient source-evidence MUs (mu_id prefix ``se_``) and any MU
        that was just auto-promoted (already handled by restore path).
        Safe to call with an empty list.
        """
        for hit in hits:
            mu_id = hit.mu.mu_id
            if mu_id.startswith("se_"):
                continue
            try:
                self.store.increment_retrieval_count(mu_id)
            except Exception as exc:
                # Advisory — never fail a response due to tracking errors,
                # but DO log so silent failures are diagnosable.
                logger.debug(
                    "SystemEngine: increment_retrieval_count failed for {}: {}",
                    mu_id, exc,
                )

    def _run_compressed_decay(self) -> None:
        """Run the compressed→forgotten decay pass and rebuild indexes if anything changed."""
        try:
            n = self.store.compressed_decay_pass(
                self.conversation_id,
                max_idle_days=self._decay_interval_days,
            )
            if n > 0:
                logger.info(
                    "SystemEngine: compressed_decay_pass decayed={} conv={}",
                    n, self.conversation_id,
                )
                with self._index_lock:
                    self.faiss_index.rebuild_from_store(
                        self.store, conversation_id=self.conversation_id
                    )
                    self.bm25_index.rebuild_from_store(
                        self.store, conversation_id=self.conversation_id
                    )
                    self.label_index.rebuild_from_store(self.store)
        except Exception as exc:
            logger.warning("SystemEngine: compressed_decay_pass failed: {}", exc)

    # ------------------------------------------------------------------
    # Fast ingest — raw turn as single claim (no LLM extraction)
    # ------------------------------------------------------------------

    def fast_ingest_turn(
        self,
        text: str,
        speaker: str = "User",
        session_id: str | None = None,
        dia_id: str | None = None,
    ) -> MemoryUnit | None:
        """Store a raw turn as a single MemoryUnit without LLM fact extraction.

        Skips the FactExtractor and ContradictionResolver for speed.  Useful
        for bulk LoCoMo ingestion where the raw turn *is* the claim.  The MU
        is scored, persisted, and indexed exactly like a normally-extracted one.

        Returns the inserted MemoryUnit, or None if the text is empty.
        """
        text = text.strip()
        if not text:
            return None

        _sid = session_id or self.new_session()
        _dia = dia_id or f"D{uuid.uuid4().hex[:6]}"

        claim = f"{speaker}: {text}" if speaker else text
        mu = MemoryUnit(
            conversation_id=self.conversation_id,
            session_id=_sid,
            claim=claim,
            original_text=text,
            source_dia_ids=[_dia],
            source_speaker=speaker,
        )
        mu.salience_score = self.scorer.score(mu)
        mu.importance = mu.salience_score

        try:
            self.store.insert_memory_unit(mu)
        except Exception as exc:
            logger.warning("fast_ingest_turn: insert failed for dia_id={}: {}", _dia, exc)
            return None

        with self._index_lock:
            self.faiss_index.add_mu(mu)
            self.bm25_index.add_mu(mu)

        return mu

    def batch_fast_ingest_turns(
        self,
        turns: list,
        *,
        skip_speakers: tuple[str, ...] = ("summary", "system"),
        min_length: int = 3,
    ) -> int:
        """Ingest a list of Turn objects as raw claims in one batched pass.

        This is significantly faster than calling fast_ingest_turn() in a loop
        because embeddings are computed in a single batch and the FAISS index is
        rebuilt once at the end rather than after each turn.

        Args:
            turns: list of :class:`~locomo_memory.data.schemas.Turn` objects.
            skip_speakers: speakers to exclude (case-insensitive).
            min_length: minimum text length to include.

        Returns:
            Number of MemoryUnits inserted.
        """
        skip_lower = {s.lower() for s in skip_speakers}
        mus: list[MemoryUnit] = []
        for turn in turns:
            if turn.speaker.lower() in skip_lower:
                continue
            text = turn.text.strip()
            if len(text) < min_length:
                continue
            _dia = turn.dia_id or f"D{uuid.uuid4().hex[:6]}"
            claim = f"{turn.speaker}: {text}" if turn.speaker else text
            mu = MemoryUnit(
                conversation_id=self.conversation_id,
                session_id=turn.session_id or self.new_session(),
                claim=claim,
                original_text=text,
                source_dia_ids=[_dia],
                source_speaker=turn.speaker,
            )
            mu.salience_score = self.scorer.score(mu)
            mu.importance = mu.salience_score
            mus.append(mu)

        if not mus:
            return 0

        # Bulk SQLite insert
        inserted: list[MemoryUnit] = []
        for mu in mus:
            try:
                self.store.insert_memory_unit(mu)
                inserted.append(mu)
            except Exception as exc:
                logger.warning("batch_fast_ingest: insert failed for {}: {}", mu.mu_id[:12], exc)

        # Single batched index build
        if inserted:
            with self._index_lock:
                self.faiss_index.add_mus(inserted)
                self.bm25_index.add_mus(inserted)

        logger.info("batch_fast_ingest: {} MUs ingested for conv={}", len(inserted), self.conversation_id)
        return len(inserted)

    def _make_chunk(self, text: str, speaker: str, session_id: str, dia_id: str | None = None) -> Chunk:
        """Wrap a raw message text into a Phase 1 Chunk for the FactExtractor."""
        dia_id = dia_id or f"D{uuid.uuid4().hex[:6]}"
        ts = datetime.now(timezone.utc).isoformat()
        chunk_text = (
            f"[Conversation: {self.conversation_id} | Session: {session_id}]\n"
            f"{speaker}: {text}"
        )
        return Chunk(
            chunk_id=f"live_{uuid.uuid4().hex[:8]}",
            conversation_id=self.conversation_id,
            sample_id=self.conversation_id,
            session_id=session_id,
            turn_index_start=0,
            turn_index_end=0,
            dia_ids=[dia_id],
            speakers=[speaker],
            timestamps=[ts],
            text=chunk_text,
            chunk_strategy="live",
        )
