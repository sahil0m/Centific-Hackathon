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

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from locomo_memory.data.schemas import Chunk
from locomo_memory.phase2.compression.llm_labeler import LLMLabeler
from locomo_memory.phase2.compression.service import CompressionService
from locomo_memory.phase2.context.builder import BuiltContext, ContextBuilder
from locomo_memory.phase2.contradiction.resolver import ContradictionResolver
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
    HybridRetrievalResult,
)
from locomo_memory.phase2.salience.scorer import SalienceScorer
from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.graph_index import MemoryGraphIndex
from locomo_memory.phase2.store.sqlite_store import MemoryStore

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

_DIM = 384


def _load_embed_fn(model_name: str = "BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    import numpy as np

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

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
        active_cap: int = 100,
    ) -> None:
        self.conversation_id = conversation_id
        self._model_extract = model_extract
        self._model_answer = model_answer

        # Resolve API key
        _key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not _key:
            raise ValueError("OPENROUTER_API_KEY not set — cannot initialise SystemEngine")

        # Persistent storage
        _db = Path(db_path)
        _db.parent.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(_db)
        logger.info("SystemEngine: store at {}", _db)

        # LLM client + cache
        _cache_dir = Path(db_path).parent / "llm_cache"
        _cache = LLMCache(str(_cache_dir))
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
    ) -> ProcessResult:
        """Run the full ingestion pipeline for one user message.

        Steps: extract → score → store → resolve contradictions → index → lifecycle.
        """
        if not text.strip():
            return ProcessResult(raw_text=text)

        _sid = session_id or self.new_session()
        chunk = self._make_chunk(text, speaker, _sid)

        # Step 1 — fact extraction (LLM call)
        extraction = self.extractor.extract_from_chunk(chunk)
        if not extraction.success:
            logger.warning("Extraction failed for chunk {}: {}", chunk.chunk_id, extraction.failure_reason)

        new_mus: list[MemoryUnit] = []
        superseded: list[str] = []
        n_contradictions = 0

        for mu in extraction.memory_units:
            # Step 2 — salience scoring
            self.scorer.score_and_update(mu)

            # Step 3 — persist
            self.store.insert_memory_unit(mu)

            # Step 4 — contradiction / supersession detection
            candidates = self.store.list_active(self.conversation_id)
            resolution = self.resolver.resolve_incoming(
                mu.mu_id,
                candidate_mu_ids=[c.mu_id for c in candidates if c.mu_id != mu.mu_id],
            )
            n_contradictions += resolution.edges_created  # int property
            from locomo_memory.phase2.schemas import EdgeType
            for action in resolution.actions:
                if (
                    action.action == "edge_created"
                    and action.edge is not None
                    and action.edge.edge_type == EdgeType.SUPERSEDED_BY
                ):
                    superseded.append(action.edge.source_mu_id)

            # Step 5 — update indexes incrementally
            self.faiss_index.add_mu(mu)
            self.bm25_index.add_mu(mu)
            new_mus.append(mu)
            logger.info("Ingested: [{:.2f}] '{}'", mu.salience_score, mu.claim[:60])

        # Step 6 — lifecycle check (auto compress/forget at capacity)
        batch = self.lifecycle.maybe_run(self.conversation_id)
        if batch.n_compressed > 0 or batch.n_forgotten > 0:
            logger.info(
                "Lifecycle: compressed={}, forgotten={}",
                batch.n_compressed, batch.n_forgotten,
            )
            self.faiss_index.rebuild_from_store(self.store, conversation_id=self.conversation_id)
            self.bm25_index.rebuild_from_store(self.store, conversation_id=self.conversation_id)
            self.label_index.rebuild_from_store(self.store)

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
        import time, hashlib

        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=question,
            conversation_id=self.conversation_id,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not generate or not retrieval.hits:
            answer = (
                retrieval.hits[0].mu.claim
                if retrieval.hits
                else "No relevant memories found."
            )
            return AskResult(
                question=question,
                answer=answer,
                hits=retrieval.hits,
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
            max_tokens=200,
            temperature=0.0,
        )
        gen_ms = (time.perf_counter() - t1) * 1000

        return AskResult(
            question=question,
            answer=response.content.strip(),
            hits=retrieval.hits,
            context=context,
            from_cache=response.from_cache,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
        )

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_memories(self, status: MemoryStatus | None = None) -> list[MemoryUnit]:
        if status is None:
            return self.store.list_all(self.conversation_id)
        return self.store.list_by_status(self.conversation_id, status)

    def status_counts(self) -> dict[str, int]:
        raw = self.store.count_by_status(self.conversation_id)
        return {s.value: n for s, n in raw.items()}

    def lifecycle_pressure(self) -> float:
        return self.lifecycle.pressure(self.conversation_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_chunk(self, text: str, speaker: str, session_id: str) -> Chunk:
        """Wrap a raw message text into a Phase 1 Chunk for the FactExtractor."""
        dia_id = f"D{uuid.uuid4().hex[:6]}"
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
