"""Phase 2 LoCoMo Runner — Milestone 10.

End-to-end evaluation harness that:
1. Loads LoCoMo conversations (via Phase 1 loader or an injected list).
2. Ingests each turn as a MemoryUnit into a per-conversation SQLite store.
3. Builds FAISS (dense) + BM25 (sparse) indexes per conversation.
4. For each QA item, runs the HybridMemoryRetriever, builds structured
   context, optionally calls an answer LLM, evaluates with the ResponseGuard,
   and scores with F1 / Exact Match / Evidence Recall.
5. Returns a Phase2RunResult and persists all output files via Phase2Evaluator.

Design constraints
------------------
- ``embed_fn`` is injected (not loaded from disk) so tests can run with a
  deterministic dummy embedder — no model downloads needed in CI.
- Generation is off by default (``config.generation.enabled = False``).
  When off, predicted_answer is empty and F1/EM are 0.0 (same convention
  as Phase 1 retrieval-only mode).
- One SQLite store per conversation (file at ``{db_dir}/{conv_id}.db``).
  Pass ``db_dir_override`` to redirect all stores to a temporary directory
  during tests.
- Phase 1 code is never modified; all imports from Phase 1 are read-only.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from locomo_memory.data.schemas import Conversation, QAItem, Turn
from locomo_memory.evaluation.qa_metrics import exact_match, token_f1
from locomo_memory.phase2.context.builder import ContextBuilder
from locomo_memory.phase2.context.guard import ResponseGuard
from locomo_memory.phase2.experiments.config import Phase2RunnerConfig
from locomo_memory.phase2.experiments.evaluator import (
    Phase2Evaluator,
    Phase2Metrics,
    Phase2PredictionRow,
    Phase2RunResult,
)
from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
from locomo_memory.phase2.indexes.label_index import CompressedLabelFAISSIndex
from locomo_memory.phase2.indexes.source_evidence_index import SourceEvidenceIndex
from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
from locomo_memory.phase2.retrieval.hybrid_retriever import (
    HybridMemoryRetriever,
    HybridRetrieverConfig,
)
from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

EmbedFn = Callable[[list[str]], np.ndarray]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Phase2LoCoMoRunner:
    """End-to-end Phase 2 evaluation runner.

    Args:
        config: validated :class:`Phase2RunnerConfig`.
        embed_fn: optional pre-built embed function.  If ``None``, the real
            ``EmbeddingGenerator`` is loaded from ``config.embedding`` at
            run time.  Always inject a dummy for tests.
        db_dir_override: if given, all SQLite stores are created here instead
            of ``config.db_dir``.  Pass ``tmp_path`` from pytest fixtures.
    """

    def __init__(
        self,
        config: Phase2RunnerConfig,
        *,
        embed_fn: EmbedFn | None = None,
        db_dir_override: str | Path | None = None,
    ) -> None:
        self.config = config
        self._embed_fn_override = embed_fn
        self._db_dir = Path(db_dir_override) if db_dir_override else Path(config.db_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        conversations: list[Conversation] | None = None,
        *,
        save: bool = True,
    ) -> Phase2RunResult:
        """Execute the full pipeline and return results.

        Args:
            conversations: pre-loaded conversations; if ``None``, loaded from
                ``config.dataset.path``.
            save: if True, persist all output files via Phase2Evaluator.

        Returns:
            :class:`Phase2RunResult` with predictions and aggregate metrics.
        """
        if conversations is None:
            conversations = self._load_data()

        cfg = self.config
        if cfg.dataset.max_conversations is not None:
            conversations = conversations[: cfg.dataset.max_conversations]

        embed_fn = self._get_embed_fn()
        guard = ResponseGuard(
            min_grounding_score=cfg.guard.min_grounding_score,
            require_uncertainty_for_conflicts=cfg.guard.require_uncertainty_for_conflicts,
        )

        all_predictions: list[Phase2PredictionRow] = []

        for conv in conversations:
            conv_preds = self._run_conversation(conv, embed_fn=embed_fn, guard=guard)
            all_predictions.extend(conv_preds)
            logger.info(
                "conv={} turns={} qa={} preds={}",
                conv.conversation_id,
                len(conv.turns),
                len(conv.qa_items),
                len(conv_preds),
            )

        evaluator = Phase2Evaluator(
            cfg.experiment_name,
            output_dir=cfg.output.dir,
        )
        metrics = evaluator.compute_metrics(
            all_predictions, n_conversations=len(conversations)
        )
        result = Phase2RunResult(
            experiment_name=cfg.experiment_name,
            n_conversations=len(conversations),
            n_qa_items=sum(len(c.qa_items) for c in conversations),
            predictions=all_predictions,
            metrics=metrics,
        )

        if save:
            evaluator.save(result)
            logger.info(
                "Phase 2 run complete: {} conversations, {} predictions, avg_f1={:.4f}",
                result.n_conversations,
                result.n_predictions,
                metrics.avg_f1,
            )

        return result

    # ------------------------------------------------------------------
    # Per-conversation pipeline
    # ------------------------------------------------------------------

    def _run_conversation(
        self,
        conv: Conversation,
        *,
        embed_fn: EmbedFn,
        guard: ResponseGuard,
    ) -> list[Phase2PredictionRow]:
        """Ingest one conversation's turns and evaluate its QA items."""
        cfg = self.config

        # Build per-conversation store + indexes
        store = self._make_store(conv.conversation_id)
        faiss_index = MemoryFAISSIndex(
            embed_fn=embed_fn,
            dim=cfg.embedding.dim,
            normalize=cfg.embedding.normalize,
        )
        bm25_index = MemoryBM25Index()
        label_index = CompressedLabelFAISSIndex(
            embed_fn=embed_fn,
            dim=cfg.embedding.dim,
            normalize=cfg.embedding.normalize,
        )
        context_builder = ContextBuilder(store, max_entries=cfg.context.max_entries)

        # Ingest turns → MemoryUnits
        self._ingest_turns(conv, store=store, faiss_index=faiss_index, bm25_index=bm25_index)

        # Build source evidence index if lane is enabled
        source_ev_index: SourceEvidenceIndex | None = None
        if cfg.retrieval.enable_source_evidence_lane:
            source_ev_index = SourceEvidenceIndex()
            source_ev_index.add_turns(conv.turns)
            active_mus = store.list_by_status(conv.conversation_id, MemoryStatus.ACTIVE)
            source_ev_index.build_links_from_mus(active_mus)
            logger.debug(
                "SourceEvidenceIndex: {} turns, {} MU links for conv={}",
                source_ev_index.size(), len(active_mus), conv.conversation_id,
            )

        # Build retriever
        retrieval_cfg = HybridRetrieverConfig(
            top_k=cfg.retrieval.top_k,
            rrf_k=cfg.retrieval.rrf_k,
            dense_candidates=cfg.retrieval.dense_candidates,
            bm25_candidates=cfg.retrieval.bm25_candidates,
            label_candidates=cfg.retrieval.label_candidates,
            enable_bm25=cfg.retrieval.enable_bm25,
            enable_label_search=cfg.retrieval.enable_label_search,
            enable_graph_traversal=cfg.retrieval.enable_graph_traversal,
            enable_forgotten_fallback=cfg.retrieval.enable_forgotten_fallback,
            enable_source_evidence_lane=cfg.retrieval.enable_source_evidence_lane,
            source_context_window=cfg.retrieval.source_context_window,
            source_bm25_top_n=cfg.retrieval.source_bm25_top_n,
            source_dense_top_n=cfg.retrieval.source_dense_top_n,
            source_lane_rrf_weight=cfg.retrieval.source_lane_rrf_weight,
            enable_cross_encoder=cfg.retrieval.enable_cross_encoder,
            cross_encoder_model=cfg.retrieval.cross_encoder_model,
            cross_encoder_weight=cfg.retrieval.cross_encoder_weight,
            cross_encoder_batch_size=cfg.retrieval.cross_encoder_batch_size,
            cross_encoder_max_length=cfg.retrieval.cross_encoder_max_length,
            cross_encoder_pool_size=cfg.retrieval.cross_encoder_pool_size,
            ce_superseded_penalty=cfg.retrieval.ce_superseded_penalty,
            ce_diversity_max_same_dia=cfg.retrieval.ce_diversity_max_same_dia,
        )
        retriever = HybridMemoryRetriever(
            store=store,
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            label_index=label_index,
            source_evidence_index=source_ev_index,
        )

        # Evaluate QA items
        qa_items = conv.qa_items
        if cfg.dataset.max_qa_per_conversation is not None:
            qa_items = qa_items[: cfg.dataset.max_qa_per_conversation]

        predictions: list[Phase2PredictionRow] = []
        for qa in qa_items:
            pred = self._process_qa(
                qa,
                retriever=retriever,
                retrieval_cfg=retrieval_cfg,
                context_builder=context_builder,
                guard=guard,
            )
            predictions.append(pred)

        return predictions

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _ingest_turns(
        self,
        conv: Conversation,
        *,
        store: MemoryStore,
        faiss_index: MemoryFAISSIndex,
        bm25_index: MemoryBM25Index,
    ) -> int:
        """Convert turns → MemoryUnits, persist to store, add to indexes."""
        cfg = self.config.ingestion
        mus: list[MemoryUnit] = []

        for turn in conv.turns:
            if cfg.skip_summary_turns and turn.speaker.lower() == "summary":
                continue
            text = turn.text.strip()
            if len(text) < cfg.min_turn_length:
                continue
            mu = self._turn_to_mu(turn)
            try:
                store.insert_memory_unit(mu)
                mus.append(mu)
            except Exception as exc:
                logger.warning("Failed to insert turn {} as MU: {}", turn.dia_id, exc)

        if mus:
            faiss_index.add_mus(mus)
            bm25_index.add_mus(mus)
            logger.debug(
                "Ingested {} MUs for conv={}", len(mus), conv.conversation_id
            )
        return len(mus)

    def _turn_to_mu(self, turn: Turn) -> MemoryUnit:
        """Convert one Turn into a MemoryUnit."""
        cfg = self.config.ingestion
        text = turn.text.strip()
        if cfg.claim_format == "speaker_text" and turn.speaker:
            claim = f"{turn.speaker}: {text}"
        else:
            claim = text

        return MemoryUnit(
            conversation_id=turn.conversation_id,
            session_id=turn.session_id,
            claim=claim,
            original_text=text,
            source_dia_ids=[turn.dia_id] if turn.dia_id else [],
            source_speaker=turn.speaker,
            timestamp=turn.timestamp or None,
        )

    # ------------------------------------------------------------------
    # QA processing
    # ------------------------------------------------------------------

    def _process_qa(
        self,
        qa: QAItem,
        *,
        retriever: HybridMemoryRetriever,
        retrieval_cfg: HybridRetrieverConfig,
        context_builder: ContextBuilder,
        guard: ResponseGuard,
    ) -> Phase2PredictionRow:
        """Retrieve + build context + (optionally) generate + evaluate one QA item."""
        t_start = time.perf_counter()

        # --- Retrieval ---
        t_ret_start = time.perf_counter()
        retrieval_result = retriever.retrieve(
            qa.question,
            conversation_id=qa.conversation_id,
            config_override=retrieval_cfg,
        )
        retrieval_latency = (time.perf_counter() - t_ret_start) * 1000

        hits = retrieval_result.hits
        retrieved_mu_ids = [h.mu.mu_id for h in hits]
        retrieved_claims = [h.mu.claim for h in hits]
        retrieved_dia_ids = [list(h.mu.source_dia_ids) for h in hits]
        hit_lane_sources = [list(h.sources) for h in hits]
        hit_session_ids = [h.mu.session_id for h in hits]

        # --- Context building ---
        built_context = context_builder.build(qa.question, hits)
        context_sections = {
            "active": [e.claim for e in built_context.active_entries],
            "superseded": [e.claim for e in built_context.superseded_entries],
            "conflicted": [e.claim for e in built_context.conflicted_entries],
            "restored": [e.claim for e in built_context.restored_entries],
        }

        # --- Answer generation (optional) ---
        predicted_answer = ""
        generation_latency = 0.0
        if self.config.generation.enabled:
            t_gen = time.perf_counter()
            predicted_answer = self._generate_answer(qa.question, built_context)
            generation_latency = (time.perf_counter() - t_gen) * 1000

        # --- Guard check ---
        verdict = guard.check(predicted_answer, built_context)

        # --- Evidence recall ---
        evidence_recall: float | None = None
        if self.config.evaluation.compute_evidence_recall and qa.gold_evidence_ids:
            all_dia_ids: set[str] = set()
            for dia_list in retrieved_dia_ids:
                all_dia_ids.update(dia_list)
            hits_count = sum(1 for gid in qa.gold_evidence_ids if gid in all_dia_ids)
            evidence_recall = round(hits_count / len(qa.gold_evidence_ids), 4)

        # --- QA metrics ---
        f1 = 0.0
        em = False
        if predicted_answer:
            f1 = token_f1(predicted_answer, qa.answer)
            em = exact_match(predicted_answer, qa.answer)

        end_to_end_latency = (time.perf_counter() - t_start) * 1000

        return Phase2PredictionRow(
            experiment_name=self.config.experiment_name,
            conversation_id=qa.conversation_id,
            qa_id=qa.qa_id,
            question=qa.question,
            gold_answer=qa.answer,
            predicted_answer=predicted_answer,
            category=qa.category,
            gold_evidence_ids=list(qa.gold_evidence_ids),
            retrieved_mu_ids=retrieved_mu_ids,
            retrieved_claims=retrieved_claims,
            retrieved_dia_ids=retrieved_dia_ids,
            context_sections=context_sections,
            f1=f1,
            exact_match=em,
            evidence_recall=evidence_recall,
            grounding_score=verdict.grounding_score,
            guard_passed=verdict.passed,
            guard_warnings=list(verdict.warnings),
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(generation_latency, 2),
            end_to_end_latency_ms=round(end_to_end_latency, 2),
            hit_lane_sources=hit_lane_sources,
            hit_session_ids=hit_session_ids,
        )

    # ------------------------------------------------------------------
    # Answer generation (pluggable)
    # ------------------------------------------------------------------

    def _generate_answer(self, question: str, built_context: Any) -> str:
        """Call the answer LLM.  Override or mock in subclasses / tests."""
        try:
            from locomo_memory.generation.llm_client import LLMClient  # type: ignore[import]
            client = LLMClient(
                provider=self.config.generation.provider,
                model_name=self.config.generation.model_name,
                temperature=self.config.generation.temperature,
                max_tokens=self.config.generation.max_output_tokens,
                cache_dir=self.config.generation.cache_dir,
            )
            user_msg = f"{built_context.rendered_text}\n\nQuestion:\n{question}\n\nAnswer:"
            return client.generate(
                system_prompt=built_context.system_prompt,
                user_message=user_msg,
            )
        except Exception as exc:
            logger.warning("Answer generation failed ({}): {}", type(exc).__name__, exc)
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> list[Conversation]:
        from locomo_memory.data.load_locomo import load_locomo
        path = self.config.dataset.path
        logger.info("Loading LoCoMo from {}", path)
        return load_locomo(path)

    def _make_store(self, conversation_id: str) -> MemoryStore:
        self._db_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._db_dir / f"{conversation_id}.db"
        return MemoryStore(db_path)

    def _get_embed_fn(self) -> EmbedFn:
        if self._embed_fn_override is not None:
            return self._embed_fn_override
        # Load the real model only in production runs
        from locomo_memory.indexing.embeddings import EmbeddingGenerator  # type: ignore[import]
        cfg = self.config.embedding
        gen = EmbeddingGenerator(
            model_name=cfg.model_name,
            normalize=cfg.normalize,
            batch_size=cfg.batch_size,
            cache_dir=cfg.cache_dir,
        )
        return gen.embed_texts


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = ["Phase2LoCoMoRunner"]
