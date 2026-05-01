"""Hybrid Memory Retriever — Phase 2 final pipeline.

Candidate generation via RRF over five lanes:

1. **Dense FAISS** — semantic similarity over active MU claims.
2. **BM25** — sparse keyword match over active MU claims.
3. **Compressed-label FAISS** — semantic search over CompressedLabel summaries.
4. **Graph traversal** — 1-hop expansion from dense FAISS hits.
5. **Source evidence** — BM25 over raw dialogue turns with ±N context window.

Final top-k selection via **Cross-Encoder reranking** (BAAI/bge-reranker-base
by default).  Cross-encoder scores rich (query, candidate) pairs drawn from a
larger pool and applies light dia_id diversity before truncating to top_k.

Optional: forgotten fallback, graph traversal (require attached index).
Deleted memories are **never** returned.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from loguru import logger

from locomo_memory.phase2.compression.service import CompressionService
from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
from locomo_memory.phase2.indexes.label_index import CompressedLabelFAISSIndex
from locomo_memory.phase2.indexes.source_evidence_index import (
    SourceEvidenceHit,
    SourceEvidenceIndex,
)
from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
from locomo_memory.phase2.schemas import EdgeType, MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.graph_index import MemoryGraphIndex
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Graph expansion constants (same as MemoryRetriever for consistency)
# ---------------------------------------------------------------------------

_GRAPH_SCORE_DISCOUNT: float = 0.8
# SUPERSEDED_BY intentionally excluded: following it would boost old/buried facts
_GRAPH_EXPAND_EDGE_TYPES: tuple[EdgeType, ...] = (
    EdgeType.RELATED_TO,
    EdgeType.CONFLICTS_WITH,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridRetrieverConfig:
    """Immutable configuration for one retrieval call.

    All fields can be overridden per-call via ``config_override``.
    """

    top_k: int = 5
    rrf_k: int = 60
    dense_candidates: int = 20
    bm25_candidates: int = 20
    label_candidates: int = 10
    enable_bm25: bool = True
    enable_label_search: bool = True
    enable_graph_traversal: bool = True
    enable_forgotten_worker: bool = True
    """Always search FORGOTTEN state in parallel; auto-promotes hits that reach top-k."""
    enable_forgotten_fallback: bool = False
    """Legacy last-resort fallback — superseded by enable_forgotten_worker."""
    forgotten_confidence_threshold: float = 0.3
    # --- Source evidence lane ---
    enable_source_evidence_lane: bool = False
    source_context_window: int = 2
    """±N turns around the central hit included in context_text."""
    source_bm25_top_n: int = 20
    """Candidate pool for source evidence BM25 search."""
    source_dense_top_n: int = 0
    """Dense candidate pool; 0 = disabled (future enhancement)."""
    source_lane_rrf_weight: float = 1.0
    """RRF contribution multiplier for the source evidence lane."""
    # --- Cross-encoder reranking ---
    enable_cross_encoder: bool = False
    """When True, a cross-encoder scores every candidate before final top_k cut.
    Supersedes lightweight_reranker and candidate_pool_selector when active."""
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    cross_encoder_weight: float = 3.0
    """Weight of normalised CE score relative to normalised RRF (1.0)."""
    cross_encoder_batch_size: int = 32
    cross_encoder_max_length: int = 512
    cross_encoder_pool_size: int = 50
    """How many candidates to hydrate before running the cross-encoder."""
    ce_superseded_penalty: float = 0.10
    """Score penalty for hits whose relation_meta.superseded_by is non-empty."""
    ce_diversity_max_same_dia: int = 2
    """Max hits sharing the same lead dia_id allowed in final top_k."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RelationMeta:
    """Relation edges for a single retrieved MU."""

    superseded_by: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    related_to: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HybridHit:
    """A single retrieved MU from the hybrid pipeline."""

    mu: MemoryUnit
    rrf_score: float
    rank: int
    sources: list[str]
    """Which lanes contributed: 'faiss', 'bm25', 'label', 'graph', 'source_evidence'."""
    label_summary: str | None
    """Short summary from the compressed label if this hit came via the label lane."""
    relation_meta: RelationMeta
    is_from_label: bool
    source_evidence_dia_ids: list[str] = field(default_factory=list)
    """dia_ids from the source evidence lane that contributed to this hit's RRF score."""
    is_from_source_evidence: bool = False
    """True when this hit has no linked MemoryUnit and was synthesised from raw source."""


@dataclass(slots=True)
class HybridRetrievalResult:
    """Full output of one hybrid retrieval call."""

    query: str
    conversation_id: str
    hits: list[HybridHit]
    top_k: int
    config: HybridRetrieverConfig
    forgotten_searched: bool
    retrieval_latency_ms: float

    @property
    def mu_ids(self) -> list[str]:
        return [h.mu.mu_id for h in self.hits]

    @property
    def mus(self) -> list[MemoryUnit]:
        return [h.mu for h in self.hits]


# ---------------------------------------------------------------------------
# Transient MU factory (source-evidence-only hits)
# ---------------------------------------------------------------------------


def _make_transient_mu(entry: "SourceEvidenceHit.entry", context_text: str) -> MemoryUnit:  # type: ignore[name-defined]
    """Synthesise a transient MemoryUnit from a raw source turn.

    Used when a source evidence turn has no linked MemoryUnit (e.g. the turn
    was too short to ingest, or the MU was deleted).  The result is never
    persisted to the store; its mu_id has the prefix ``se_``.
    """
    from locomo_memory.phase2.indexes.source_evidence_index import SourceEvidenceEntry
    assert isinstance(entry, SourceEvidenceEntry)
    return MemoryUnit(
        mu_id=f"se_{entry.dia_id}",
        conversation_id=entry.conversation_id,
        session_id=entry.session_id or "unknown",
        claim=context_text or entry.text or f"source:{entry.dia_id}",
        original_text=entry.text,
        source_dia_ids=[entry.dia_id],
        source_speaker=entry.speaker,
        timestamp=entry.timestamp,
        status=MemoryStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class HybridMemoryRetriever:
    """Hybrid retriever combining dense FAISS, BM25, label FAISS, and graph.

    Args:
        store: SQLite source of truth.
        faiss_index: dense FAISS index over active MU claims.
        bm25_index: BM25 index over active MU claims.
        label_index: FAISS index over CompressedLabel short summaries.
        graph: optional NetworkX graph for 1-hop expansion.
        compression_service: used for archive-preview on label hits.  If
            ``None``, label hits still appear in results but ``label_summary``
            will contain the short_summary only (no full-claim peek).
        default_config: default retrieval configuration.
    """

    def __init__(
        self,
        store: MemoryStore,
        faiss_index: MemoryFAISSIndex,
        bm25_index: MemoryBM25Index,
        label_index: CompressedLabelFAISSIndex,
        *,
        source_evidence_index: SourceEvidenceIndex | None = None,
        graph: MemoryGraphIndex | None = None,
        compression_service: CompressionService | None = None,
        default_config: HybridRetrieverConfig | None = None,
        cross_encoder=None,  # CrossEncoderRerankerProtocol | None
    ) -> None:
        self.store = store
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.label_index = label_index
        self.source_evidence_index = source_evidence_index
        self.graph = graph
        self.compression_service = compression_service
        self.default_config = default_config or HybridRetrieverConfig()
        self._cross_encoder = cross_encoder  # injected instance (tests / prod override)
        self._ce_instance = None  # lazy-created from config on first use

    # ------------------------------------------------------------------
    # Primary retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        conversation_id: str,
        config_override: HybridRetrieverConfig | None = None,
    ) -> HybridRetrievalResult:
        """Retrieve the top-k most relevant MUs using hybrid fusion.

        Args:
            query: natural-language question or search string.
            conversation_id: scope all retrieval to this conversation.
            config_override: if given, overrides ``self.default_config`` for
                this call only.

        Returns:
            :class:`HybridRetrievalResult` with ranked hits and metadata.
        """
        cfg = config_override or self.default_config
        t0 = time.perf_counter()

        # ---- Lanes 1-3: Dense FAISS + BM25 + Label — run in parallel ----
        # Each index search is a read-only operation; SQLite WAL mode and
        # FAISS / BM25 reads are safe for concurrent access from threads.
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="mem_retrieval") as _pool:
            # Worker 1 — dense FAISS over ACTIVE
            _f_faiss = _pool.submit(
                self.faiss_index.search,
                query,
                top_k=cfg.dense_candidates,
                conversation_id=conversation_id,
            )
            # Worker 2 — BM25 sparse over ACTIVE
            _f_bm25 = (
                _pool.submit(
                    self.bm25_index.search,
                    query,
                    top_k=cfg.bm25_candidates,
                    conversation_id=conversation_id,
                )
                if cfg.enable_bm25
                else None
            )
            # Worker 3 — FAISS over COMPRESSED labels → pointer follow → archive
            _f_label = (
                _pool.submit(
                    self.label_index.search,
                    query,
                    top_k=cfg.label_candidates,
                    conversation_id=conversation_id,
                )
                if cfg.enable_label_search
                else None
            )
            # Worker 4 — BM25 over FORGOTTEN; hits are auto-promoted to ACTIVE
            _f_forgotten = (
                _pool.submit(
                    self._search_forgotten,
                    query,
                    conversation_id=conversation_id,
                    top_k=cfg.dense_candidates,
                )
                if cfg.enable_forgotten_worker
                else None
            )

            try:
                faiss_results = _f_faiss.result(timeout=15.0)
            except Exception as _exc:
                logger.warning("Dense FAISS lane error: {}", _exc)
                faiss_results = []

            bm25_results = []
            if _f_bm25 is not None:
                try:
                    bm25_results = _f_bm25.result(timeout=15.0)
                except Exception as _exc:
                    logger.warning("BM25 lane error: {}", _exc)

            label_results = []
            if _f_label is not None:
                try:
                    label_results = _f_label.result(timeout=15.0)
                except Exception as _exc:
                    logger.warning("Label FAISS lane error: {}", _exc)

            forgotten_hits_raw: list[HybridHit] = []
            if _f_forgotten is not None:
                try:
                    forgotten_hits_raw = _f_forgotten.result(timeout=15.0)
                except Exception as _exc:
                    logger.warning("Forgotten worker error: {}", _exc)

        # ---- RRF fusion across all lanes --------------------------------
        # per-mu_id: {rrf_score, sources, label_summary}
        rrf_map: dict[str, dict] = {}

        def _add_rrf(
            mu_id: str,
            rank: int,
            source: str,
            label_summary: str | None = None,
            weight: float = 1.0,
            salience: float = 0.0,
        ) -> None:
            contribution = weight / (cfg.rrf_k + rank)
            if mu_id not in rrf_map:
                rrf_map[mu_id] = {"rrf": 0.0, "sources": [], "label_summary": None, "salience": salience}
            rrf_map[mu_id]["rrf"] += contribution
            # Keep highest salience seen (newer facts have higher recency in salience)
            if salience > rrf_map[mu_id].get("salience", 0.0):
                rrf_map[mu_id]["salience"] = salience
            if source not in rrf_map[mu_id]["sources"]:
                rrf_map[mu_id]["sources"].append(source)
            if label_summary is not None and rrf_map[mu_id]["label_summary"] is None:
                rrf_map[mu_id]["label_summary"] = label_summary

        for res in faiss_results:
            _add_rrf(res.mu_id, res.rank, "faiss")

        for res in bm25_results:
            _add_rrf(res.mu_id, res.rank, "bm25")

        # Label lane: map label → mu_id; mu must be COMPRESSED or we skip it
        label_mu_map: dict[str, str] = {}  # mu_id → short_summary for is_from_label tracking
        for res in label_results:
            mu_id = res.mu_id
            # Verify the MU exists and is COMPRESSED (label should be linked to it)
            mu = self.store.get_memory_unit(mu_id)
            if mu is None:
                # MU was hard-deleted (row removed from memory_units); skip.
                continue
            # Accept ARCHIVED (new: original MU in archive layer) and COMPRESSED
            # (legacy rows created before the ARCHIVED-status migration).
            if mu.status not in (MemoryStatus.ARCHIVED, MemoryStatus.COMPRESSED):
                # Label exists but MU was already restored or forgotten; skip.
                continue
            label_mu_map[mu_id] = res.short_summary
            _add_rrf(mu_id, res.rank, "label", label_summary=res.short_summary, salience=mu.salience_score)

        # ---- Worker 4 results: Forgotten → feed into RRF ---------------
        for i, h in enumerate(forgotten_hits_raw, start=1):
            _add_rrf(h.mu.mu_id, i, "forgotten")

        # ---- Lane 5: Graph traversal ------------------------------------
        if cfg.enable_graph_traversal and self.graph is not None and faiss_results:
            seed_ids = [r.mu_id for r in faiss_results]
            graph_extras = self._expand_via_graph(seed_ids, rrf_map, conversation_id=conversation_id)
            for mu_id, g_rrf in graph_extras.items():
                if mu_id not in rrf_map:
                    rrf_map[mu_id] = {"rrf": g_rrf, "sources": ["graph"], "label_summary": None}
                else:
                    rrf_map[mu_id]["rrf"] += g_rrf
                    if "graph" not in rrf_map[mu_id]["sources"]:
                        rrf_map[mu_id]["sources"].append("graph")

        # ---- Lane 5: Source evidence ------------------------------------
        # dia_id → [mu_ids] for provenance tracking on hits
        se_mu_dia_map: dict[str, list[str]] = {}
        # virtual_mu_id → SourceEvidenceHit for turns with no linked MU
        se_no_mu_map: dict[str, "SourceEvidenceHit"] = {}
        se_hit_count = 0

        if cfg.enable_source_evidence_lane and self.source_evidence_index is not None:
            se_results = self.source_evidence_index.search_bm25(
                query,
                top_n=cfg.source_bm25_top_n,
                conversation_id=conversation_id,
                context_window=cfg.source_context_window,
            )
            se_hit_count = len(se_results)
            for se_hit in se_results:
                linked = se_hit.entry.linked_mu_ids
                if linked:
                    for mu_id in linked:
                        _add_rrf(
                            mu_id, se_hit.rank, "source_evidence",
                            weight=cfg.source_lane_rrf_weight,
                        )
                        se_mu_dia_map.setdefault(mu_id, [])
                        if se_hit.entry.dia_id not in se_mu_dia_map[mu_id]:
                            se_mu_dia_map[mu_id].append(se_hit.entry.dia_id)
                else:
                    # No linked MU — register as a virtual entry in rrf_map
                    virtual_id = f"se_{se_hit.entry.dia_id}"
                    _add_rrf(
                        virtual_id, se_hit.rank, "source_evidence",
                        weight=cfg.source_lane_rrf_weight,
                    )
                    se_no_mu_map[virtual_id] = se_hit

        # ---- Hydrate MUs, drop missing rows -----------------------------
        # Sort: primary = RRF score desc, secondary = salience desc (recency tiebreaker)
        hits_pool: list[tuple[str, dict]] = sorted(
            rrf_map.items(),
            key=lambda kv: (kv[1]["rrf"], kv[1].get("salience", 0.0)),
            reverse=True,
        )

        # Determine hydration pool size.  CE needs a larger set than top_k.
        if cfg.enable_cross_encoder:
            pool_limit = max(cfg.cross_encoder_pool_size, cfg.top_k)
        else:
            pool_limit = cfg.top_k

        ranked_hits: list[HybridHit] = []
        for mu_id, meta in hits_pool:
            if len(ranked_hits) >= pool_limit:
                break

            # Source-evidence-only hit: virtual ID, no store entry
            if mu_id in se_no_mu_map:
                se_hit = se_no_mu_map[mu_id]
                transient_mu = _make_transient_mu(se_hit.entry, se_hit.context_text)
                ranked_hits.append(HybridHit(
                    mu=transient_mu,
                    rrf_score=meta["rrf"],
                    rank=0,
                    sources=list(meta["sources"]),
                    label_summary=None,
                    relation_meta=RelationMeta(),
                    is_from_label=False,
                    source_evidence_dia_ids=[se_hit.entry.dia_id],
                    is_from_source_evidence=True,
                ))
                continue

            mu = self.store.get_memory_unit(mu_id)
            if mu is None:
                # Hard-deleted between index hit and hydration; skip.
                continue
            # ACTIVE and COMPRESSED (via label) are valid by default.
            # FORGOTTEN is allowed only when worker 4 contributed to this mu_id.
            if mu.status == MemoryStatus.FORGOTTEN and "forgotten" not in meta["sources"]:
                continue

            is_label_hit = mu_id in label_mu_map and mu.status in (
                MemoryStatus.ARCHIVED, MemoryStatus.COMPRESSED
            )
            label_summary = meta["label_summary"]

            # Record label access so the decay clock resets on use.
            if is_label_hit and mu.compressed_label_id:
                try:
                    self.store.increment_label_access(mu.compressed_label_id)
                except Exception:
                    pass  # advisory — never fail retrieval

            # Pointer follow: when a compressed label matched, load the full
            # archived MU and use it as the hit's memory unit.  This injects
            # the complete original claim into the LLM context instead of the
            # truncated label.  The label_summary is kept for display only.
            if is_label_hit and self.compression_service is not None:
                archived_mu = self.compression_service.peek_archive(mu_id)
                if archived_mu is not None:
                    mu = archived_mu  # full original claim now surfaces in context
                    logger.debug(
                        "HybridRetriever: label hit mu={} → loaded full archive claim",
                        mu_id,
                    )

            relation_meta = self._build_relation_meta(mu_id)

            ranked_hits.append(HybridHit(
                mu=mu,
                rrf_score=meta["rrf"],
                rank=0,  # assigned below
                sources=list(meta["sources"]),
                label_summary=label_summary,
                relation_meta=relation_meta,
                is_from_label=is_label_hit,
                source_evidence_dia_ids=se_mu_dia_map.get(mu_id, []),
            ))

        # ---- Salience tiebreaker (recency): within similar RRF scores, newer facts win ---
        ranked_hits.sort(
            key=lambda h: h.rrf_score + 1e-4 * h.mu.salience_score,
            reverse=True,
        )

        # ---- Freshness guard: drop hits replaced by a newer hit in the set ---
        # If hit A has been SUPERSEDED_BY hit B and B is also in ranked_hits,
        # A is stale information — keeping it in the answer context risks the
        # LLM presenting outdated facts.  We always prefer the newer claim.
        # This guard is intentionally narrow: we only drop A when the fresher B
        # is *also* retrieved.  If B is not in the result set, A is kept (the
        # caller may still want historical context) but tagged so the prompt
        # builder can present it under [HISTORICAL].
        if ranked_hits:
            present_ids = {h.mu.mu_id for h in ranked_hits}
            kept: list[HybridHit] = []
            n_dropped = 0
            for h in ranked_hits:
                superseded_targets = list(getattr(h.relation_meta, "superseded_by", []) or [])
                fresher_present = [t for t in superseded_targets if t in present_ids]
                if fresher_present:
                    n_dropped += 1
                    logger.debug(
                        "HybridRetriever: drop stale mu={} (superseded by {} in hit set)",
                        h.mu.mu_id[:12], ",".join(t[:12] for t in fresher_present),
                    )
                    continue
                kept.append(h)
            if n_dropped:
                logger.info(
                    "HybridRetriever: freshness guard dropped {} stale hit(s)", n_dropped,
                )
            ranked_hits = kept

        # ---- Cross-encoder reranking / simple truncation ---------------
        # Graceful degradation: if the CE model fails to load or score (e.g.
        # offline, OOM, missing weights), we MUST NOT take down the whole
        # retrieval call.  Fall back to RRF-ordered top-k truncation, which
        # is the same path used when CE is disabled.  This keeps `ask()`
        # functional under any single-component outage.
        if cfg.enable_cross_encoder and ranked_hits:
            try:
                ranked_hits = self._cross_encoder_rerank(query, ranked_hits, cfg)
            except Exception as exc:
                logger.warning(
                    "HybridRetriever: cross-encoder rerank failed ({}); "
                    "falling back to RRF-only top-k.", exc,
                )
                ranked_hits = ranked_hits[: cfg.top_k]
        else:
            ranked_hits = ranked_hits[: cfg.top_k]

        # ---- Auto-promote forgotten hits that reached final top-k -------
        # Any FORGOTTEN MU that survived RRF + top-k selection is clearly
        # relevant again — restore it to ACTIVE so future retrievals find it
        # through the normal FAISS/BM25 path.  Indexes are updated immediately
        # so the next ingestion cycle benefits without a full rebuild.
        forgotten_searched = bool(forgotten_hits_raw)
        n_promoted = 0
        for hit in ranked_hits:
            if hit.mu.status == MemoryStatus.FORGOTTEN:
                try:
                    restored_mu = self.store.restore_from_forgotten(hit.mu.mu_id)
                    hit.mu = restored_mu
                    self.faiss_index.add_mu(restored_mu)
                    self.bm25_index.add_mu(restored_mu)
                    n_promoted += 1
                    logger.info(
                        "HybridRetriever: forgotten mu={} auto-promoted to active",
                        restored_mu.mu_id,
                    )
                except Exception as _exc:
                    logger.warning(
                        "HybridRetriever: could not promote forgotten mu={}: {}",
                        hit.mu.mu_id, _exc,
                    )

        # ---- Auto-promote ARCHIVED (compressed) hits that reached final top-k --
        # A label-lane hit that makes it into top-k means the original full-detail
        # memory is needed again.  Promote it ARCHIVED→ACTIVE so subsequent
        # queries find it directly via FAISS/BM25 without going through the label
        # tier.  The lifecycle engine will re-compress it if pressure rises again.
        n_archived_promoted = 0
        for hit in ranked_hits:
            if not hit.is_from_label:
                continue
            # The hit.mu here may be the archived snapshot (status=ACTIVE in JSON)
            # rather than the live DB row.  Re-fetch the live row to confirm status.
            live_mu = self.store.get_memory_unit(hit.mu.mu_id)
            if live_mu is None or live_mu.status not in (
                MemoryStatus.ARCHIVED, MemoryStatus.COMPRESSED
            ):
                continue
            try:
                promoted_mu = self.store.promote_archived_to_active(hit.mu.mu_id)
                hit.mu = promoted_mu
                hit.is_from_label = False  # now a first-class active hit
                self.faiss_index.add_mu(promoted_mu)
                self.bm25_index.add_mu(promoted_mu)
                n_archived_promoted += 1
                logger.info(
                    "HybridRetriever: archived mu={} auto-promoted to active (label hit)",
                    promoted_mu.mu_id,
                )
            except Exception as _exc:
                logger.warning(
                    "HybridRetriever: could not promote archived mu={}: {}",
                    hit.mu.mu_id, _exc,
                )

        # ---- Assign final ranks -----------------------------------------
        for i, hit in enumerate(ranked_hits, start=1):
            hit.rank = i

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "HybridMemoryRetriever: conv={} top_k={} faiss={} bm25={} labels={} "
            "forgotten_found={} promoted_forgotten={} promoted_archived={} "
            "source_ev={} final={} latency={:.1f}ms",
            conversation_id, cfg.top_k,
            len(faiss_results), len(bm25_results), len(label_results),
            len(forgotten_hits_raw), n_promoted, n_archived_promoted,
            se_hit_count, len(ranked_hits), latency_ms,
        )

        return HybridRetrievalResult(
            query=query,
            conversation_id=conversation_id,
            hits=ranked_hits,
            top_k=cfg.top_k,
            config=cfg,
            forgotten_searched=forgotten_searched,
            retrieval_latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def _cross_encoder_rerank(
        self,
        query: str,
        pool: list[HybridHit],
        cfg: HybridRetrieverConfig,
    ) -> list[HybridHit]:
        """Score pool with cross-encoder, apply light diversity, return top_k."""
        from locomo_memory.phase2.retrieval.cross_encoder_reranker import (
            SentenceTransformersCrossEncoderReranker,
            build_candidate_text,
        )

        # Resolve CE instance: injected > lazy-created
        ce = self._cross_encoder
        if ce is None:
            if self._ce_instance is None:
                self._ce_instance = SentenceTransformersCrossEncoderReranker(
                    model_name=cfg.cross_encoder_model,
                    batch_size=cfg.cross_encoder_batch_size,
                    max_length=cfg.cross_encoder_max_length,
                )
            ce = self._ce_instance

        # Build rich candidate texts (claim + original_text + context)
        texts = [
            build_candidate_text(h, self.source_evidence_index)
            for h in pool
        ]

        # Score all pairs in one batch
        ce_scores = ce.score_pairs(query, texts)

        # Normalise CE scores within pool so they're comparable to RRF norm
        min_ce = min(ce_scores) if ce_scores else 0.0
        max_ce = max(ce_scores) if ce_scores else 1.0
        ce_range = (max_ce - min_ce) or 1.0

        max_rrf = max((h.rrf_score for h in pool), default=1.0) or 1.0

        # Compute final scores
        scored: list[tuple[float, HybridHit]] = []
        for hit, raw_ce in zip(pool, ce_scores):
            ce_norm = (raw_ce - min_ce) / ce_range      # [0, 1] within pool
            rrf_norm = hit.rrf_score / max_rrf           # [0, 1] within pool
            salience = getattr(hit.mu, "salience_score", None) or 0.0
            confidence = getattr(hit.mu, "confidence", None) or 0.5
            rel_meta = getattr(hit, "relation_meta", None)
            sup_pen = (
                cfg.ce_superseded_penalty
                if rel_meta and getattr(rel_meta, "superseded_by", None)
                else 0.0
            )
            final = (
                ce_norm * cfg.cross_encoder_weight
                + rrf_norm
                + 0.02 * salience
                + 0.01 * confidence
                - sup_pen
            )
            scored.append((final, hit))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Light diversity: cap hits sharing the same lead dia_id
        seen_dia: dict[str, int] = {}
        diverse: list[HybridHit] = []
        for _, hit in scored:
            dia_ids = set(hit.mu.source_dia_ids or [])
            key = min(dia_ids) if dia_ids else hit.mu.mu_id
            count = seen_dia.get(key, 0)
            if count >= cfg.ce_diversity_max_same_dia:
                continue
            seen_dia[key] = count + 1
            diverse.append(hit)
            if len(diverse) >= cfg.top_k:
                break

        # Backfill if diversity filter was too aggressive
        if len(diverse) < cfg.top_k:
            used = {h.mu.mu_id for h in diverse}
            for _, hit in scored:
                if hit.mu.mu_id not in used:
                    diverse.append(hit)
                    used.add(hit.mu.mu_id)
                    if len(diverse) >= cfg.top_k:
                        break

        return diverse[: cfg.top_k]

    # ------------------------------------------------------------------
    # Graph expansion
    # ------------------------------------------------------------------

    def _expand_via_graph(
        self,
        seed_mu_ids: list[str],
        existing_rrf: dict[str, dict],
        *,
        conversation_id: str,
    ) -> dict[str, float]:
        """Follow 1-hop graph edges and return new-MU → discounted_rrf_score."""
        assert self.graph is not None
        new_entries: dict[str, float] = {}

        for seed_id in seed_mu_ids:
            if seed_id not in existing_rrf:
                continue
            seed_rrf = existing_rrf[seed_id]["rrf"]
            for edge_type in _GRAPH_EXPAND_EDGE_TYPES:
                for nbr_id in self.graph.neighbors(seed_id, edge_type=edge_type):
                    if nbr_id in existing_rrf or nbr_id in new_entries:
                        continue
                    mu = self.store.get_memory_unit(nbr_id)
                    if (
                        mu is None
                        or mu.status == MemoryStatus.FORGOTTEN
                        or mu.conversation_id != conversation_id
                    ):
                        continue
                    new_entries[nbr_id] = seed_rrf * _GRAPH_SCORE_DISCOUNT

        return new_entries

    # ------------------------------------------------------------------
    # Forgotten fallback
    # ------------------------------------------------------------------

    def _search_forgotten(
        self, query: str, *, conversation_id: str, top_k: int
    ) -> list[HybridHit]:
        """BM25 search over FORGOTTEN MUs for the given conversation."""
        forgotten_mus = self.store.list_by_status(conversation_id, MemoryStatus.FORGOTTEN)
        if not forgotten_mus:
            return []

        tmp_bm25 = MemoryBM25Index()
        tmp_bm25.rebuild(forgotten_mus)
        results = tmp_bm25.search(query, top_k=top_k, conversation_id=conversation_id)

        hits: list[HybridHit] = []
        for res in results:
            mu = self.store.get_memory_unit(res.mu_id)
            if mu is None:
                # Hard-deleted; skip.
                continue
            relation_meta = self._build_relation_meta(res.mu_id)
            hits.append(HybridHit(
                mu=mu,
                rrf_score=res.score,
                rank=0,
                sources=["forgotten"],
                label_summary=None,
                relation_meta=relation_meta,
                is_from_label=False,
            ))
        return hits

    # ------------------------------------------------------------------
    # Relation metadata
    # ------------------------------------------------------------------

    def _build_relation_meta(self, mu_id: str) -> RelationMeta:
        """Build RelationMeta by querying edges from and to this MU."""
        meta = RelationMeta()
        try:
            for edge in self.store.edges_from(mu_id):
                if edge.edge_type == EdgeType.SUPERSEDED_BY:
                    meta.superseded_by.append(edge.target_mu_id)
                elif edge.edge_type == EdgeType.CONFLICTS_WITH:
                    meta.conflicts_with.append(edge.target_mu_id)
                elif edge.edge_type == EdgeType.RELATED_TO:
                    meta.related_to.append(edge.target_mu_id)
            for edge in self.store.edges_to(mu_id):
                if edge.edge_type == EdgeType.SUPERSEDED_BY:
                    # This MU supersedes edge.source_mu_id (we track both directions)
                    pass  # not stored in RelationMeta (keep to outbound only)
                elif edge.edge_type == EdgeType.CONFLICTS_WITH:
                    if edge.source_mu_id not in meta.conflicts_with:
                        meta.conflicts_with.append(edge.source_mu_id)
                elif edge.edge_type == EdgeType.RELATED_TO:
                    if edge.source_mu_id not in meta.related_to:
                        meta.related_to.append(edge.source_mu_id)
        except Exception:
            pass  # relation metadata is advisory; never fail retrieval
        return meta

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def sync_all_indexes(self) -> int:
        """Process all ``needs_reindex`` MUs in FAISS and BM25.

        Returns the number of MUs processed.
        """
        n = self.faiss_index.sync_reindex(self.store)
        if n > 0:
            self.bm25_index.rebuild_from_store(self.store)
        return n

    def rebuild_all_indexes(
        self,
        *,
        conversation_id: str | None = None,
        rebuild_labels: bool = True,
        rebuild_graph: bool = False,
    ) -> dict[str, int]:
        """Full rebuild of FAISS, BM25, and optionally label FAISS + graph.

        Returns a dict with 'faiss', 'bm25', 'labels' counts.
        """
        n_faiss = self.faiss_index.rebuild_from_store(
            self.store, conversation_id=conversation_id
        )
        n_bm25 = self.bm25_index.rebuild_from_store(
            self.store, conversation_id=conversation_id
        )
        n_labels = 0
        if rebuild_labels:
            n_labels = self.label_index.rebuild_from_store(
                self.store, conversation_id=conversation_id
            )
        if rebuild_graph and self.graph is not None:
            self.graph.rebuild_from_store(self.store)
        return {"faiss": n_faiss, "bm25": n_bm25, "labels": n_labels}


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "HybridHit",
    "HybridMemoryRetriever",
    "HybridRetrieverConfig",
    "HybridRetrievalResult",
    "RelationMeta",
    "SourceEvidenceHit",
    "SourceEvidenceIndex",
]
