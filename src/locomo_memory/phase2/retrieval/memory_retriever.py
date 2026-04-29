"""Memory-Aware Retriever — Phase 2 Milestone 8.

Combines FAISS dense search with optional NetworkX graph expansion to retrieve
MemoryUnit objects for a given query, scoped to a single conversation.

Retrieval pipeline
------------------
1. Embed the query text via the FAISS index's embed function.
2. Search the FAISS index for the top ``top_k * oversample`` candidates.
3. Filter candidates to the requested ``conversation_id``.
4. (Optional) Graph expansion: for each FAISS hit, follow RELATED_TO and
   SUPERSEDED_BY edges 1 hop; add active neighbor MUs not already in the
   hit set (with a slight score discount).
5. Truncate to ``top_k``, assign final ranks.
6. (Optional) Increment ``retrieval_count`` on every returned MU.

The retriever does **not** change memory lifecycle state — that belongs to the
LifecycleEngine and CompressionService.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger

from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
from locomo_memory.phase2.schemas import EdgeType, MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.graph_index import MemoryGraphIndex
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetrievalHit:
    """A single retrieved MemoryUnit with its score and provenance."""

    mu: MemoryUnit
    score: float
    rank: int
    source: str
    """'faiss' for direct FAISS hits; 'graph' for graph-expanded neighbours."""


@dataclass(slots=True)
class RetrievalResult:
    """Full output of one retrieval call."""

    query: str
    conversation_id: str
    hits: list[RetrievalHit]
    top_k: int
    graph_expanded: bool
    retrieval_latency_ms: float

    @property
    def mu_ids(self) -> list[str]:
        return [h.mu.mu_id for h in self.hits]

    @property
    def mus(self) -> list[MemoryUnit]:
        return [h.mu for h in self.hits]


# ---------------------------------------------------------------------------
# Graph expansion constants
# ---------------------------------------------------------------------------

_GRAPH_SCORE_DISCOUNT: float = 0.8
"""Score multiplier applied to graph-expanded hits."""

_GRAPH_EXPAND_EDGE_TYPES: tuple[EdgeType, ...] = (
    EdgeType.RELATED_TO,
    EdgeType.SUPERSEDED_BY,
    EdgeType.CONFLICTS_WITH,
)
"""Edge types followed during graph expansion."""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class MemoryRetriever:
    """Memory-aware retriever combining FAISS search and graph expansion.

    Args:
        store: SQLite source of truth for fetching MU objects and updating
            retrieval counts.
        faiss_index: the dense vector index over active MU claims.
        graph: optional :class:`~locomo_memory.phase2.store.graph_index.MemoryGraphIndex`
            for graph-based neighbour expansion.  Pass ``None`` to disable
            graph expansion even when ``expand_graph=True`` is requested.
    """

    def __init__(
        self,
        store: MemoryStore,
        faiss_index: MemoryFAISSIndex,
        *,
        graph: MemoryGraphIndex | None = None,
    ) -> None:
        self.store = store
        self.faiss_index = faiss_index
        self.graph = graph

    # ------------------------------------------------------------------
    # Primary retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        conversation_id: str,
        top_k: int = 5,
        expand_graph: bool = False,
        increment_retrieval: bool = True,
    ) -> RetrievalResult:
        """Retrieve the top-k most relevant active MUs for a query.

        Args:
            query: natural-language question or search string.
            conversation_id: scope all retrieval to this conversation.
            top_k: number of results to return.
            expand_graph: if ``True`` and a graph is attached, expand 1 hop
                from FAISS hits along relationship edges.
            increment_retrieval: if ``True``, increment ``retrieval_count``
                on each returned MU.

        Returns:
            :class:`RetrievalResult` with ranked hits and metadata.
        """
        t0 = time.perf_counter()

        faiss_results = self.faiss_index.search(
            query, top_k=top_k * 2, conversation_id=conversation_id
        )

        # Hydrate MU objects and filter to active status
        hits_map: dict[str, RetrievalHit] = {}
        for res in faiss_results:
            mu = self.store.get_memory_unit(res.mu_id)
            if mu is None or mu.status != MemoryStatus.ACTIVE:
                continue
            hits_map[mu.mu_id] = RetrievalHit(
                mu=mu,
                score=res.score,
                rank=0,
                source="faiss",
            )

        # Graph expansion
        actually_expanded = False
        if expand_graph and self.graph is not None and hits_map:
            extra = self._expand_via_graph(
                list(hits_map.keys()),
                hits_map,
                conversation_id=conversation_id,
            )
            if extra:
                hits_map.update(extra)
                actually_expanded = True

        # Sort by score descending, truncate, assign ranks
        ranked = sorted(hits_map.values(), key=lambda h: h.score, reverse=True)[:top_k]
        for i, hit in enumerate(ranked, start=1):
            hit.rank = i

        # Increment retrieval counts
        if increment_retrieval:
            for hit in ranked:
                try:
                    self.store.increment_retrieval_count(hit.mu.mu_id)
                    hit.mu.retrieval_count += 1
                except Exception:
                    pass  # retrieval count is advisory; never fail the retrieval

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "MemoryRetriever: conv={} top_k={} faiss_hits={} final={} "
            "graph_expanded={} latency={:.1f}ms",
            conversation_id,
            top_k,
            len(faiss_results),
            len(ranked),
            actually_expanded,
            latency_ms,
        )

        return RetrievalResult(
            query=query,
            conversation_id=conversation_id,
            hits=ranked,
            top_k=top_k,
            graph_expanded=actually_expanded,
            retrieval_latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Graph expansion
    # ------------------------------------------------------------------

    def _expand_via_graph(
        self,
        seed_mu_ids: list[str],
        existing_hits: dict[str, RetrievalHit],
        *,
        conversation_id: str,
    ) -> dict[str, RetrievalHit]:
        """Follow graph edges from seed MUs and return new active-only hits.

        Returns a dict of newly discovered mu_id → RetrievalHit (not already
        in ``existing_hits``).  Scores are the seed score times
        ``_GRAPH_SCORE_DISCOUNT``.
        """
        new_hits: dict[str, RetrievalHit] = {}
        assert self.graph is not None

        for seed_id in seed_mu_ids:
            seed_score = existing_hits[seed_id].score
            for edge_type in _GRAPH_EXPAND_EDGE_TYPES:
                neighbours = self.graph.neighbors(seed_id, edge_type=edge_type)
                for nbr_id in neighbours:
                    if nbr_id in existing_hits or nbr_id in new_hits:
                        continue
                    mu = self.store.get_memory_unit(nbr_id)
                    if (
                        mu is None
                        or mu.status != MemoryStatus.ACTIVE
                        or mu.conversation_id != conversation_id
                    ):
                        continue
                    new_hits[nbr_id] = RetrievalHit(
                        mu=mu,
                        score=seed_score * _GRAPH_SCORE_DISCOUNT,
                        rank=0,
                        source="graph",
                    )

        return new_hits

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def sync_index(self, *, conversation_id: str | None = None) -> int:
        """Process all ``needs_reindex`` MUs and update FAISS + graph.

        Returns the number of MUs processed.
        """
        n = self.faiss_index.sync_reindex(self.store)
        if n > 0 and self.graph is not None:
            if conversation_id is not None:
                mus = self.store.list_active(conversation_id)
            else:
                mus = list(self.store.iter_active())
            for mu in mus:
                self.graph.upsert_node(
                    mu.mu_id,
                    conversation_id=mu.conversation_id,
                    status=mu.status.value,
                    salience_score=mu.salience_score,
                    user_pinned=mu.user_pinned,
                )
        return n

    def rebuild_index(
        self,
        *,
        conversation_id: str | None = None,
        rebuild_graph: bool = False,
    ) -> int:
        """Full rebuild of the FAISS index (and optionally the graph).

        Args:
            conversation_id: scope to one conversation; ``None`` rebuilds all.
            rebuild_graph: also rebuild the NetworkX graph from the store.

        Returns the number of MUs indexed.
        """
        n = self.faiss_index.rebuild_from_store(
            self.store, conversation_id=conversation_id
        )
        if rebuild_graph and self.graph is not None:
            self.graph.rebuild_from_store(self.store)
        return n


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "MemoryRetriever",
    "RetrievalHit",
    "RetrievalResult",
]
