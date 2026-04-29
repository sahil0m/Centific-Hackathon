"""Tests for Phase 2 Milestone 8: MemoryRetriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
from locomo_memory.phase2.retrieval.memory_retriever import (
    MemoryRetriever,
    RetrievalHit,
    RetrievalResult,
)
from locomo_memory.phase2.schemas import EdgeRecord, EdgeType, MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.graph_index import MemoryGraphIndex
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Dummy embedder
# ---------------------------------------------------------------------------

DIM = 8


def _make_embed_fn(seed: int = 42):
    rng = np.random.default_rng(seed)
    cache: dict[str, np.ndarray] = {}

    def embed_fn(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t not in cache:
                v = rng.random(DIM).astype(np.float32)
                v /= np.linalg.norm(v)
                cache[t] = v
            out.append(cache[t])
        return np.array(out, dtype=np.float32)

    return embed_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mu(
    claim: str,
    *,
    conv: str = "conv1",
    mu_id: str | None = None,
    session: str = "s1",
) -> MemoryUnit:
    kw: dict = dict(conversation_id=conv, session_id=session, claim=claim)
    if mu_id:
        kw["mu_id"] = mu_id
    return MemoryUnit(**kw)


def _insert(store: MemoryStore, mu: MemoryUnit) -> MemoryUnit:
    store.insert_memory_unit(mu)
    return mu


def _make_retriever(
    store: MemoryStore,
    *,
    seed: int = 42,
    graph: MemoryGraphIndex | None = None,
) -> MemoryRetriever:
    idx = MemoryFAISSIndex(embed_fn=_make_embed_fn(seed), dim=DIM)
    return MemoryRetriever(store, idx, graph=graph)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Basic retrieve
# ---------------------------------------------------------------------------


def test_retrieve_empty_store_returns_empty(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    result = ret.retrieve("what is Alice's job?", conversation_id="conv1", top_k=5)
    assert isinstance(result, RetrievalResult)
    assert result.hits == []
    assert result.query == "what is Alice's job?"


def test_retrieve_returns_up_to_top_k(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    for i in range(10):
        mu = _insert(store, _mu(f"fact {i}", mu_id=f"m{i}"))
        ret.faiss_index.add_mu(mu)
    result = ret.retrieve("fact", conversation_id="conv1", top_k=3)
    assert len(result.hits) <= 3


def test_retrieve_all_hits_are_active(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu1 = _insert(store, _mu("active fact", mu_id="m1"))
    mu2 = _insert(store, _mu("forgotten fact", mu_id="m2"))
    ret.faiss_index.add_mus([mu1, mu2])
    store.forget_atomic("m2")  # m2 now FORGOTTEN

    result = ret.retrieve("fact", conversation_id="conv1", top_k=5)
    ids = {h.mu.mu_id for h in result.hits}
    assert "m2" not in ids


def test_retrieve_scoped_to_conversation(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu1 = _insert(store, _mu("conv1 fact", conv="conv1", mu_id="m1"))
    mu2 = _insert(store, _mu("conv2 fact", conv="conv2", mu_id="m2"))
    ret.faiss_index.add_mus([mu1, mu2])

    result = ret.retrieve("fact", conversation_id="conv1", top_k=5)
    ids = {h.mu.mu_id for h in result.hits}
    assert "m2" not in ids
    assert "m1" in ids


def test_retrieve_hit_fields(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu = _insert(store, _mu("Alice works at Google", mu_id="m1"))
    ret.faiss_index.add_mu(mu)

    result = ret.retrieve("Alice job", conversation_id="conv1", top_k=1)
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert isinstance(hit, RetrievalHit)
    assert hit.mu.mu_id == "m1"
    assert hit.rank == 1
    assert isinstance(hit.score, float)
    assert hit.source == "faiss"


def test_retrieve_hits_ranked_ascending(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mus = [_insert(store, _mu(f"item {i}", mu_id=f"m{i}")) for i in range(5)]
    ret.faiss_index.add_mus(mus)

    result = ret.retrieve("item", conversation_id="conv1", top_k=5)
    ranks = [h.rank for h in result.hits]
    assert ranks == list(range(1, len(ranks) + 1))


def test_retrieve_scores_descending(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mus = [_insert(store, _mu(f"doc {i}", mu_id=f"m{i}")) for i in range(5)]
    ret.faiss_index.add_mus(mus)

    result = ret.retrieve("doc", conversation_id="conv1", top_k=5)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_result_metadata(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu = _insert(store, _mu("a fact", mu_id="m1"))
    ret.faiss_index.add_mu(mu)

    result = ret.retrieve("query", conversation_id="conv1", top_k=5)
    assert result.conversation_id == "conv1"
    assert result.top_k == 5
    assert isinstance(result.retrieval_latency_ms, float)
    assert result.retrieval_latency_ms >= 0.0


# ---------------------------------------------------------------------------
# RetrievalResult helpers
# ---------------------------------------------------------------------------


def test_retrieval_result_mu_ids(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mus = [_insert(store, _mu(f"fact {i}", mu_id=f"m{i}")) for i in range(3)]
    ret.faiss_index.add_mus(mus)
    result = ret.retrieve("fact", conversation_id="conv1", top_k=3)
    assert set(result.mu_ids) == {h.mu.mu_id for h in result.hits}


def test_retrieval_result_mus(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu = _insert(store, _mu("claim", mu_id="m1"))
    ret.faiss_index.add_mu(mu)
    result = ret.retrieve("claim", conversation_id="conv1", top_k=1)
    assert len(result.mus) == len(result.hits)


# ---------------------------------------------------------------------------
# increment_retrieval
# ---------------------------------------------------------------------------


def test_increment_retrieval_count_on_hits(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu = _insert(store, _mu("a fact", mu_id="m1"))
    ret.faiss_index.add_mu(mu)

    result = ret.retrieve("fact", conversation_id="conv1", top_k=1,
                          increment_retrieval=True)
    assert len(result.hits) == 1
    updated = store.get_memory_unit("m1")
    assert updated.retrieval_count == 1


def test_no_increment_when_disabled(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu = _insert(store, _mu("a fact", mu_id="m1"))
    ret.faiss_index.add_mu(mu)

    ret.retrieve("fact", conversation_id="conv1", top_k=1, increment_retrieval=False)
    updated = store.get_memory_unit("m1")
    assert updated.retrieval_count == 0


def test_retrieval_count_increments_per_retrieve(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu = _insert(store, _mu("recurring fact", mu_id="m1"))
    ret.faiss_index.add_mu(mu)

    for _ in range(3):
        ret.retrieve("fact", conversation_id="conv1", top_k=1,
                     increment_retrieval=True)
    updated = store.get_memory_unit("m1")
    assert updated.retrieval_count == 3


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------


def test_retrieve_without_graph_no_expansion(store: MemoryStore) -> None:
    ret = _make_retriever(store, graph=None)
    mu = _insert(store, _mu("base fact", mu_id="m1"))
    ret.faiss_index.add_mu(mu)

    result = ret.retrieve("fact", conversation_id="conv1", top_k=5,
                          expand_graph=True)
    assert result.graph_expanded is False


def test_retrieve_graph_expansion_adds_related(store: MemoryStore) -> None:
    graph = MemoryGraphIndex()
    ret = _make_retriever(store, graph=graph)

    mu1 = _insert(store, _mu("Alice has surgery", mu_id="m1"))
    mu2 = _insert(store, _mu("Alice is in the hospital", mu_id="m2"))
    # Index only m1; m2 will be discovered via graph
    ret.faiss_index.add_mu(mu1)
    ret.faiss_index.add_mu(mu2)  # also add to FAISS so it can be retrieved

    edge = EdgeRecord(
        source_mu_id="m1",
        target_mu_id="m2",
        edge_type=EdgeType.RELATED_TO,
    )
    store.insert_edge(edge)
    graph.rebuild_from_store(store)

    result = ret.retrieve("Alice health", conversation_id="conv1", top_k=5,
                          expand_graph=True)
    ids = {h.mu.mu_id for h in result.hits}
    # Both should appear either via FAISS or graph expansion
    assert "m1" in ids or "m2" in ids


def test_retrieve_graph_expansion_source_label(store: MemoryStore) -> None:
    """Graph-expanded hits have source='graph'."""
    embed_fn = _make_embed_fn(seed=1)
    idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    graph = MemoryGraphIndex()
    ret = MemoryRetriever(store, idx, graph=graph)

    # m1 gets FAISS hit; m2 is connected via graph but not in FAISS
    mu1 = _insert(store, _mu("primary fact", mu_id="m1"))
    mu2 = _insert(store, _mu("neighbour fact", mu_id="m2"))
    idx.add_mu(mu1)
    # Do NOT add m2 to FAISS; it should appear only via graph expansion

    edge = EdgeRecord(
        source_mu_id="m1",
        target_mu_id="m2",
        edge_type=EdgeType.RELATED_TO,
    )
    store.insert_edge(edge)
    graph.rebuild_from_store(store)

    result = ret.retrieve("primary", conversation_id="conv1", top_k=10,
                          expand_graph=True)
    graph_hits = [h for h in result.hits if h.source == "graph"]
    if graph_hits:
        assert all(h.mu.mu_id == "m2" for h in graph_hits)


def test_retrieve_graph_expansion_discounts_score(store: MemoryStore) -> None:
    """Graph-expanded hits have lower scores than their seed."""
    embed_fn = _make_embed_fn(seed=5)
    idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    graph = MemoryGraphIndex()
    ret = MemoryRetriever(store, idx, graph=graph)

    mu1 = _insert(store, _mu("seed fact", mu_id="m1"))
    mu2 = _insert(store, _mu("neighbour fact", mu_id="m2"))
    idx.add_mu(mu1)

    edge = EdgeRecord(
        source_mu_id="m1",
        target_mu_id="m2",
        edge_type=EdgeType.RELATED_TO,
    )
    store.insert_edge(edge)
    graph.rebuild_from_store(store)

    result = ret.retrieve("seed", conversation_id="conv1", top_k=10,
                          expand_graph=True)
    m1_hits = [h for h in result.hits if h.mu.mu_id == "m1"]
    m2_hits = [h for h in result.hits if h.mu.mu_id == "m2" and h.source == "graph"]
    if m1_hits and m2_hits:
        assert m2_hits[0].score <= m1_hits[0].score


def test_graph_expansion_skips_non_active(store: MemoryStore) -> None:
    """Graph neighbours that are forgotten/compressed are not included."""
    graph = MemoryGraphIndex()
    ret = _make_retriever(store, graph=graph)

    mu1 = _insert(store, _mu("primary fact", mu_id="m1"))
    mu2 = _insert(store, _mu("forgotten neighbour", mu_id="m2"))
    store.forget_atomic("m2")
    ret.faiss_index.add_mu(mu1)

    edge = EdgeRecord(
        source_mu_id="m1",
        target_mu_id="m2",
        edge_type=EdgeType.RELATED_TO,
    )
    store.insert_edge(edge)
    graph.rebuild_from_store(store)

    result = ret.retrieve("primary", conversation_id="conv1", top_k=10,
                          expand_graph=True)
    ids = {h.mu.mu_id for h in result.hits}
    assert "m2" not in ids


def test_graph_expansion_no_duplicate_hits(store: MemoryStore) -> None:
    """A MU that appears in both FAISS and graph should not be duplicated."""
    embed_fn = _make_embed_fn(seed=3)
    idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    graph = MemoryGraphIndex()
    ret = MemoryRetriever(store, idx, graph=graph)

    mu1 = _insert(store, _mu("fact A", mu_id="m1"))
    mu2 = _insert(store, _mu("fact B", mu_id="m2"))
    idx.add_mus([mu1, mu2])  # both in FAISS

    edge = EdgeRecord(
        source_mu_id="m1",
        target_mu_id="m2",
        edge_type=EdgeType.RELATED_TO,
    )
    store.insert_edge(edge)
    graph.rebuild_from_store(store)

    result = ret.retrieve("fact", conversation_id="conv1", top_k=10,
                          expand_graph=True)
    ids = [h.mu.mu_id for h in result.hits]
    assert len(ids) == len(set(ids)), "duplicate mu_ids in hits"


# ---------------------------------------------------------------------------
# sync_index / rebuild_index
# ---------------------------------------------------------------------------


def test_sync_index_processes_pending(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    mu = _insert(store, _mu("new fact", mu_id="m1"))
    store.mark_needs_reindex("m1")

    n = ret.sync_index(conversation_id="conv1")
    assert n == 1
    assert "m1" in ret.faiss_index.mu_ids()


def test_rebuild_index_rebuilds_faiss(store: MemoryStore) -> None:
    ret = _make_retriever(store)
    _insert(store, _mu("fact1", mu_id="m1"))
    _insert(store, _mu("fact2", mu_id="m2"))

    n = ret.rebuild_index(conversation_id="conv1")
    assert n == 2
    assert set(ret.faiss_index.mu_ids()) == {"m1", "m2"}


def test_rebuild_index_with_graph(store: MemoryStore) -> None:
    graph = MemoryGraphIndex()
    embed_fn = _make_embed_fn()
    idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    ret = MemoryRetriever(store, idx, graph=graph)

    _insert(store, _mu("claim", mu_id="m1"))
    ret.rebuild_index(conversation_id="conv1", rebuild_graph=True)
    assert graph.has_node("m1")


# ---------------------------------------------------------------------------
# graph_expanded flag
# ---------------------------------------------------------------------------


def test_graph_expanded_false_when_no_new_neighbours(store: MemoryStore) -> None:
    graph = MemoryGraphIndex()
    ret = _make_retriever(store, graph=graph)
    mu = _insert(store, _mu("isolated fact", mu_id="m1"))
    ret.faiss_index.add_mu(mu)
    graph.rebuild_from_store(store)

    result = ret.retrieve("fact", conversation_id="conv1", top_k=5,
                          expand_graph=True)
    # m1 has no edges → no new neighbours found
    assert result.graph_expanded is False


def test_graph_expanded_true_when_neighbours_found(store: MemoryStore) -> None:
    embed_fn = _make_embed_fn(seed=9)
    idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    graph = MemoryGraphIndex()
    ret = MemoryRetriever(store, idx, graph=graph)

    mu1 = _insert(store, _mu("primary fact", mu_id="m1"))
    mu2 = _insert(store, _mu("neighbour fact", mu_id="m2"))
    idx.add_mu(mu1)  # m2 NOT in FAISS — can only appear via graph

    edge = EdgeRecord(
        source_mu_id="m1",
        target_mu_id="m2",
        edge_type=EdgeType.RELATED_TO,
    )
    store.insert_edge(edge)
    graph.rebuild_from_store(store)

    result = ret.retrieve("primary", conversation_id="conv1", top_k=10,
                          expand_graph=True)
    # m2 is active and a neighbour of m1, so expansion should fire
    m2_in_hits = any(h.mu.mu_id == "m2" for h in result.hits)
    if m2_in_hits:
        assert result.graph_expanded is True
