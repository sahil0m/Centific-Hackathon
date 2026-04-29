"""Tests for the NetworkX graph index.

The graph is a derived index over the SQLite edges table. These tests verify
that the interface works correctly in isolation, and that it can be
faithfully rebuilt from a SQLite source of truth — including recovery from
out-of-sync state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from locomo_memory.phase2.schemas import (
    EdgeRecord,
    EdgeType,
    MemoryUnit,
)
from locomo_memory.phase2.store.graph_index import MemoryGraphIndex
from locomo_memory.phase2.store.sqlite_store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_mu(claim: str = "x", conv: str = "conv_1") -> MemoryUnit:
    return MemoryUnit(
        conversation_id=conv,
        session_id="s1",
        claim=claim,
        original_text="t",
        source_dia_ids=["D1:1"],
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


@pytest.fixture
def graph() -> MemoryGraphIndex:
    return MemoryGraphIndex()


# ---------------------------------------------------------------------------
# Empty graph
# ---------------------------------------------------------------------------


class TestEmptyGraph:
    def test_initial_state(self, graph: MemoryGraphIndex) -> None:
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert graph.has_node("mu_x") is False
        assert graph.neighbors("mu_x") == set()
        assert graph.k_hop_neighbors("mu_x", k=1) == set()
        assert graph.degree("mu_x") == 0
        assert graph.degree_centrality() == {}
        assert graph.betweenness_centrality() == {}
        assert graph.node_attrs("mu_x") == {}

    def test_invalid_k_hop(self, graph: MemoryGraphIndex) -> None:
        with pytest.raises(ValueError):
            graph.k_hop_neighbors("mu_x", k=0)
        with pytest.raises(ValueError):
            graph.k_hop_neighbors("mu_x", k=-1)

    def test_repr(self, graph: MemoryGraphIndex) -> None:
        assert "nodes=0" in repr(graph)
        assert "edges=0" in repr(graph)


# ---------------------------------------------------------------------------
# Node mutations
# ---------------------------------------------------------------------------


class TestNodeMutation:
    def test_upsert_node_adds(self, graph: MemoryGraphIndex) -> None:
        graph.upsert_node("mu_a", salience=0.7, conversation_id="conv_1")
        assert graph.has_node("mu_a")
        attrs = graph.node_attrs("mu_a")
        assert attrs["salience"] == 0.7
        assert attrs["conversation_id"] == "conv_1"

    def test_upsert_node_updates_existing(self, graph: MemoryGraphIndex) -> None:
        graph.upsert_node("mu_a", salience=0.5)
        graph.upsert_node("mu_a", salience=0.9, status="active")
        attrs = graph.node_attrs("mu_a")
        assert attrs["salience"] == 0.9
        assert attrs["status"] == "active"

    def test_upsert_empty_id_rejected(self, graph: MemoryGraphIndex) -> None:
        with pytest.raises(ValueError):
            graph.upsert_node("")

    def test_remove_node(self, graph: MemoryGraphIndex) -> None:
        graph.upsert_node("mu_a")
        assert graph.has_node("mu_a")
        graph.remove_node("mu_a")
        assert not graph.has_node("mu_a")

    def test_remove_nonexistent_node_silent(self, graph: MemoryGraphIndex) -> None:
        graph.remove_node("mu_ghost")  # should not raise

    def test_remove_node_drops_incident_edges(self, graph: MemoryGraphIndex) -> None:
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        ))
        assert graph.edge_count() == 1
        graph.remove_node("mu_a")
        assert graph.edge_count() == 0


# ---------------------------------------------------------------------------
# Edge mutations
# ---------------------------------------------------------------------------


class TestEdgeMutation:
    def test_add_edge_auto_creates_endpoints(self, graph: MemoryGraphIndex) -> None:
        edge = EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        )
        graph.add_edge(edge)
        assert graph.has_node("mu_a")
        assert graph.has_node("mu_b")
        assert graph.edge_count() == 1

    def test_remove_edge(self, graph: MemoryGraphIndex) -> None:
        edge = EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        )
        graph.add_edge(edge)
        removed = graph.remove_edge(edge.edge_id)
        assert removed is True
        assert graph.edge_count() == 0

    def test_remove_nonexistent_edge_returns_false(self, graph: MemoryGraphIndex) -> None:
        assert graph.remove_edge("edg_ghost") is False

    def test_multiple_edges_between_same_nodes(self, graph: MemoryGraphIndex) -> None:
        e1 = EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        )
        e2 = EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.SUPERSEDED_BY,
        )
        graph.add_edge(e1)
        graph.add_edge(e2)
        assert graph.edge_count() == 2


# ---------------------------------------------------------------------------
# Neighbor queries
# ---------------------------------------------------------------------------


class TestNeighbors:
    def test_one_hop_combines_in_and_out(self, graph: MemoryGraphIndex) -> None:
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        ))
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_c", target_mu_id="mu_a",
            edge_type=EdgeType.RELATED_TO,
        ))
        neigh = graph.neighbors("mu_a")
        assert neigh == {"mu_b", "mu_c"}

    def test_neighbors_filter_by_type(self, graph: MemoryGraphIndex) -> None:
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        ))
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_c",
            edge_type=EdgeType.SUPERSEDED_BY,
        ))
        related = graph.neighbors("mu_a", edge_type=EdgeType.RELATED_TO)
        assert related == {"mu_b"}
        superseded = graph.neighbors("mu_a", edge_type=EdgeType.SUPERSEDED_BY)
        assert superseded == {"mu_c"}

    def test_k_hop_neighbors_chain(self, graph: MemoryGraphIndex) -> None:
        for src, tgt in [("mu_a", "mu_b"), ("mu_b", "mu_c"), ("mu_c", "mu_d")]:
            graph.add_edge(EdgeRecord(
                source_mu_id=src, target_mu_id=tgt,
                edge_type=EdgeType.RELATED_TO,
            ))
        assert graph.k_hop_neighbors("mu_a", k=1) == {"mu_b"}
        assert graph.k_hop_neighbors("mu_a", k=2) == {"mu_b", "mu_c"}
        assert graph.k_hop_neighbors("mu_a", k=3) == {"mu_b", "mu_c", "mu_d"}
        assert graph.k_hop_neighbors("mu_a", k=10) == {"mu_b", "mu_c", "mu_d"}

    def test_k_hop_excludes_origin(self, graph: MemoryGraphIndex) -> None:
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        ))
        assert "mu_a" not in graph.k_hop_neighbors("mu_a", k=1)


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


class TestCentrality:
    def test_degree_centrality_chain(self, graph: MemoryGraphIndex) -> None:
        # a -> b -> c -> d (linear chain)
        for src, tgt in [("mu_a", "mu_b"), ("mu_b", "mu_c"), ("mu_c", "mu_d")]:
            graph.add_edge(EdgeRecord(
                source_mu_id=src, target_mu_id=tgt,
                edge_type=EdgeType.RELATED_TO,
            ))
        dc = graph.degree_centrality()
        # b and c have degree 2; a and d have degree 1
        assert dc["mu_b"] > dc["mu_a"]
        assert dc["mu_c"] > dc["mu_d"]

    def test_betweenness_runs_without_error(self, graph: MemoryGraphIndex) -> None:
        for src, tgt in [("mu_a", "mu_b"), ("mu_b", "mu_c")]:
            graph.add_edge(EdgeRecord(
                source_mu_id=src, target_mu_id=tgt,
                edge_type=EdgeType.RELATED_TO,
            ))
        bc = graph.betweenness_centrality()
        assert "mu_b" in bc
        assert bc["mu_b"] >= bc["mu_a"]


# ---------------------------------------------------------------------------
# Edges by type
# ---------------------------------------------------------------------------


class TestEdgesByType:
    def test_filter_by_type(self, graph: MemoryGraphIndex) -> None:
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_a", target_mu_id="mu_b",
            edge_type=EdgeType.RELATED_TO,
        ))
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_b", target_mu_id="mu_c",
            edge_type=EdgeType.SUPERSEDED_BY,
        ))
        related = graph.edges_by_type(EdgeType.RELATED_TO)
        superseded = graph.edges_by_type(EdgeType.SUPERSEDED_BY)
        assert len(related) == 1
        assert len(superseded) == 1
        assert related[0][0] == "mu_a"
        assert superseded[0][0] == "mu_b"


# ---------------------------------------------------------------------------
# Rebuild from SQLite source of truth
# ---------------------------------------------------------------------------


class TestRebuildFromStore:
    def test_rebuild_empty_store(
        self, store: MemoryStore, graph: MemoryGraphIndex
    ) -> None:
        graph.rebuild_from_store(store)
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_rebuild_populated_store(
        self, store: MemoryStore, graph: MemoryGraphIndex
    ) -> None:
        a = _make_mu("a")
        b = _make_mu("b")
        c = _make_mu("c")
        for m in (a, b, c):
            store.insert_memory_unit(m)
        store.insert_edge(EdgeRecord(
            source_mu_id=a.mu_id, target_mu_id=b.mu_id,
            edge_type=EdgeType.RELATED_TO,
        ))
        store.insert_edge(EdgeRecord(
            source_mu_id=b.mu_id, target_mu_id=c.mu_id,
            edge_type=EdgeType.SUPERSEDED_BY,
        ))

        graph.rebuild_from_store(store)

        assert graph.node_count() == 3
        assert graph.edge_count() == 2
        assert graph.has_node(a.mu_id)
        assert graph.has_node(b.mu_id)
        assert graph.has_node(c.mu_id)

    def test_rebuild_drops_stale_in_memory_state(
        self, store: MemoryStore, graph: MemoryGraphIndex
    ) -> None:
        # Populate SQLite truth
        a = _make_mu("a")
        store.insert_memory_unit(a)
        # Add stale node + edge to in-memory graph that is NOT in SQLite
        graph.upsert_node("mu_stale")
        graph.add_edge(EdgeRecord(
            source_mu_id="mu_stale", target_mu_id="mu_other",
            edge_type=EdgeType.RELATED_TO,
        ))
        # Rebuild — stale entries gone
        graph.rebuild_from_store(store)
        assert graph.has_node("mu_stale") is False
        assert graph.has_node("mu_other") is False
        assert graph.has_node(a.mu_id)

    def test_rebuild_preserves_node_attrs(
        self, store: MemoryStore, graph: MemoryGraphIndex
    ) -> None:
        mu = _make_mu("a", conv="conv_42")
        mu.salience_score = 0.81
        mu.user_pinned = True
        store.insert_memory_unit(mu)

        graph.rebuild_from_store(store)

        attrs = graph.node_attrs(mu.mu_id)
        assert attrs["status"] == "active"
        assert abs(attrs["salience_score"] - 0.81) < 1e-6
        assert attrs["conversation_id"] == "conv_42"
        assert attrs["user_pinned"] is True

    def test_rebuild_includes_compressed_and_forgotten(
        self, store: MemoryStore, graph: MemoryGraphIndex
    ) -> None:
        # Forgotten MU should still be in graph (relationship history retained)
        mu = _make_mu("a")
        store.insert_memory_unit(mu)
        store.forget_atomic(mu.mu_id)
        graph.rebuild_from_store(store)
        assert graph.has_node(mu.mu_id)
        assert graph.node_attrs(mu.mu_id)["status"] == "forgotten"

    def test_rebuild_after_store_mutation(
        self, store: MemoryStore, graph: MemoryGraphIndex
    ) -> None:
        a = _make_mu("a")
        store.insert_memory_unit(a)
        graph.rebuild_from_store(store)
        assert graph.node_count() == 1

        b = _make_mu("b")
        store.insert_memory_unit(b)
        # Graph still has just a until rebuild
        assert graph.node_count() == 1

        graph.rebuild_from_store(store)
        assert graph.node_count() == 2
