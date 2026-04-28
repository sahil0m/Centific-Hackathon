"""NetworkX-backed graph of MemoryUnit relationships (derived index).

This is a prototype-grade graph layer for the Phase 2 demo and benchmark. It
is **not** a transactional store — the SQLite ``edges`` table is the source
of truth. This index is rebuilt from SQLite on startup and updated
incrementally as edges are inserted or removed elsewhere in the system.

For production scale (millions of MUs across many users), migrate this layer
to Neo4j or Memgraph. The public API of ``MemoryGraphIndex`` is intentionally
narrow so a future driver swap is local.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from loguru import logger

from locomo_memory.phase2.schemas import EdgeRecord, EdgeType
from locomo_memory.phase2.store.sqlite_store import MemoryStore


class MemoryGraphIndex:
    """In-memory directed multi-graph of Memory Unit relationships.

    Nodes
    -----
    Each node is keyed by ``mu_id``. Node attributes mirror useful columns
    from the ``memory_units`` table (``conversation_id``, ``status``,
    ``salience_score``, ``user_pinned``) for fast filtering.

    Edges
    -----
    Edges are typed (see :class:`EdgeType`) and directed. The graph allows
    multiple parallel edges between the same pair of nodes as long as they
    differ in type — this matches the SQLite UNIQUE constraint on
    ``(source_mu_id, target_mu_id, edge_type)``.
    """

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Build / rebuild from SQLite (the source of truth)
    # ------------------------------------------------------------------

    def rebuild_from_store(self, store: MemoryStore) -> None:
        """Drop the in-memory graph and rebuild from SQLite.

        Iterates every Memory Unit (regardless of status — relationship history
        is retained even for compressed/forgotten MUs) and every edge. After
        this call the graph exactly mirrors the SQLite truth.
        """
        logger.info("Rebuilding graph index from SQLite store: {}", store.db_path)
        new_graph: nx.MultiDiGraph = nx.MultiDiGraph()

        with store.reader() as conn:
            cursor = conn.execute(
                "SELECT mu_id, conversation_id, status, salience_score, user_pinned "
                "FROM memory_units"
            )
            for row in cursor:
                new_graph.add_node(
                    row["mu_id"],
                    conversation_id=row["conversation_id"],
                    status=row["status"],
                    salience_score=row["salience_score"],
                    user_pinned=bool(row["user_pinned"]),
                )

        # Replace the live graph atomically (no half-built state visible).
        self.graph = new_graph

        # Add edges (auto-creates endpoints if the MU rows reference unknowns,
        # which would indicate a referential issue worth surfacing).
        edge_count = 0
        for edge in store.iter_edges():
            self.add_edge(edge)
            edge_count += 1

        logger.info(
            "Graph rebuilt: {} nodes, {} edges",
            self.graph.number_of_nodes(),
            edge_count,
        )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def upsert_node(self, mu_id: str, **attrs: Any) -> None:
        """Insert a node, or merge attributes onto an existing one."""
        if not mu_id:
            raise ValueError("mu_id must be non-empty")
        if self.graph.has_node(mu_id):
            self.graph.nodes[mu_id].update(attrs)
        else:
            self.graph.add_node(mu_id, **attrs)

    def remove_node(self, mu_id: str) -> None:
        """Remove a node and all incident edges. No-op if absent."""
        if self.graph.has_node(mu_id):
            self.graph.remove_node(mu_id)

    def add_edge(self, edge: EdgeRecord) -> None:
        """Add a typed edge. Endpoints are auto-created if missing."""
        if not self.graph.has_node(edge.source_mu_id):
            self.graph.add_node(edge.source_mu_id)
        if not self.graph.has_node(edge.target_mu_id):
            self.graph.add_node(edge.target_mu_id)
        self.graph.add_edge(
            edge.source_mu_id,
            edge.target_mu_id,
            key=edge.edge_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            metadata_json=edge.metadata_json,
        )

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge by id. Returns True if removed, False if not found."""
        for u, v, key in list(self.graph.edges(keys=True)):
            if key == edge_id:
                self.graph.remove_edge(u, v, key=key)
                return True
        return False

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def has_node(self, mu_id: str) -> bool:
        return self.graph.has_node(mu_id)

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def node_attrs(self, mu_id: str) -> dict[str, Any]:
        """Return the node attributes for a given MU (copy)."""
        if not self.graph.has_node(mu_id):
            return {}
        return dict(self.graph.nodes[mu_id])

    def neighbors(
        self, mu_id: str, edge_type: EdgeType | None = None
    ) -> set[str]:
        """Return the set of direct (1-hop) neighbors, undirected.

        If ``edge_type`` is given, only edges of that type are followed.
        """
        if not self.graph.has_node(mu_id):
            return set()
        result: set[str] = set()
        for _, target, attrs in self.graph.out_edges(mu_id, data=True):
            if edge_type is None or attrs.get("edge_type") == edge_type.value:
                result.add(target)
        for source, _, attrs in self.graph.in_edges(mu_id, data=True):
            if edge_type is None or attrs.get("edge_type") == edge_type.value:
                result.add(source)
        return result

    def k_hop_neighbors(self, mu_id: str, k: int = 1) -> set[str]:
        """All distinct nodes within k hops (undirected). Excludes ``mu_id``."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not self.graph.has_node(mu_id):
            return set()
        ug = self.graph.to_undirected(as_view=True)
        seen: set[str] = {mu_id}
        frontier: set[str] = {mu_id}
        for _ in range(k):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(ug.neighbors(node))
            new = next_frontier - seen
            if not new:
                break
            seen.update(new)
            frontier = new
        seen.discard(mu_id)
        return seen

    def degree(self, mu_id: str) -> int:
        """In-degree + out-degree (parallel edges count separately)."""
        if not self.graph.has_node(mu_id):
            return 0
        return self.graph.degree(mu_id)  # type: ignore[return-value]

    def degree_centrality(self) -> dict[str, float]:
        """Return NetworkX degree centrality for all nodes."""
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.degree_centrality(self.graph)

    def betweenness_centrality(self, k: int | None = None) -> dict[str, float]:
        """Approximate betweenness centrality (sample size ``k`` for speed)."""
        if self.graph.number_of_nodes() == 0:
            return {}
        # NetworkX returns floats; we cast for clarity.
        return dict(nx.betweenness_centrality(self.graph, k=k, normalized=True))

    def edges_by_type(self, edge_type: EdgeType) -> list[tuple[str, str, str]]:
        """List of (source_mu_id, target_mu_id, edge_id) for the given type."""
        return [
            (u, v, key)
            for u, v, key, attrs in self.graph.edges(keys=True, data=True)
            if attrs.get("edge_type") == edge_type.value
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MemoryGraphIndex(nodes={self.node_count()}, "
            f"edges={self.edge_count()})"
        )


__all__ = ["MemoryGraphIndex"]
