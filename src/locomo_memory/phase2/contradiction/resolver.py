"""Contradiction Resolver with Provenance — Phase 2 Milestone 7.

Compares MemoryUnit claims using rule-based token overlap, topic
classification, and pattern matching.  Creates EdgeRecords in the store for
provenance tracking.  No LLM call required.

Relationship taxonomy
---------------------
SAME_FACT        High token Jaccard (≥0.70) — near-duplicate or identical claim.
UPDATED_FACT     Newer claim supersedes older on same topic (update verb present).
TEMPORAL_CHANGE  Newer claim explicitly references a past state of the same topic.
CONTRADICTION    Newer claim negates older on same entities/topic.
RELATED          Same topic or moderate overlap — no clear update or conflict.
UNRELATED        No meaningful overlap.

Conservative design
-------------------
When uncertain the resolver prefers RELATED or UNRELATED over a false-positive
CONTRADICTION.  CONTRADICTION requires *both* a negation pattern in the newer
claim *and* sufficiently high token Jaccard (≥0.25) or strong entity overlap
(≥2 shared entities).

Edge policy
-----------
SAME_FACT / UPDATED_FACT / TEMPORAL_CHANGE  →  old_mu -[SUPERSEDED_BY]→ new_mu
CONTRADICTION                                →  old_mu -[CONFLICTS_WITH]→ new_mu
                                                new_mu -[CONFLICTS_WITH]→ old_mu
RELATED                                      →  old_mu -[RELATED_TO]→ new_mu
UNRELATED                                    →  no edge

Both MUs always keep their provenance; status is never changed here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from locomo_memory.phase2.ingestion.importance import TopicImportanceEstimator
from locomo_memory.phase2.schemas import (
    EdgeRecord,
    EdgeType,
    MemoryUnit,
)
from locomo_memory.phase2.store.sqlite_store import (
    MemoryStore,
    MemoryStoreError,
    MemoryUnitNotFoundError,
)


# ---------------------------------------------------------------------------
# Stop words (Jaccard only — semantic negation words excluded intentionally)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "his", "its", "our", "your", "their", "that", "this", "these", "those",
    "what", "which", "who", "when", "where", "how", "so", "very", "just",
    "also", "then", "than", "more", "most",
})

# ---------------------------------------------------------------------------
# Compiled rule patterns  (checked against the raw claim text, not tokens)
# ---------------------------------------------------------------------------

_I = re.IGNORECASE

_NEGATION_RE = re.compile(
    r"\b(no longer|never|not|isn't|are\s+n't|aren't|wasn't|weren't"
    r"|doesn't|don't|didn't|quit|resigned|left|divorced|broke\s+up|stopped)\b",
    _I,
)

_UPDATE_VERB_RE = re.compile(
    r"\b(joined|moved\s+to|relocated\s+to|is\s+now|now\s+works|now\s+lives"
    r"|changed\s+to|switched\s+to|transferred\s+to|started\s+at|enrolled\s+in"
    r"|graduated\s+from|got\s+married|got\s+engaged|had\s+a\s+baby|hired|promoted)\b",
    _I,
)

_TEMPORAL_RE = re.compile(
    r"\b(used\s+to|previously|formerly|back\s+when|at\s+that\s+time"
    r"|in\s+the\s+past|until\s+recently|at\s+one\s+point)\b",
    _I,
)

# Jaccard thresholds
_SAME_FACT_J: float = 0.70
_RELATED_J: float = 0.10
# CONTRADICTION: must clear this OR have ≥2 shared entities
_CONTRADICTION_J: float = 0.25
_CONTRADICTION_ENTITY: int = 2


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RelationshipType(str, Enum):
    """Classified relationship between two MemoryUnit claims."""

    SAME_FACT = "same_fact"
    UPDATED_FACT = "updated_fact"
    TEMPORAL_CHANGE = "temporal_change"
    CONTRADICTION = "contradiction"
    RELATED = "related"
    UNRELATED = "unrelated"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ComparisonResult:
    """Outcome of comparing two claims."""

    mu_a_id: str
    """Existing/older MU."""
    mu_b_id: str
    """Incoming/newer MU."""
    relationship: RelationshipType
    confidence: float
    """Heuristic confidence in [0, 1]."""
    reason: str
    """Short human-readable explanation (for logging / debugging)."""


@dataclass(slots=True)
class ResolutionAction:
    """One edge-write attempt made during resolution."""

    mu_id: str
    """Source MU of the edge attempt."""
    action: str
    """'edge_created' | 'edge_exists' | 'edge_failed'"""
    edge: EdgeRecord | None = None
    error: str | None = None


@dataclass(slots=True)
class ResolutionResult:
    """Full output of resolving one incoming MU against a set of candidates."""

    incoming_mu_id: str
    comparisons: list[ComparisonResult] = field(default_factory=list)
    actions: list[ResolutionAction] = field(default_factory=list)

    @property
    def n_conflicts(self) -> int:
        return sum(
            1 for c in self.comparisons
            if c.relationship == RelationshipType.CONTRADICTION
        )

    @property
    def n_updates(self) -> int:
        return sum(
            1 for c in self.comparisons
            if c.relationship in (
                RelationshipType.UPDATED_FACT,
                RelationshipType.SAME_FACT,
                RelationshipType.TEMPORAL_CHANGE,
            )
        )

    @property
    def n_related(self) -> int:
        return sum(
            1 for c in self.comparisons
            if c.relationship == RelationshipType.RELATED
        )

    @property
    def edges_created(self) -> int:
        return sum(1 for a in self.actions if a.action == "edge_created")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase, strip punctuation, split, remove stop words."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return frozenset(t for t in cleaned.split() if t not in _STOP_WORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _entity_overlap(entities_a: list[str], entities_b: list[str]) -> int:
    set_a = {e.lower() for e in entities_a}
    set_b = {e.lower() for e in entities_b}
    return len(set_a & set_b)


def _meta_json(comp: ComparisonResult) -> str:
    return json.dumps({
        "relationship": comp.relationship.value,
        "confidence": comp.confidence,
        "reason": comp.reason,
    })


# ---------------------------------------------------------------------------
# ContradictionResolver
# ---------------------------------------------------------------------------


class ContradictionResolver:
    """Rule-based contradiction detection and provenance edge creation.

    Args:
        store: the SQLite-backed :class:`~locomo_memory.phase2.store.sqlite_store.MemoryStore`.

    All ``compare*`` methods are pure (no DB access).
    ``create_edges_for``, ``resolve_incoming``, ``resolve_mu``, and
    ``scan_conversation`` write to the store.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._estimator = TopicImportanceEstimator()

    # ------------------------------------------------------------------
    # Core comparison  (pure — no DB access)
    # ------------------------------------------------------------------

    def compare(self, mu_a: MemoryUnit, mu_b: MemoryUnit) -> ComparisonResult:
        """Classify the relationship between two MUs.

        Convention: ``mu_a`` is the existing/older claim; ``mu_b`` is the
        incoming/newer one.  Patterns are checked in mu_b's claim text.

        Rules (applied in priority order)
        ----------------------------------
        1. Jaccard ≥ 0.70                         → SAME_FACT
        2. Negation in mu_b + strong overlap      → CONTRADICTION
        3. Update verb in mu_b + topic/entity hit → UPDATED_FACT
        4. Temporal marker in mu_b + topic hit    → TEMPORAL_CHANGE
        5. Same topic or Jaccard ≥ 0.10           → RELATED
        6. Otherwise                              → UNRELATED
        """
        claim_a = mu_a.claim
        claim_b = mu_b.claim

        tok_a = _tokenize(claim_a)
        tok_b = _tokenize(claim_b)
        j = _jaccard(tok_a, tok_b)

        topic_a = self._estimator.detect_topic(claim_a)
        topic_b = self._estimator.detect_topic(claim_b)
        same_topic = (topic_a == topic_b) and (topic_a != "general")

        ent_a = self._estimator.extract_entities(claim_a)
        ent_b = self._estimator.extract_entities(claim_b)
        ent_ov = _entity_overlap(ent_a, ent_b)

        # 1. SAME_FACT
        if j >= _SAME_FACT_J:
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.SAME_FACT,
                confidence=min(1.0, j),
                reason=f"high Jaccard={j:.2f}",
            )

        # 2. CONTRADICTION — negation in newer + (high-j OR ≥2 shared entities)
        #    AND at least one anchor: entity overlap OR same topic.
        #    Conservative: requires substantial evidence to avoid false positives
        #    from incidental shared tokens (e.g. "John likes pizza" vs
        #    "John doesn't like sushi" share only the entity "John" and have
        #    Jaccard < 0.25, so they correctly resolve to RELATED not CONTRADICTION).
        has_negation = bool(_NEGATION_RE.search(claim_b))
        if has_negation:
            strong_overlap = j >= _CONTRADICTION_J or ent_ov >= _CONTRADICTION_ENTITY
            anchored = ent_ov >= 1 or same_topic
            if strong_overlap and anchored:
                conf = min(1.0, 0.6 + 0.1 * min(ent_ov, 3) + 0.1 * float(same_topic))
                return ComparisonResult(
                    mu_a_id=mu_a.mu_id,
                    mu_b_id=mu_b.mu_id,
                    relationship=RelationshipType.CONTRADICTION,
                    confidence=conf,
                    reason=(
                        f"negation in newer claim; Jaccard={j:.2f}, "
                        f"entity_overlap={ent_ov}, same_topic={same_topic}"
                    ),
                )

        # 3. UPDATED_FACT — update verb in newer + topic or entity anchor
        has_update = bool(_UPDATE_VERB_RE.search(claim_b))
        if has_update and (same_topic or ent_ov >= 1):
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.UPDATED_FACT,
                confidence=0.75,
                reason=(
                    f"update verb in newer claim; same_topic={same_topic}, "
                    f"entity_overlap={ent_ov}"
                ),
            )

        # 4. TEMPORAL_CHANGE — temporal marker in newer + topic or entity anchor
        has_temporal = bool(_TEMPORAL_RE.search(claim_b))
        if has_temporal and (same_topic or ent_ov >= 1):
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.TEMPORAL_CHANGE,
                confidence=0.70,
                reason=(
                    f"temporal marker in newer claim; same_topic={same_topic}, "
                    f"entity_overlap={ent_ov}"
                ),
            )

        # 5. RELATED
        if same_topic or j >= _RELATED_J:
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.RELATED,
                confidence=max(j, 0.3 if same_topic else 0.0),
                reason=f"same_topic={same_topic}, Jaccard={j:.2f}",
            )

        # 6. UNRELATED
        return ComparisonResult(
            mu_a_id=mu_a.mu_id,
            mu_b_id=mu_b.mu_id,
            relationship=RelationshipType.UNRELATED,
            confidence=max(0.0, 1.0 - j),
            reason=f"low overlap; Jaccard={j:.2f}, same_topic={same_topic}",
        )

    def compare_all(
        self,
        incoming: MemoryUnit,
        candidates: list[MemoryUnit],
    ) -> list[ComparisonResult]:
        """Compare incoming against all candidates. No DB writes.

        Returns one :class:`ComparisonResult` per candidate in the same order
        as ``candidates``.  ``incoming`` is treated as the newer MU in each pair.
        """
        return [self.compare(candidate, incoming) for candidate in candidates]

    # ------------------------------------------------------------------
    # Edge creation
    # ------------------------------------------------------------------

    def create_edges_for(
        self,
        incoming: MemoryUnit,
        comparisons: list[ComparisonResult],
    ) -> list[ResolutionAction]:
        """Persist EdgeRecords for all non-UNRELATED comparisons.

        Silently skips duplicate edges (returns action='edge_exists').
        Captures store errors without raising.
        """
        actions: list[ResolutionAction] = []
        for comp in comparisons:
            if comp.relationship == RelationshipType.UNRELATED:
                continue
            actions.extend(self._edges_for_comparison(comp))
        return actions

    def _edges_for_comparison(
        self,
        comp: ComparisonResult,
    ) -> list[ResolutionAction]:
        actions: list[ResolutionAction] = []
        meta = _meta_json(comp)

        if comp.relationship in (
            RelationshipType.SAME_FACT,
            RelationshipType.UPDATED_FACT,
            RelationshipType.TEMPORAL_CHANGE,
        ):
            # old_mu → new_mu : SUPERSEDED_BY
            edge = EdgeRecord(
                source_mu_id=comp.mu_a_id,
                target_mu_id=comp.mu_b_id,
                edge_type=EdgeType.SUPERSEDED_BY,
                weight=comp.confidence,
                metadata_json=meta,
            )
            actions.append(self._try_insert(edge))

        elif comp.relationship == RelationshipType.CONTRADICTION:
            # Bidirectional CONFLICTS_WITH
            edge_fwd = EdgeRecord(
                source_mu_id=comp.mu_a_id,
                target_mu_id=comp.mu_b_id,
                edge_type=EdgeType.CONFLICTS_WITH,
                weight=comp.confidence,
                metadata_json=meta,
            )
            edge_rev = EdgeRecord(
                source_mu_id=comp.mu_b_id,
                target_mu_id=comp.mu_a_id,
                edge_type=EdgeType.CONFLICTS_WITH,
                weight=comp.confidence,
                metadata_json=meta,
            )
            actions.append(self._try_insert(edge_fwd))
            actions.append(self._try_insert(edge_rev))

        elif comp.relationship == RelationshipType.RELATED:
            edge = EdgeRecord(
                source_mu_id=comp.mu_a_id,
                target_mu_id=comp.mu_b_id,
                edge_type=EdgeType.RELATED_TO,
                weight=comp.confidence,
                metadata_json=meta,
            )
            actions.append(self._try_insert(edge))

        return actions

    def _try_insert(self, edge: EdgeRecord) -> ResolutionAction:
        try:
            self.store.insert_edge(edge)
            logger.debug(
                "ContradictionResolver: {} -[{}]-> {}",
                edge.source_mu_id, edge.edge_type.value, edge.target_mu_id,
            )
            return ResolutionAction(
                mu_id=edge.source_mu_id, action="edge_created", edge=edge,
            )
        except MemoryStoreError as exc:
            if "Duplicate" in str(exc):
                return ResolutionAction(
                    mu_id=edge.source_mu_id, action="edge_exists", edge=edge,
                )
            return ResolutionAction(
                mu_id=edge.source_mu_id, action="edge_failed", error=str(exc),
            )

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def resolve_incoming(
        self,
        incoming_mu_id: str,
        *,
        candidate_mu_ids: list[str] | None = None,
    ) -> ResolutionResult:
        """Compare an incoming MU against candidates and create edges.

        If ``candidate_mu_ids`` is ``None``, scans all active MUs in the same
        conversation (excluding the incoming MU itself).

        Args:
            incoming_mu_id: the MU being ingested/compared.
            candidate_mu_ids: explicit list of MU ids to compare against, or
                ``None`` to auto-select all active MUs in the conversation.

        Returns:
            :class:`ResolutionResult` with comparisons and edge actions taken.

        Raises:
            MemoryUnitNotFoundError: if ``incoming_mu_id`` does not exist.
        """
        incoming = self.store.get_memory_unit(incoming_mu_id)
        if incoming is None:
            raise MemoryUnitNotFoundError(incoming_mu_id)

        if candidate_mu_ids is None:
            candidates = [
                mu for mu in self.store.list_active(incoming.conversation_id)
                if mu.mu_id != incoming_mu_id
            ]
        else:
            candidates = []
            for cid in candidate_mu_ids:
                if cid == incoming_mu_id:
                    continue
                mu = self.store.get_memory_unit(cid)
                if mu is not None:
                    candidates.append(mu)

        return self.resolve_mu(incoming, candidates)

    def resolve_mu(
        self,
        incoming: MemoryUnit,
        candidates: list[MemoryUnit],
    ) -> ResolutionResult:
        """Compare incoming against a list of candidate MUs and create edges.

        Unlike :meth:`resolve_incoming` this does not raise if individual
        candidates are missing — ``candidates`` is a pre-fetched list.
        """
        comparisons = self.compare_all(incoming, candidates)
        relevant = [
            c for c in comparisons
            if c.relationship != RelationshipType.UNRELATED
        ]
        actions = self.create_edges_for(incoming, relevant)

        logger.info(
            "ContradictionResolver: mu={} vs {} candidates: "
            "{} conflict(s), {} update(s), {} related",
            incoming.mu_id,
            len(candidates),
            sum(1 for c in comparisons if c.relationship == RelationshipType.CONTRADICTION),
            sum(
                1 for c in comparisons
                if c.relationship in (
                    RelationshipType.UPDATED_FACT,
                    RelationshipType.SAME_FACT,
                    RelationshipType.TEMPORAL_CHANGE,
                )
            ),
            sum(1 for c in comparisons if c.relationship == RelationshipType.RELATED),
        )

        return ResolutionResult(
            incoming_mu_id=incoming.mu_id,
            comparisons=comparisons,
            actions=actions,
        )

    def scan_conversation(
        self,
        conversation_id: str,
    ) -> list[ResolutionResult]:
        """Scan all active MUs in a conversation pairwise for contradictions.

        Iterates MUs in creation order. For each MU (index i), compares it
        against all prior MUs (indices 0..i-1) and creates edges for non-UNRELATED
        relationships.  The first MU has no prior candidates and is skipped.

        Useful for a one-time consistency pass after bulk ingestion.  Returns
        one :class:`ResolutionResult` per MU that had at least one candidate.
        """
        mus = self.store.list_active(conversation_id)
        results: list[ResolutionResult] = []
        for i, mu in enumerate(mus):
            prior = mus[:i]
            if not prior:
                continue
            result = self.resolve_mu(mu, prior)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ComparisonResult",
    "ContradictionResolver",
    "RelationshipType",
    "ResolutionAction",
    "ResolutionResult",
]
