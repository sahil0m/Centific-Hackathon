"""Contradiction Resolver with Provenance — Phase 2 Milestone 7.

Compares MemoryUnit claims using NLI (Natural Language Inference) as the
primary signal, with rule-based pattern matching for update/temporal
categorisation in the NLI-neutral zone.  Creates EdgeRecords in the store
for provenance tracking.  No LLM call required at inference time (the NLI
model is a pre-trained cross-encoder loaded once).

Research basis
--------------
NLI / Textual Entailment is the standard NLP approach for determining whether
one sentence contradicts, entails, or is neutral with respect to another
(Bowman et al. 2015 — SNLI; Williams et al. 2018 — MultiNLI).  Using
``cross-encoder/nli-deberta-v3-large`` (He et al. 2021) gives state-of-the-art
zero-shot contradiction detection without hand-crafted keyword lists.

Hybrid design: NLI + rules
---------------------------
NLI is used as the primary signal for contradiction and same-fact detection
because it reads both claims jointly and captures semantic polarity (negation,
antonyms, scope).  Rules remain necessary for categorising the NLI-neutral
zone into UPDATED_FACT / TEMPORAL_CHANGE / RELATED / UNRELATED, because these
categories depend on domain-specific markers (update verbs, temporal adverbs)
rather than logical entailment alone.

Relationship taxonomy
---------------------
SAME_FACT        NLI entailment ≥ 0.70 (and no update verb) — near-duplicate.
UPDATED_FACT     NLI neutral zone + update verb present on same topic/entity.
TEMPORAL_CHANGE  NLI neutral zone + temporal marker present on same topic/entity.
CONTRADICTION    NLI contradiction ≥ 0.70 — newer claim negates older semantically.
RELATED          Same topic or moderate token overlap — no stronger signal.
UNRELATED        No meaningful overlap or topic match.

Conservative design
-------------------
When NLI is uncertain (all scores < threshold), the resolver falls back to
rules and prefers RELATED over a false-positive CONTRADICTION.

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
    r"\b("
    # Job / employer changes
    r"joined|started\s+(at|working|a\s+new)|began\s+working|now\s+works|"
    r"now\s+working|currently\s+works|currently\s+working|now\s+employed|"
    r"recently\s+(joined|started|hired)|accepted\s+(a|an)\s+(new\s+)?job|"
    r"took\s+(a|an)\s+(new\s+)?job|got\s+(a|an)\s+(new\s+)?job|"
    r"is\s+now\s+(at|working|employed)|left\s+for\s+(a\s+new\s+)?|"
    r"hired\s+(at|by|as)|promoted\s+(to|as)|switched\s+(to|companies)|"
    r"changed\s+(to|companies|jobs?)|transferred\s+to|"
    # Location changes
    r"moved\s+(to|into|back\s+to)|relocated\s+(to|from)|"
    r"now\s+lives|living\s+(in|at)\s+(?:a\s+new|my\s+new)|"
    # Relationship changes
    r"got\s+married|got\s+engaged|got\s+divorced|"
    r"is\s+now\s+(married|engaged|single|dating)|"
    r"started\s+dating|broke\s+up|separated\s+from|"
    # Education
    r"enrolled\s+in|graduated\s+from|completed\s+(a|his|her|their)|"
    # Generic update markers
    r"is\s+now|now\s+is|recently\s+became|became\s+(a|an|the)"
    r")\b",
    _I,
)

_TEMPORAL_RE = re.compile(
    r"\b(used\s+to|previously|formerly|back\s+when|at\s+that\s+time"
    r"|in\s+the\s+past|until\s+recently|at\s+one\s+point)\b",
    _I,
)

# NLI decision thresholds — probabilities from the cross-encoder
_NLI_CONTRADICTION_THRESHOLD: float = 0.70
_NLI_ENTAILMENT_THRESHOLD: float = 0.70

# Jaccard thresholds for rule-based fallback (NLI-neutral zone)
_SAME_FACT_J: float = 0.60
_RELATED_J: float = 0.10


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
    """NLI-primary contradiction detection with provenance edge creation.

    Uses ``cross-encoder/nli-deberta-v3-large`` as the primary signal for
    contradiction and same-fact detection, with rule-based pattern matching
    for UPDATED_FACT / TEMPORAL_CHANGE classification in the NLI-neutral zone.

    Args:
        store: the SQLite-backed :class:`~locomo_memory.phase2.store.sqlite_store.MemoryStore`.
        nli_classifier: optional pre-constructed NLI classifier.  If ``None``,
            :class:`~locomo_memory.phase2.contradiction.nli_classifier.NLIContradictionClassifier`
            is lazy-loaded on the first :meth:`compare` call.  Pass a
            :class:`~locomo_memory.phase2.contradiction.nli_classifier.FakeNLIClassifier`
            in unit tests to avoid downloading the model.

    All ``compare*`` methods are pure (no DB access).
    ``create_edges_for``, ``resolve_incoming``, ``resolve_mu``, and
    ``scan_conversation`` write to the store.
    """

    def __init__(self, store: MemoryStore, *, nli_classifier=None) -> None:
        self.store = store
        self._estimator = TopicImportanceEstimator()
        self._nli = nli_classifier  # None → lazy-loaded real model on first use

    def _get_nli(self):
        if self._nli is None:
            try:
                from locomo_memory.phase2.contradiction.nli_classifier import (
                    NLIContradictionClassifier,
                )
                clf = NLIContradictionClassifier()
                clf._load()  # trigger model download now so any failure is caught here
                self._nli = clf
            except Exception as exc:
                from locomo_memory.phase2.contradiction.nli_classifier import FakeNLIClassifier
                logger.warning(
                    "NLI model unavailable ({}), falling back to heuristic classifier", exc
                )
                self._nli = FakeNLIClassifier()
        return self._nli

    # ------------------------------------------------------------------
    # Core comparison  (pure — no DB access)
    # ------------------------------------------------------------------

    def compare(self, mu_a: MemoryUnit, mu_b: MemoryUnit) -> ComparisonResult:
        """Classify the relationship between two MUs using NLI + rules.

        Convention: ``mu_a`` is the existing/older claim; ``mu_b`` is the
        incoming/newer one.  Pattern matching is applied to ``mu_b``'s claim.

        Decision order
        --------------
        1. NLI contradiction ≥ 0.70 + negation in B + Jaccard ≥ 0.25  → CONTRADICTION
        2. NLI entailment   ≥ 0.70 + same_topic + update verb          → UPDATED_FACT
        3. NLI entailment   ≥ 0.70 + same_topic                        → SAME_FACT
        3x. NLI entailment  ≥ 0.70 + different topics                  → RELATED (guard)
        4. (NLI neutral zone) Update verb + same_topic                  → UPDATED_FACT
        5. (NLI neutral zone) Implicit update (same high-value topic, diff entities) → UPDATED_FACT
        6. (NLI neutral zone) Temporal marker + same_topic              → TEMPORAL_CHANGE
        7. Same topic or Jaccard ≥ 0.10                                 → RELATED
        8. Otherwise                                                     → UNRELATED

        All supersession-capable paths (steps 2–6) require ``same_topic``.
        Entity overlap alone is never sufficient to trigger supersession.
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

        has_update = bool(_UPDATE_VERB_RE.search(claim_b))
        has_temporal = bool(_TEMPORAL_RE.search(claim_b))
        has_negation_b = bool(_NEGATION_RE.search(claim_b))

        # --- 1. NLI primary signal ---
        nli = self._get_nli().classify(claim_a, claim_b)

        # --- 2. NLI contradiction → CONTRADICTION ---
        # The NLI model treats any two claims that cannot both be true
        # simultaneously as "contradiction" — this includes updates ("joined
        # Microsoft") and temporal changes ("used to live in Paris"), which
        # are logically incompatible with the older claim but belong to
        # different categories in our taxonomy.
        #
        # Guard: require an explicit negation word in claim_b AND sufficient
        # token overlap (≥ 0.25).  Updates / temporal changes have no
        # negation, so they fall through to the rule-based zone below.
        # Different-object negations ("doesn't like sushi" vs "likes pizza")
        # have low overlap and are also excluded.
        if (
            nli.contradiction >= _NLI_CONTRADICTION_THRESHOLD
            and has_negation_b
            and j >= 0.25
        ):
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.CONTRADICTION,
                confidence=nli.contradiction,
                reason=(
                    f"NLI contradiction={nli.contradiction:.2f} + negation in B; "
                    f"Jaccard={j:.2f}, entity_overlap={ent_ov}, same_topic={same_topic}"
                ),
            )

        # --- 3. NLI entailment → SAME_FACT or UPDATED_FACT ---
        # Guard: require SAME TOPIC before treating high-entailment claims as
        # the same/updated fact.  Using entity overlap (ent_ov ≥ 1) as an OR
        # bypass is intentionally removed — the DeBERTa NLI model returns high
        # entailment for any two positive statements about the same person
        # (e.g. "graduated from IIT" vs "works at Centific") because they share
        # an implicit subject.  After cleaning generic words ("The", pronouns)
        # from entity extraction the ent_ov signal is more reliable, but topic
        # agreement is still required to prevent cross-domain supersession.
        if nli.entailment >= _NLI_ENTAILMENT_THRESHOLD:
            if not same_topic:
                # High NLI entailment but different topics — the model is
                # capturing a shared subject, not true semantic equivalence.
                # Downgrade to RELATED; no supersession edge is written.
                return ComparisonResult(
                    mu_a_id=mu_a.mu_id,
                    mu_b_id=mu_b.mu_id,
                    relationship=RelationshipType.RELATED,
                    confidence=nli.entailment * 0.5,
                    reason=(
                        f"NLI entailment={nli.entailment:.2f} but topic mismatch "
                        f"({topic_a}≠{topic_b}); downgraded to RELATED"
                    ),
                )
            # Within the same topic, an update verb means the newer claim
            # supersedes rather than merely duplicates the older one.
            if has_update:
                return ComparisonResult(
                    mu_a_id=mu_a.mu_id,
                    mu_b_id=mu_b.mu_id,
                    relationship=RelationshipType.UPDATED_FACT,
                    confidence=nli.entailment,
                    reason=(
                        f"NLI entailment={nli.entailment:.2f} + update verb; "
                        f"topic={topic_a}, entity_overlap={ent_ov}"
                    ),
                )
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.SAME_FACT,
                confidence=nli.entailment,
                reason=(
                    f"NLI entailment={nli.entailment:.2f}; topic={topic_a}, "
                    f"Jaccard={j:.2f}, entity_overlap={ent_ov}"
                ),
            )

        # --- 4. NLI-neutral zone: rule-based categorisation ---

        # 4a. UPDATED_FACT — explicit update verb in newer claim.
        # Requires same_topic: an update verb ("graduated from", "moved to",
        # "joined") only supersedes facts within THE SAME life domain.
        # Accepting ent_ov ≥ 1 as a bypass was the root cause of cross-topic
        # supersession when "The" polluted the entity overlap signal.
        if has_update and same_topic:
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.UPDATED_FACT,
                confidence=0.75,
                reason=(
                    f"update verb in newer claim; topic={topic_a} (same), "
                    f"entity_overlap={ent_ov}"
                ),
            )

        # 4b. UPDATED_FACT (implicit) — same high-importance topic, different
        #     named entities, low token overlap.  Catches "works at Centific" →
        #     "works at Microsoft" without an explicit update verb.
        _IMPLICIT_UPDATE_TOPICS = {"employment", "location", "relationships"}
        if (
            same_topic
            and topic_a in _IMPLICIT_UPDATE_TOPICS
            and j < _SAME_FACT_J
            and j < 0.30
            and len(ent_a) >= 1
            and len(ent_b) >= 1
            and _entity_overlap(ent_a, ent_b) == 0
        ):
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.UPDATED_FACT,
                confidence=0.68,
                reason=(
                    f"implicit update: same topic '{topic_a}', different entities "
                    f"({ent_a} → {ent_b}), Jaccard={j:.2f}"
                ),
            )

        # 4c. TEMPORAL_CHANGE — temporal marker in newer claim.
        # Requires same_topic for the same reason as 4a: "I used to play chess"
        # should only create a temporal edge against other lifestyle/hobby facts,
        # not against location or employment facts that happen to share an entity.
        if has_temporal and same_topic:
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.TEMPORAL_CHANGE,
                confidence=0.70,
                reason=(
                    f"temporal marker in newer claim; topic={topic_a} (same), "
                    f"entity_overlap={ent_ov}"
                ),
            )

        # 4d. RELATED — same topic or moderate token overlap
        if same_topic or j >= _RELATED_J:
            return ComparisonResult(
                mu_a_id=mu_a.mu_id,
                mu_b_id=mu_b.mu_id,
                relationship=RelationshipType.RELATED,
                confidence=max(j, 0.3 if same_topic else 0.0),
                reason=f"same_topic={same_topic}, Jaccard={j:.2f}",
            )

        # 4e. UNRELATED
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
