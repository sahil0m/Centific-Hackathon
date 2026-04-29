"""Tests for Phase 2 Milestone 7: ContradictionResolver with Provenance."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from locomo_memory.phase2.contradiction.resolver import (
    ComparisonResult,
    ContradictionResolver,
    RelationshipType,
    ResolutionAction,
    ResolutionResult,
    _jaccard,
    _tokenize,
)
from locomo_memory.phase2.schemas import (
    EdgeType,
    MemoryStatus,
    MemoryUnit,
)
from locomo_memory.phase2.store.sqlite_store import MemoryStore, MemoryUnitNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


@pytest.fixture()
def resolver(store: MemoryStore) -> ContradictionResolver:
    return ContradictionResolver(store)


def _mu(
    claim: str,
    *,
    conv: str = "conv1",
    session: str = "s1",
    mu_id: str | None = None,
) -> MemoryUnit:
    kwargs: dict = dict(conversation_id=conv, session_id=session, claim=claim)
    if mu_id is not None:
        kwargs["mu_id"] = mu_id
    return MemoryUnit(**kwargs)


def _insert(store: MemoryStore, mu: MemoryUnit) -> MemoryUnit:
    store.insert_memory_unit(mu)
    return mu


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_tokenize_removes_stop_words() -> None:
    tokens = _tokenize("I work at Google")
    assert "i" not in tokens
    assert "at" not in tokens
    assert "work" in tokens
    assert "google" in tokens


def test_tokenize_removes_punctuation() -> None:
    tokens = _tokenize("He's married, really!")
    assert "he's" not in tokens or "hes" in tokens  # apostrophe stripped


def test_tokenize_empty_string() -> None:
    assert _tokenize("") == frozenset()


def test_jaccard_identical() -> None:
    s = frozenset({"a", "b", "c"})
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint() -> None:
    assert _jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_partial() -> None:
    a = frozenset({"a", "b"})
    b = frozenset({"b", "c"})
    assert abs(_jaccard(a, b) - 1 / 3) < 1e-9


def test_jaccard_empty_sets() -> None:
    assert _jaccard(frozenset(), frozenset()) == 0.0


# ---------------------------------------------------------------------------
# compare() — SAME_FACT
# ---------------------------------------------------------------------------


def test_compare_same_fact_identical(resolver: ContradictionResolver) -> None:
    a = _mu("Alice works at Acme Corp")
    b = _mu("Alice works at Acme Corp", mu_id="mu_b")
    result = resolver.compare(a, b)
    assert result.relationship == RelationshipType.SAME_FACT
    assert result.confidence >= 0.9


def test_compare_same_fact_high_overlap(resolver: ContradictionResolver) -> None:
    # "graduated from" is an update verb → would trigger UPDATED_FACT.
    # Use a case with no update verb so high-Jaccard SAME_FACT fires instead.
    a = _mu("Alice works at Google headquarters in the city")
    b = _mu("Alice works at Google headquarters", mu_id="mu_b")
    result = resolver.compare(a, b)
    assert result.relationship == RelationshipType.SAME_FACT


def test_compare_same_fact_ids(resolver: ContradictionResolver) -> None:
    a = _mu("The cat sat on the mat", mu_id="mu_a")
    b = _mu("The cat sat on the mat", mu_id="mu_b")
    result = resolver.compare(a, b)
    assert result.mu_a_id == "mu_a"
    assert result.mu_b_id == "mu_b"


# ---------------------------------------------------------------------------
# compare() — UPDATED_FACT
# ---------------------------------------------------------------------------


def test_compare_updated_fact_employment(resolver: ContradictionResolver) -> None:
    old = _mu("I work at Google")
    new = _mu("I joined Microsoft", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.UPDATED_FACT


def test_compare_updated_fact_location(resolver: ContradictionResolver) -> None:
    old = _mu("She lives in Boston")
    new = _mu("She moved to Seattle last month", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.UPDATED_FACT


def test_compare_updated_fact_relationship(resolver: ContradictionResolver) -> None:
    old = _mu("They are dating")
    new = _mu("They got married last weekend", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.UPDATED_FACT


def test_compare_updated_fact_confidence_range(resolver: ContradictionResolver) -> None:
    old = _mu("He works at Acme")
    new = _mu("He joined BetaCorp", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.UPDATED_FACT
    assert 0.0 < result.confidence <= 1.0


# ---------------------------------------------------------------------------
# compare() — TEMPORAL_CHANGE
# ---------------------------------------------------------------------------


def test_compare_temporal_change(resolver: ContradictionResolver) -> None:
    old = _mu("He lives in Paris")
    new = _mu("He used to live in Paris, now in London", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.TEMPORAL_CHANGE


def test_compare_temporal_previously(resolver: ContradictionResolver) -> None:
    old = _mu("She works as a teacher")
    new = _mu("She previously worked as a teacher", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.TEMPORAL_CHANGE


def test_compare_temporal_formerly(resolver: ContradictionResolver) -> None:
    old = _mu("He is married to Jane")
    new = _mu("He was formerly married, now divorced", mu_id="mu_new")
    # formerly + same relationship topic
    result = resolver.compare(old, new)
    assert result.relationship in (
        RelationshipType.TEMPORAL_CHANGE,
        RelationshipType.CONTRADICTION,  # "divorced" is also a negation marker
    )


# ---------------------------------------------------------------------------
# compare() — CONTRADICTION
# ---------------------------------------------------------------------------


def test_compare_contradiction_no_longer(resolver: ContradictionResolver) -> None:
    old = _mu("I work at Google")
    new = _mu("I no longer work at Google", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.CONTRADICTION


def test_compare_contradiction_negation_not(resolver: ContradictionResolver) -> None:
    old = _mu("She lives in New York")
    new = _mu("She does not live in New York anymore", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.CONTRADICTION


def test_compare_contradiction_never(resolver: ContradictionResolver) -> None:
    old = _mu("He graduated from MIT")
    new = _mu("He never graduated from MIT", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.CONTRADICTION


def test_compare_contradiction_confidence_range(resolver: ContradictionResolver) -> None:
    old = _mu("I work at Google")
    new = _mu("I no longer work at Google", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert 0.5 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# compare() — NOT a contradiction (health example from CLAUDE.md)
# ---------------------------------------------------------------------------


def test_compare_health_not_contradiction(resolver: ContradictionResolver) -> None:
    """Surgery and cold are both health claims but do NOT contradict."""
    old = _mu("I have surgery next week")
    new = _mu("I have a cold, suggest cold medicine", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship != RelationshipType.CONTRADICTION


def test_compare_different_objects_not_contradiction(
    resolver: ContradictionResolver,
) -> None:
    """Low token Jaccard with one shared entity → not a contradiction."""
    old = _mu("John likes pizza")
    new = _mu("John doesn't like sushi", mu_id="mu_new")
    result = resolver.compare(old, new)
    # Should be RELATED (same entity "John") or UNRELATED — not CONTRADICTION
    assert result.relationship != RelationshipType.CONTRADICTION


# ---------------------------------------------------------------------------
# compare() — RELATED
# ---------------------------------------------------------------------------


def test_compare_related_same_topic(resolver: ContradictionResolver) -> None:
    # Both claims trigger the "health" topic pattern (surgery / hospital).
    old = _mu("He had surgery last month")
    new = _mu("He is still recovering in the hospital", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.RELATED


def test_compare_related_moderate_jaccard(resolver: ContradictionResolver) -> None:
    old = _mu("Alice likes hiking and outdoor activities")
    new = _mu("Alice enjoys hiking on weekends", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship in (RelationshipType.RELATED, RelationshipType.SAME_FACT)


def test_compare_related_same_entity(resolver: ContradictionResolver) -> None:
    old = _mu("John lives in Chicago")
    new = _mu("John commutes to work in Chicago daily", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship in (RelationshipType.RELATED, RelationshipType.SAME_FACT)


# ---------------------------------------------------------------------------
# compare() — UNRELATED
# ---------------------------------------------------------------------------


def test_compare_unrelated(resolver: ContradictionResolver) -> None:
    old = _mu("I prefer coffee in the morning")
    new = _mu("The weather is nice today", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.UNRELATED


def test_compare_unrelated_short_claims(resolver: ContradictionResolver) -> None:
    old = _mu("Cats are great pets")
    new = _mu("He works at IBM", mu_id="mu_new")
    result = resolver.compare(old, new)
    assert result.relationship == RelationshipType.UNRELATED


# ---------------------------------------------------------------------------
# compare() — general contract
# ---------------------------------------------------------------------------


def test_compare_result_has_reason(resolver: ContradictionResolver) -> None:
    a = _mu("I work at Google")
    b = _mu("I joined Microsoft", mu_id="mu_b")
    result = resolver.compare(a, b)
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


def test_compare_confidence_in_range(resolver: ContradictionResolver) -> None:
    pairs = [
        ("I work at Google", "I joined Microsoft"),
        ("I work at Google", "I no longer work at Google"),
        ("I have surgery", "I have a cold"),
        ("Coffee is good", "The sky is blue"),
    ]
    for claim_a, claim_b in pairs:
        result = resolver.compare(_mu(claim_a), _mu(claim_b, mu_id="mu_b"))
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence out of range for ({claim_a!r}, {claim_b!r}): "
            f"{result.confidence}"
        )


# ---------------------------------------------------------------------------
# compare_all()
# ---------------------------------------------------------------------------


def test_compare_all_returns_one_per_candidate(
    resolver: ContradictionResolver,
) -> None:
    incoming = _mu("I joined Microsoft", mu_id="incoming")
    candidates = [
        _mu("I work at Google", mu_id="c1"),
        _mu("I live in Seattle", mu_id="c2"),
        _mu("I have a cat", mu_id="c3"),
    ]
    results = resolver.compare_all(incoming, candidates)
    assert len(results) == 3
    assert all(r.mu_b_id == "incoming" for r in results)
    assert [r.mu_a_id for r in results] == ["c1", "c2", "c3"]


def test_compare_all_empty_candidates(resolver: ContradictionResolver) -> None:
    incoming = _mu("I work at Google", mu_id="inc")
    assert resolver.compare_all(incoming, []) == []


# ---------------------------------------------------------------------------
# create_edges_for() — edge creation
# ---------------------------------------------------------------------------


def test_create_edges_superseded_by(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I work at Google", mu_id="old"))
    new = _insert(store, _mu("I joined Microsoft", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.UPDATED_FACT,
        confidence=0.75,
        reason="test",
    )
    actions = resolver.create_edges_for(new, [comp])
    assert len(actions) == 1
    assert actions[0].action == "edge_created"
    assert actions[0].edge is not None
    assert actions[0].edge.edge_type == EdgeType.SUPERSEDED_BY
    assert actions[0].edge.source_mu_id == "old"
    assert actions[0].edge.target_mu_id == "new"


def test_create_edges_same_fact_superseded_by(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("Alice works at Acme", mu_id="old"))
    new = _insert(store, _mu("Alice works at Acme", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.SAME_FACT,
        confidence=1.0,
        reason="test",
    )
    actions = resolver.create_edges_for(new, [comp])
    assert actions[0].edge.edge_type == EdgeType.SUPERSEDED_BY


def test_create_edges_temporal_superseded_by(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("He lives in Paris", mu_id="old"))
    new = _insert(store, _mu("He used to live in Paris", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.TEMPORAL_CHANGE,
        confidence=0.70,
        reason="test",
    )
    actions = resolver.create_edges_for(new, [comp])
    assert actions[0].edge.edge_type == EdgeType.SUPERSEDED_BY


def test_create_edges_contradiction_bidirectional(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I work at Google", mu_id="old"))
    new = _insert(store, _mu("I no longer work at Google", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.CONTRADICTION,
        confidence=0.8,
        reason="test",
    )
    actions = resolver.create_edges_for(new, [comp])
    assert len(actions) == 2
    assert all(a.edge.edge_type == EdgeType.CONFLICTS_WITH for a in actions)
    sources = {a.edge.source_mu_id for a in actions}
    targets = {a.edge.target_mu_id for a in actions}
    assert sources == {"old", "new"}
    assert targets == {"old", "new"}


def test_create_edges_related_to(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("He has diabetes", mu_id="old"))
    new = _insert(store, _mu("He takes insulin daily", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.RELATED,
        confidence=0.4,
        reason="test",
    )
    actions = resolver.create_edges_for(new, [comp])
    assert len(actions) == 1
    assert actions[0].edge.edge_type == EdgeType.RELATED_TO


def test_create_edges_unrelated_no_edge(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I like coffee", mu_id="old"))
    new = _insert(store, _mu("The sky is blue", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.UNRELATED,
        confidence=0.9,
        reason="test",
    )
    actions = resolver.create_edges_for(new, [comp])
    assert actions == []


def test_create_edges_duplicate_returns_edge_exists(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I work at Google", mu_id="old"))
    new = _insert(store, _mu("I joined Microsoft", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.UPDATED_FACT,
        confidence=0.75,
        reason="test",
    )
    actions1 = resolver.create_edges_for(new, [comp])
    assert actions1[0].action == "edge_created"

    actions2 = resolver.create_edges_for(new, [comp])
    assert actions2[0].action == "edge_exists"


def test_create_edges_metadata_json_stored(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I work at Google", mu_id="old"))
    new = _insert(store, _mu("I joined Microsoft", mu_id="new"))
    comp = ComparisonResult(
        mu_a_id="old",
        mu_b_id="new",
        relationship=RelationshipType.UPDATED_FACT,
        confidence=0.75,
        reason="update verb: joined",
    )
    actions = resolver.create_edges_for(new, [comp])
    assert actions[0].edge.metadata_json is not None
    import json
    meta = json.loads(actions[0].edge.metadata_json)
    assert meta["relationship"] == "updated_fact"
    assert "reason" in meta


# ---------------------------------------------------------------------------
# resolve_incoming()
# ---------------------------------------------------------------------------


def test_resolve_incoming_not_found_raises(resolver: ContradictionResolver) -> None:
    with pytest.raises(MemoryUnitNotFoundError):
        resolver.resolve_incoming("nonexistent_id")


def test_resolve_incoming_no_candidates(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    mu = _insert(store, _mu("I work at Google", mu_id="only"))
    result = resolver.resolve_incoming("only")
    assert result.incoming_mu_id == "only"
    assert result.comparisons == []
    assert result.actions == []


def test_resolve_incoming_auto_selects_same_conversation(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I work at Google", mu_id="old", conv="conv1"))
    new = _insert(store, _mu("I joined Microsoft", mu_id="new", conv="conv1"))
    # Also insert a MU in a different conversation — should be ignored
    other = _insert(store, _mu("I live in Tokyo", mu_id="other", conv="conv2"))

    result = resolver.resolve_incoming("new")
    candidate_ids = {c.mu_a_id for c in result.comparisons}
    assert "old" in candidate_ids
    assert "other" not in candidate_ids


def test_resolve_incoming_excludes_self(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    mu = _insert(store, _mu("I work at Google", mu_id="self"))
    result = resolver.resolve_incoming("self")
    assert all(c.mu_a_id != "self" for c in result.comparisons)


def test_resolve_incoming_explicit_candidates(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old1 = _insert(store, _mu("I work at Google", mu_id="c1"))
    old2 = _insert(store, _mu("I live in NYC", mu_id="c2"))
    new = _insert(store, _mu("I joined Microsoft", mu_id="inc"))
    result = resolver.resolve_incoming("inc", candidate_mu_ids=["c1"])
    assert len(result.comparisons) == 1
    assert result.comparisons[0].mu_a_id == "c1"


def test_resolve_incoming_creates_correct_edges(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I work at Google", mu_id="old"))
    new = _insert(store, _mu("I joined Microsoft", mu_id="new"))
    result = resolver.resolve_incoming("new")
    employment_comp = next(
        (c for c in result.comparisons if c.mu_a_id == "old"), None
    )
    assert employment_comp is not None
    assert employment_comp.relationship == RelationshipType.UPDATED_FACT
    assert result.edges_created >= 1


def test_resolve_incoming_contradiction_edges(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    old = _insert(store, _mu("I work at Google", mu_id="old"))
    new = _insert(store, _mu("I no longer work at Google", mu_id="new"))
    result = resolver.resolve_incoming("new")
    assert result.n_conflicts == 1
    # Two CONFLICTS_WITH edges (bidirectional)
    assert result.edges_created == 2


# ---------------------------------------------------------------------------
# resolve_mu()
# ---------------------------------------------------------------------------


def test_resolve_mu_empty_candidates(resolver: ContradictionResolver) -> None:
    mu = _mu("I work at Google", mu_id="inc")
    result = resolver.resolve_mu(mu, [])
    assert result.incoming_mu_id == "inc"
    assert result.comparisons == []
    assert result.actions == []


def test_resolve_mu_returns_all_comparisons(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    candidates = [
        _mu("I work at Google", mu_id="c1"),
        _mu("I live in NYC", mu_id="c2"),
        _mu("I have a dog", mu_id="c3"),
    ]
    incoming = _mu("I joined Microsoft", mu_id="inc")
    result = resolver.resolve_mu(incoming, candidates)
    assert len(result.comparisons) == 3


# ---------------------------------------------------------------------------
# scan_conversation()
# ---------------------------------------------------------------------------


def test_scan_conversation_single_mu_no_results(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    _insert(store, _mu("I work at Google", mu_id="m1"))
    results = resolver.scan_conversation("conv1")
    assert results == []


def test_scan_conversation_two_mus(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    _insert(store, _mu("I work at Google", mu_id="m1"))
    _insert(store, _mu("I joined Microsoft", mu_id="m2"))
    results = resolver.scan_conversation("conv1")
    assert len(results) == 1
    assert results[0].incoming_mu_id == "m2"
    assert len(results[0].comparisons) == 1


def test_scan_conversation_creates_edges(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    _insert(store, _mu("I work at Google", mu_id="m1"))
    _insert(store, _mu("I no longer work at Google", mu_id="m2"))
    resolver.scan_conversation("conv1")
    edges = store.edges_from("m1", EdgeType.CONFLICTS_WITH)
    assert len(edges) == 1


def test_scan_conversation_three_mus(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    _insert(store, _mu("I work at Google", mu_id="m1"))
    _insert(store, _mu("I live in NYC", mu_id="m2"))
    _insert(store, _mu("I joined Microsoft", mu_id="m3"))
    # m3 compared against m1, m2
    results = resolver.scan_conversation("conv1")
    assert len(results) == 2  # m2 vs [m1], m3 vs [m1, m2]


def test_scan_conversation_idempotent_edges(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    _insert(store, _mu("I work at Google", mu_id="m1"))
    _insert(store, _mu("I no longer work at Google", mu_id="m2"))
    resolver.scan_conversation("conv1")
    resolver.scan_conversation("conv1")  # second call — no crash, edge_exists
    edges = store.edges_from("m1", EdgeType.CONFLICTS_WITH)
    assert len(edges) == 1  # no duplicates


def test_scan_conversation_ignores_non_active(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    mu1 = _insert(store, _mu("I work at Google", mu_id="m1"))
    mu2 = _insert(store, _mu("I joined Microsoft", mu_id="m2"))
    store.forget_atomic("m1")  # m1 is now FORGOTTEN
    results = resolver.scan_conversation("conv1")
    # m2 is alone in active — no candidates
    assert results == []


# ---------------------------------------------------------------------------
# ResolutionResult properties
# ---------------------------------------------------------------------------


def test_resolution_result_n_conflicts() -> None:
    result = ResolutionResult(
        incoming_mu_id="inc",
        comparisons=[
            ComparisonResult("a", "inc", RelationshipType.CONTRADICTION, 0.8, ""),
            ComparisonResult("b", "inc", RelationshipType.UPDATED_FACT, 0.7, ""),
        ],
    )
    assert result.n_conflicts == 1
    assert result.n_updates == 1
    assert result.n_related == 0


def test_resolution_result_n_updates_includes_same_and_temporal() -> None:
    result = ResolutionResult(
        incoming_mu_id="inc",
        comparisons=[
            ComparisonResult("a", "inc", RelationshipType.SAME_FACT, 1.0, ""),
            ComparisonResult("b", "inc", RelationshipType.TEMPORAL_CHANGE, 0.7, ""),
            ComparisonResult("c", "inc", RelationshipType.UPDATED_FACT, 0.75, ""),
        ],
    )
    assert result.n_updates == 3


def test_resolution_result_edges_created() -> None:
    from locomo_memory.phase2.schemas import EdgeRecord, EdgeType
    edge = EdgeRecord(
        source_mu_id="a", target_mu_id="b", edge_type=EdgeType.SUPERSEDED_BY
    )
    result = ResolutionResult(
        incoming_mu_id="b",
        actions=[
            ResolutionAction(mu_id="a", action="edge_created", edge=edge),
            ResolutionAction(mu_id="a", action="edge_exists", edge=edge),
        ],
    )
    assert result.edges_created == 1


def test_resolution_result_n_related() -> None:
    result = ResolutionResult(
        incoming_mu_id="inc",
        comparisons=[
            ComparisonResult("a", "inc", RelationshipType.RELATED, 0.4, ""),
            ComparisonResult("b", "inc", RelationshipType.RELATED, 0.3, ""),
            ComparisonResult("c", "inc", RelationshipType.UNRELATED, 0.9, ""),
        ],
    )
    assert result.n_related == 2


# ---------------------------------------------------------------------------
# End-to-end: employer update scenario from CLAUDE.md
# ---------------------------------------------------------------------------


def test_employer_update_scenario(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    """Session 1: I work at Google → Session 10: I joined Microsoft.

    Google MU should get SUPERSEDED_BY edge pointing to Microsoft MU.
    Both MUs remain active (status unchanged by resolver).
    """
    google_mu = _insert(store, _mu("I work at Google", mu_id="google_mu"))
    microsoft_mu = _insert(store, _mu("I joined Microsoft", mu_id="ms_mu"))

    result = resolver.resolve_incoming("ms_mu")

    comp = next(c for c in result.comparisons if c.mu_a_id == "google_mu")
    assert comp.relationship == RelationshipType.UPDATED_FACT

    edges = store.edges_from("google_mu", EdgeType.SUPERSEDED_BY)
    assert len(edges) == 1
    assert edges[0].target_mu_id == "ms_mu"

    # Both MUs keep their provenance — status unchanged
    assert store.get_memory_unit("google_mu").status == MemoryStatus.ACTIVE
    assert store.get_memory_unit("ms_mu").status == MemoryStatus.ACTIVE


def test_health_context_not_contradiction(
    resolver: ContradictionResolver, store: MemoryStore
) -> None:
    """Surgery and cold are related health context, not a contradiction."""
    surgery_mu = _insert(
        store, _mu("I have surgery next week", mu_id="surgery")
    )
    cold_mu = _insert(
        store, _mu("I have a cold, suggest cold medicine", mu_id="cold")
    )
    result = resolver.resolve_incoming("cold")
    comp = next(c for c in result.comparisons if c.mu_a_id == "surgery")
    assert comp.relationship != RelationshipType.CONTRADICTION
    # No CONFLICTS_WITH edges created
    assert len(store.edges_from("surgery", EdgeType.CONFLICTS_WITH)) == 0
