"""Tests for Phase 2 Milestone 9: ContextBuilder + ResponseGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from locomo_memory.phase2.context.builder import (
    SECTION_ACTIVE,
    SECTION_CONFLICTED,
    SECTION_RESTORED,
    SECTION_SUPERSEDED,
    SYSTEM_PROMPT,
    BuiltContext,
    ContextBuilder,
    ContextEntry,
)
from locomo_memory.phase2.context.guard import GuardVerdict, ResponseGuard
from locomo_memory.phase2.retrieval.hybrid_retriever import (
    HybridHit,
    RelationMeta,
)
from locomo_memory.phase2.schemas import (
    EdgeRecord,
    EdgeType,
    MemoryStatus,
    MemoryUnit,
)
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mu(
    claim: str,
    *,
    mu_id: str | None = None,
    conv: str = "conv1",
    session: str = "s1",
    speaker: str = "Alice",
    timestamp: str | None = "2024-01-01",
    confidence: float = 0.9,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryUnit:
    kw = dict(
        conversation_id=conv,
        session_id=session,
        claim=claim,
        source_speaker=speaker,
        timestamp=timestamp,
        confidence=confidence,
        status=status,
    )
    if mu_id:
        kw["mu_id"] = mu_id
    return MemoryUnit(**kw)


def _hit(
    mu: MemoryUnit,
    *,
    rrf_score: float = 0.5,
    sources: list[str] | None = None,
    label_summary: str | None = None,
    is_from_label: bool = False,
    superseded_by: list[str] | None = None,
    conflicts_with: list[str] | None = None,
    related_to: list[str] | None = None,
) -> HybridHit:
    return HybridHit(
        mu=mu,
        rrf_score=rrf_score,
        rank=0,
        sources=sources or ["faiss"],
        label_summary=label_summary,
        relation_meta=RelationMeta(
            superseded_by=superseded_by or [],
            conflicts_with=conflicts_with or [],
            related_to=related_to or [],
        ),
        is_from_label=is_from_label,
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


@pytest.fixture()
def builder(store: MemoryStore) -> ContextBuilder:
    return ContextBuilder(store)


def _built_context(
    *claims: str,
    store: MemoryStore,
    query: str = "test query",
) -> BuiltContext:
    """Build context from plain active claim strings."""
    builder = ContextBuilder(store)
    hits = [_hit(_mu(c)) for c in claims]
    return builder.build(query, hits)


# ===========================================================================
# ContextBuilder tests
# ===========================================================================


class TestContextBuilderConstruction:
    def test_default_max_entries(self, builder):
        assert builder.max_entries == 10

    def test_custom_max_entries(self, store):
        b = ContextBuilder(store, max_entries=3)
        assert b.max_entries == 3

    def test_custom_system_prompt(self, store):
        b = ContextBuilder(store, system_prompt="custom prompt")
        assert b.system_prompt == "custom prompt"


class TestContextBuilderBuildEmpty:
    def test_empty_hits_no_entries(self, builder):
        ctx = builder.build("query", [])
        assert ctx.entries == []
        assert ctx.total_entries == 0

    def test_empty_hits_flags_false(self, builder):
        ctx = builder.build("query", [])
        assert ctx.has_active is False
        assert ctx.has_superseded is False
        assert ctx.has_conflicted is False
        assert ctx.has_restored is False

    def test_empty_hits_rendered_text(self, builder):
        ctx = builder.build("query", [])
        assert "No memory evidence" in ctx.rendered_text

    def test_empty_hits_empty_evidence_tokens(self, builder):
        ctx = builder.build("query", [])
        assert len(ctx.evidence_tokens) == 0

    def test_query_preserved(self, builder):
        ctx = builder.build("what is Alice's job", [])
        assert ctx.query == "what is Alice's job"

    def test_system_prompt_present(self, builder):
        ctx = builder.build("query", [])
        assert ctx.system_prompt == SYSTEM_PROMPT
        assert "No information available" in ctx.system_prompt


class TestContextBuilderSectionAssignment:
    def test_plain_active_hit(self, store, builder):
        hit = _hit(_mu("Alice works at Google", mu_id="m1"))
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].section == SECTION_ACTIVE
        assert ctx.has_active is True

    def test_restored_takes_priority_over_all(self, store, builder):
        hit = _hit(
            _mu("Alice works at Google", mu_id="m1"),
            is_from_label=True,
            superseded_by=["m2"],
            conflicts_with=["m3"],
        )
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].section == SECTION_RESTORED

    def test_superseded_takes_priority_over_conflicted(self, store, builder):
        hit = _hit(
            _mu("Alice works at Google", mu_id="m1"),
            superseded_by=["m2"],
            conflicts_with=["m3"],
        )
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].section == SECTION_SUPERSEDED

    def test_conflicted_when_only_conflicts(self, store, builder):
        hit = _hit(
            _mu("Alice likes cats", mu_id="m1"),
            conflicts_with=["m2"],
        )
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].section == SECTION_CONFLICTED

    def test_active_when_no_relations(self, store, builder):
        hit = _hit(_mu("Bob lives in NYC", mu_id="m1"))
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].section == SECTION_ACTIVE

    def test_mixed_sections(self, store, builder):
        hits = [
            _hit(_mu("active fact", mu_id="m1")),
            _hit(_mu("old employer", mu_id="m2"), superseded_by=["m1"]),
            _hit(_mu("conflicting claim", mu_id="m3"), conflicts_with=["m4"]),
            _hit(_mu("compressed fact", mu_id="m5"), is_from_label=True, label_summary="short label"),
        ]
        ctx = builder.build("query", hits)
        sections = {e.section for e in ctx.entries}
        assert SECTION_ACTIVE in sections
        assert SECTION_SUPERSEDED in sections
        assert SECTION_CONFLICTED in sections
        assert SECTION_RESTORED in sections

    def test_flags_reflect_sections(self, store, builder):
        hits = [
            _hit(_mu("active", mu_id="m1")),
            _hit(_mu("superseded", mu_id="m2"), superseded_by=["m1"]),
        ]
        ctx = builder.build("query", hits)
        assert ctx.has_active is True
        assert ctx.has_superseded is True
        assert ctx.has_conflicted is False
        assert ctx.has_restored is False


class TestContextBuilderEntryFields:
    def test_entry_index_starts_at_one(self, store, builder):
        hits = [_hit(_mu("claim A")), _hit(_mu("claim B"))]
        ctx = builder.build("query", hits)
        assert ctx.entries[0].index == 1
        assert ctx.entries[1].index == 2

    def test_entry_mu_id_preserved(self, store, builder):
        mu = _mu("Alice works at Google", mu_id="m42")
        ctx = builder.build("query", [_hit(mu)])
        assert ctx.entries[0].mu_id == "m42"

    def test_entry_claim_preserved(self, store, builder):
        mu = _mu("Alice works at Google", mu_id="m1")
        ctx = builder.build("query", [_hit(mu)])
        assert ctx.entries[0].claim == "Alice works at Google"

    def test_entry_confidence_preserved(self, store, builder):
        mu = _mu("claim", confidence=0.77)
        ctx = builder.build("query", [_hit(mu)])
        assert abs(ctx.entries[0].confidence - 0.77) < 1e-6

    def test_entry_speaker_preserved(self, store, builder):
        mu = _mu("claim", speaker="Caroline")
        ctx = builder.build("query", [_hit(mu)])
        assert ctx.entries[0].source_speaker == "Caroline"

    def test_entry_session_preserved(self, store, builder):
        mu = _mu("claim", session="session_5")
        ctx = builder.build("query", [_hit(mu)])
        assert ctx.entries[0].source_session == "session_5"

    def test_entry_timestamp_preserved(self, store, builder):
        mu = _mu("claim", timestamp="2024-03-15")
        ctx = builder.build("query", [_hit(mu)])
        assert ctx.entries[0].source_timestamp == "2024-03-15"

    def test_entry_superseded_by_ids(self, store, builder):
        hit = _hit(_mu("old fact", mu_id="m1"), superseded_by=["m2", "m3"])
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].superseded_by_ids == ["m2", "m3"]

    def test_entry_conflicts_with_ids(self, store, builder):
        hit = _hit(_mu("conflicting", mu_id="m1"), conflicts_with=["m5"])
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].conflicts_with_ids == ["m5"]

    def test_entry_related_to_ids(self, store, builder):
        hit = _hit(_mu("related", mu_id="m1"), related_to=["m9", "m10"])
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].related_to_ids == ["m9", "m10"]

    def test_entry_label_summary_preserved(self, store, builder):
        hit = _hit(
            _mu("compressed claim", mu_id="m1"),
            is_from_label=True,
            label_summary="short summary text",
        )
        ctx = builder.build("query", [hit])
        assert ctx.entries[0].label_summary == "short summary text"

    def test_mu_ids_property(self, store, builder):
        hits = [
            _hit(_mu("a", mu_id="ma")),
            _hit(_mu("b", mu_id="mb")),
        ]
        ctx = builder.build("q", hits)
        assert ctx.mu_ids == ["ma", "mb"]


class TestContextBuilderMaxEntries:
    def test_max_entries_truncates(self, store):
        builder = ContextBuilder(store, max_entries=3)
        hits = [_hit(_mu(f"claim {i}", mu_id=f"m{i}")) for i in range(10)]
        ctx = builder.build("query", hits)
        assert ctx.total_entries == 3

    def test_max_entries_default_ten(self, store):
        builder = ContextBuilder(store)
        hits = [_hit(_mu(f"claim {i}", mu_id=f"m{i}")) for i in range(15)]
        ctx = builder.build("query", hits)
        assert ctx.total_entries == 10

    def test_fewer_than_max_all_included(self, store, builder):
        hits = [_hit(_mu(f"claim {i}")) for i in range(3)]
        ctx = builder.build("query", hits)
        assert ctx.total_entries == 3


class TestContextBuilderRendering:
    def test_rendered_text_contains_claim(self, store, builder):
        mu = _mu("Alice works at Google", mu_id="m1")
        ctx = builder.build("query", [_hit(mu)])
        assert "Alice works at Google" in ctx.rendered_text

    def test_rendered_text_contains_active_header(self, store, builder):
        ctx = builder.build("query", [_hit(_mu("a fact"))])
        assert "ACTIVE MEMORIES" in ctx.rendered_text

    def test_rendered_text_contains_superseded_header(self, store, builder):
        hit = _hit(_mu("old fact", mu_id="m1"), superseded_by=["m2"])
        ctx = builder.build("query", [hit])
        assert "HISTORICAL CONTEXT" in ctx.rendered_text

    def test_rendered_text_contains_conflicted_header(self, store, builder):
        hit = _hit(_mu("conflicting fact", mu_id="m1"), conflicts_with=["m2"])
        ctx = builder.build("query", [hit])
        assert "CONFLICTING" in ctx.rendered_text

    def test_rendered_text_contains_restored_header(self, store, builder):
        hit = _hit(
            _mu("compressed fact", mu_id="m1"),
            is_from_label=True,
            label_summary="short label",
        )
        ctx = builder.build("query", [hit])
        assert "RESTORED FROM COMPRESSED" in ctx.rendered_text

    def test_rendered_text_contains_entry_index(self, store, builder):
        ctx = builder.build("query", [_hit(_mu("a claim"))])
        assert "[1]" in ctx.rendered_text

    def test_rendered_text_contains_confidence(self, store, builder):
        mu = _mu("claim", confidence=0.88)
        ctx = builder.build("query", [_hit(mu)])
        assert "0.88" in ctx.rendered_text

    def test_superseded_entry_shows_superseded_by(self, store):
        # Insert the superseding MU into the store so claims can be resolved
        sup_mu = _mu("Alice works at Microsoft", mu_id="m2")
        store.insert_memory_unit(sup_mu)
        builder = ContextBuilder(store)
        hit = _hit(_mu("Alice works at Google", mu_id="m1"), superseded_by=["m2"])
        ctx = builder.build("query", [hit])
        assert "SUPERSEDED BY" in ctx.rendered_text
        assert "Alice works at Microsoft" in ctx.rendered_text

    def test_superseded_entry_falls_back_to_id(self, store, builder):
        # m99 not in store — should fall back to the raw ID
        hit = _hit(_mu("old fact", mu_id="m1"), superseded_by=["m99"])
        ctx = builder.build("query", [hit])
        assert "m99" in ctx.rendered_text

    def test_conflicted_entry_shows_conflicts_with(self, store):
        conf_mu = _mu("Alice dislikes cats", mu_id="m2")
        store.insert_memory_unit(conf_mu)
        builder = ContextBuilder(store)
        hit = _hit(_mu("Alice likes cats", mu_id="m1"), conflicts_with=["m2"])
        ctx = builder.build("query", [hit])
        assert "CONFLICTS WITH" in ctx.rendered_text
        assert "Alice dislikes cats" in ctx.rendered_text

    def test_restored_entry_shows_label_summary(self, store, builder):
        hit = _hit(
            _mu("hiking fact", mu_id="m1"),
            is_from_label=True,
            label_summary="outdoor activities",
        )
        ctx = builder.build("query", [hit])
        assert "outdoor activities" in ctx.rendered_text

    def test_no_section_header_for_empty_section(self, store, builder):
        ctx = builder.build("query", [_hit(_mu("active fact"))])
        assert "HISTORICAL CONTEXT" not in ctx.rendered_text
        assert "CONFLICTING" not in ctx.rendered_text
        assert "RESTORED FROM COMPRESSED" not in ctx.rendered_text


class TestContextBuilderEvidenceTokens:
    def test_evidence_tokens_from_claims(self, store, builder):
        mu = _mu("Alice works Google headquarters")
        ctx = builder.build("query", [_hit(mu)])
        # "works", "google", "headquarters", "alice" should be in tokens (stopwords removed)
        assert "alice" in ctx.evidence_tokens
        assert "google" in ctx.evidence_tokens

    def test_evidence_tokens_include_label_summary(self, store, builder):
        hit = _hit(
            _mu("compressed claim", mu_id="m1"),
            is_from_label=True,
            label_summary="outdoor activities hiking",
        )
        ctx = builder.build("query", [hit])
        assert "hiking" in ctx.evidence_tokens

    def test_evidence_tokens_empty_when_no_entries(self, store, builder):
        ctx = builder.build("query", [])
        assert len(ctx.evidence_tokens) == 0

    def test_evidence_tokens_is_frozenset(self, store, builder):
        ctx = builder.build("query", [_hit(_mu("a claim"))])
        assert isinstance(ctx.evidence_tokens, frozenset)


class TestContextBuilderBuildPrompt:
    def test_build_prompt_returns_three_values(self, store, builder):
        result = builder.build_prompt("query", [_hit(_mu("a claim"))])
        assert len(result) == 3
        system_prompt, user_message, ctx = result
        assert isinstance(system_prompt, str)
        assert isinstance(user_message, str)
        assert isinstance(ctx, BuiltContext)

    def test_user_message_contains_query(self, store, builder):
        _, user_msg, _ = builder.build_prompt("What is Alice's job?", [_hit(_mu("Alice works"))])
        assert "What is Alice's job?" in user_msg

    def test_user_message_contains_answer_prompt(self, store, builder):
        _, user_msg, _ = builder.build_prompt("query", [_hit(_mu("a claim"))])
        assert "Answer:" in user_msg

    def test_system_prompt_returned(self, store, builder):
        sys, _, _ = builder.build_prompt("query", [])
        assert sys == SYSTEM_PROMPT


class TestContextEntrySourceInfo:
    def test_source_info_includes_confidence(self):
        entry = ContextEntry(
            index=1, mu_id="m1", claim="fact", section=SECTION_ACTIVE,
            confidence=0.92, source_speaker="Alice", source_session="s1",
            source_timestamp="2024-01-01",
        )
        assert "0.92" in entry.source_info

    def test_source_info_includes_speaker(self):
        entry = ContextEntry(
            index=1, mu_id="m1", claim="fact", section=SECTION_ACTIVE,
            confidence=0.9, source_speaker="Caroline", source_session="s1",
            source_timestamp=None,
        )
        assert "Caroline" in entry.source_info

    def test_source_info_skips_none_timestamp(self):
        entry = ContextEntry(
            index=1, mu_id="m1", claim="fact", section=SECTION_ACTIVE,
            confidence=0.9, source_speaker="", source_session="s1",
            source_timestamp=None,
        )
        assert "Date" not in entry.source_info


# ===========================================================================
# ResponseGuard tests
# ===========================================================================


def _empty_context(store: MemoryStore) -> BuiltContext:
    return ContextBuilder(store).build("query", [])


def _context_with(*claims: str, store: MemoryStore) -> BuiltContext:
    builder = ContextBuilder(store)
    return builder.build("query", [_hit(_mu(c)) for c in claims])


def _conflicted_context(store: MemoryStore) -> BuiltContext:
    builder = ContextBuilder(store)
    hits = [
        _hit(_mu("Alice likes cats", mu_id="m1"), conflicts_with=["m2"]),
    ]
    return builder.build("query", hits)


class TestResponseGuardDefaults:
    def test_default_min_grounding_zero(self):
        guard = ResponseGuard()
        assert guard.min_grounding_score == 0.0

    def test_default_require_uncertainty(self):
        guard = ResponseGuard()
        assert guard.require_uncertainty_for_conflicts is True


class TestResponseGuardNoInfoAnswer:
    def test_no_info_with_no_evidence_passes(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("No information available.", ctx)
        assert verdict.passed is True
        assert verdict.is_no_info is True
        assert verdict.grounding_score == 1.0

    def test_no_info_case_insensitive(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("NO INFORMATION AVAILABLE", ctx)
        assert verdict.is_no_info is True

    def test_no_info_with_evidence_passes_but_warns(self, store):
        guard = ResponseGuard()
        ctx = _context_with("Alice works at Google", store=store)
        verdict = guard.check("No information available.", ctx)
        assert verdict.passed is True
        assert verdict.is_no_info is True
        assert verdict.has_warnings is True

    def test_empty_answer_treated_as_no_info(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("", ctx)
        assert verdict.is_no_info is True
        assert verdict.passed is True

    def test_whitespace_only_answer_treated_as_no_info(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("   \n\t  ", ctx)
        assert verdict.is_no_info is True


class TestResponseGuardGrounding:
    def test_grounded_answer_passes(self, store):
        guard = ResponseGuard()
        ctx = _context_with("Alice works at Google headquarters", store=store)
        verdict = guard.check("Alice works at Google.", ctx)
        assert verdict.passed is True
        assert verdict.grounding_score > 0.0

    def test_unrelated_answer_has_low_score(self, store):
        guard = ResponseGuard()
        ctx = _context_with("Alice works at Google", store=store)
        verdict = guard.check("The weather in Paris is sunny today.", ctx)
        # Very few token overlaps expected
        assert verdict.grounding_score < 0.5

    def test_no_evidence_non_refusal_fails(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("Alice works at Google.", ctx)
        assert verdict.passed is False
        assert verdict.has_warnings is True

    def test_grounding_score_range(self, store):
        guard = ResponseGuard()
        ctx = _context_with("Alice works at Google", store=store)
        verdict = guard.check("Alice Google", ctx)
        assert 0.0 <= verdict.grounding_score <= 1.0

    def test_grounding_score_with_threshold_fail(self, store):
        guard = ResponseGuard(min_grounding_score=0.9)
        ctx = _context_with("Alice works at Google", store=store)
        verdict = guard.check("The weather in Paris is sunny.", ctx)
        assert verdict.passed is False
        assert any("threshold" in w.lower() or "minimum" in w.lower() for w in verdict.warnings)

    def test_grounding_score_with_threshold_pass(self, store):
        guard = ResponseGuard(min_grounding_score=0.1)
        ctx = _context_with("Alice works at Google headquarters", store=store)
        verdict = guard.check("Alice works Google", ctx)
        assert verdict.passed is True

    def test_verdict_token_counts(self, store):
        guard = ResponseGuard()
        ctx = _context_with("Alice works at Google", store=store)
        verdict = guard.check("Alice works", ctx)
        assert verdict.answer_token_count >= 1
        assert verdict.evidence_token_count >= 1
        assert verdict.overlap_token_count >= 0
        assert verdict.overlap_token_count <= min(
            verdict.answer_token_count, verdict.evidence_token_count
        )


class TestResponseGuardConflictWarning:
    def test_conflict_without_uncertainty_warns(self, store):
        guard = ResponseGuard(require_uncertainty_for_conflicts=True)
        ctx = _conflicted_context(store)
        verdict = guard.check("Alice likes cats.", ctx)
        # No uncertainty marker → should warn
        assert any("conflict" in w.lower() or "uncertain" in w.lower() for w in verdict.warnings)

    def test_conflict_with_uncertainty_no_warn(self, store):
        guard = ResponseGuard(require_uncertainty_for_conflicts=True)
        ctx = _conflicted_context(store)
        verdict = guard.check("It is unclear whether Alice likes cats.", ctx)
        conflict_warns = [
            w for w in verdict.warnings
            if "conflict" in w.lower() or "uncertain" in w.lower()
        ]
        assert len(conflict_warns) == 0

    def test_conflict_requirement_disabled(self, store):
        guard = ResponseGuard(require_uncertainty_for_conflicts=False)
        ctx = _conflicted_context(store)
        verdict = guard.check("Alice likes cats.", ctx)
        conflict_warns = [
            w for w in verdict.warnings
            if "conflict" in w.lower() or "uncertain" in w.lower()
        ]
        assert len(conflict_warns) == 0

    def test_no_conflict_in_context_no_warn(self, store):
        guard = ResponseGuard(require_uncertainty_for_conflicts=True)
        ctx = _context_with("Alice works at Google", store=store)
        verdict = guard.check("Alice works at Google.", ctx)
        conflict_warns = [
            w for w in verdict.warnings
            if "conflict" in w.lower() or "uncertain" in w.lower()
        ]
        assert len(conflict_warns) == 0


class TestResponseGuardBatch:
    def test_batch_same_length(self, store):
        guard = ResponseGuard()
        ctxs = [_context_with("claim", store=store)] * 3
        answers = ["answer A", "answer B", "answer C"]
        verdicts = guard.check_batch(answers, ctxs)
        assert len(verdicts) == 3
        assert all(isinstance(v, GuardVerdict) for v in verdicts)

    def test_batch_length_mismatch_raises(self, store):
        guard = ResponseGuard()
        ctxs = [_empty_context(store)] * 2
        answers = ["a", "b", "c"]
        with pytest.raises(ValueError, match="same length"):
            guard.check_batch(answers, ctxs)


class TestResponseGuardVerdict:
    def test_has_warnings_true(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("Some answer with no evidence.", ctx)
        assert verdict.has_warnings is True

    def test_has_warnings_false(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("No information available.", ctx)
        assert verdict.has_warnings is False

    def test_passed_attribute(self, store):
        guard = ResponseGuard()
        ctx = _empty_context(store)
        verdict = guard.check("No information available.", ctx)
        assert isinstance(verdict.passed, bool)
