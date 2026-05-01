"""Tests for the Salience Scorer — Ebbinghaus + Topic Importance.

Research basis:
    Ebbinghaus forgetting curve (MemoryBank, Zhong et al. 2023)
    Generative Agents structure (Park et al. 2023)

Formula under test:
    S          = base_stability × 2^min(retrieval_count, 10)
    ebbinghaus = e^(-t / S)
    salience   = 0.60 × ebbinghaus + 0.40 × importance − graph_penalty

All tests are deterministic: they pass an explicit ``now`` datetime to every
scoring call so results never depend on the wall clock.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from locomo_memory.phase2.salience import SalienceResult, SalienceScorer
from locomo_memory.phase2.schemas import MemoryUnit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _mu(
    *,
    claim: str = "Alice works at Acme Corp.",
    importance: float = 0.5,
    retrieval_count: int = 0,
    user_pinned: bool = False,
    created_at: datetime | None = None,
    last_accessed: datetime | None = None,
) -> MemoryUnit:
    created = created_at or _NOW
    return MemoryUnit(
        conversation_id="conv_1",
        session_id="session_1",
        claim=claim,
        importance=importance,
        retrieval_count=retrieval_count,
        user_pinned=user_pinned,
        created_at=created,
        last_accessed=last_accessed,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_base_stability(self) -> None:
        s = SalienceScorer()
        assert s.base_stability == pytest.approx(2.0)

    def test_custom_base_stability(self) -> None:
        s = SalienceScorer(base_stability=5.0)
        assert s.base_stability == pytest.approx(5.0)

    def test_zero_base_stability_raises(self) -> None:
        with pytest.raises(ValueError, match="base_stability"):
            SalienceScorer(base_stability=0.0)

    def test_negative_base_stability_raises(self) -> None:
        with pytest.raises(ValueError):
            SalienceScorer(base_stability=-1.0)


# ---------------------------------------------------------------------------
# score() — basic properties
# ---------------------------------------------------------------------------


class TestScore:
    def test_returns_float_in_unit_interval(self) -> None:
        s = SalienceScorer()
        assert 0.0 <= s.score(_mu(), now=_NOW) <= 1.0

    def test_higher_importance_gives_higher_score(self) -> None:
        s = SalienceScorer()
        low = _mu(importance=0.1)
        high = _mu(importance=0.9)
        assert s.score(high, now=_NOW) > s.score(low, now=_NOW)

    def test_more_retrievals_gives_higher_score(self) -> None:
        # More retrievals → higher S → slower decay → higher ebbinghaus
        s = SalienceScorer()
        never = _mu(retrieval_count=0, created_at=_NOW - timedelta(days=3))
        often = _mu(retrieval_count=5, created_at=_NOW - timedelta(days=3))
        assert s.score(often, now=_NOW) > s.score(never, now=_NOW)

    def test_recently_accessed_higher_than_stale(self) -> None:
        s = SalienceScorer()
        fresh = _mu(created_at=_NOW)
        stale = _mu(created_at=_NOW - timedelta(days=30))
        assert s.score(fresh, now=_NOW) > s.score(stale, now=_NOW)

    def test_last_accessed_overrides_created_at(self) -> None:
        s = SalienceScorer()
        # Old creation but accessed right now
        recently = _mu(created_at=_NOW - timedelta(days=60), last_accessed=_NOW)
        # Same old creation, never accessed
        stale = _mu(created_at=_NOW - timedelta(days=60))
        assert s.score(recently, now=_NOW) > s.score(stale, now=_NOW)

    def test_score_without_now_does_not_raise(self) -> None:
        s = SalienceScorer()
        v = s.score(_mu())
        assert 0.0 <= v <= 1.0

    def test_graph_penalty_reduces_score(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        no_penalty = s.score(mu, now=_NOW, graph_penalty=0.0)
        with_penalty = s.score(mu, now=_NOW, graph_penalty=0.30)
        assert with_penalty < no_penalty

    def test_graph_penalty_capped_at_040(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        # penalty=1.0 should be capped at 0.40 internally
        capped = s.score(mu, now=_NOW, graph_penalty=1.0)
        explicit_040 = s.score(mu, now=_NOW, graph_penalty=0.40)
        assert capped == pytest.approx(explicit_040)

    def test_score_never_negative(self) -> None:
        s = SalienceScorer()
        mu = _mu(importance=0.0, created_at=_NOW - timedelta(days=365))
        assert s.score(mu, now=_NOW, graph_penalty=0.40) >= 0.0


# ---------------------------------------------------------------------------
# Ebbinghaus forgetting curve behaviour
# ---------------------------------------------------------------------------


class TestEbbinghaus:
    def test_just_created_ebbinghaus_is_one(self) -> None:
        s = SalienceScorer()
        mu = _mu(created_at=_NOW, retrieval_count=0)
        r = s.detail(mu, now=_NOW)
        assert r.sub_scores["ebbinghaus"] == pytest.approx(1.0, abs=1e-4)

    def test_half_life_for_never_retrieved(self) -> None:
        # S = base_stability × 2^0 = 2.0
        # half-life t½ = S × ln(2) ≈ 1.386 days
        s = SalienceScorer(base_stability=2.0)
        half_life = 2.0 * math.log(2)
        mu = _mu(created_at=_NOW - timedelta(days=half_life), retrieval_count=0)
        r = s.detail(mu, now=_NOW)
        assert r.sub_scores["ebbinghaus"] == pytest.approx(0.5, abs=1e-3)

    def test_more_retrievals_slower_decay(self) -> None:
        # Both MUs created same time ago; the one with more retrievals
        # should have a higher ebbinghaus sub-score.
        s = SalienceScorer()
        age = timedelta(days=5)
        few = _mu(created_at=_NOW - age, retrieval_count=1)
        many = _mu(created_at=_NOW - age, retrieval_count=5)
        r_few = s.detail(few, now=_NOW).sub_scores["ebbinghaus"]
        r_many = s.detail(many, now=_NOW).sub_scores["ebbinghaus"]
        assert r_many > r_few

    def test_retrieval_count_capped_at_10(self) -> None:
        # retrieval_count=10 and retrieval_count=20 should score identically
        s = SalienceScorer()
        age = timedelta(days=10)
        at_cap = _mu(created_at=_NOW - age, retrieval_count=10)
        over_cap = _mu(created_at=_NOW - age, retrieval_count=20)
        assert s.score(at_cap, now=_NOW) == pytest.approx(s.score(over_cap, now=_NOW))

    def test_stability_doubles_per_retrieval(self) -> None:
        # S = base × 2^N, so ebbinghaus(N=2) corresponds to S=4×base
        # vs ebbinghaus(N=1) which is S=2×base.
        # At same age t, ratio = e^(-t/(4b)) / e^(-t/(2b)) = e^(t/(4b)) > 1
        s = SalienceScorer(base_stability=1.0)
        age_days = 2.0
        one = _mu(created_at=_NOW - timedelta(days=age_days), retrieval_count=1)
        two = _mu(created_at=_NOW - timedelta(days=age_days), retrieval_count=2)
        assert s.detail(two, now=_NOW).sub_scores["ebbinghaus"] > \
               s.detail(one, now=_NOW).sub_scores["ebbinghaus"]


# ---------------------------------------------------------------------------
# Graph penalty
# ---------------------------------------------------------------------------


class TestGraphPenalty:
    def test_zero_penalty_by_default(self) -> None:
        s = SalienceScorer()
        r = s.detail(_mu(), now=_NOW)
        assert r.graph_penalty == pytest.approx(0.0)

    def test_superseded_penalty_applied(self) -> None:
        s = SalienceScorer()
        base = s.score(_mu(), now=_NOW, graph_penalty=0.0)
        penalised = s.score(_mu(), now=_NOW, graph_penalty=0.30)
        assert base - penalised == pytest.approx(0.30, abs=1e-4)

    def test_conflict_penalty_smaller_than_superseded(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        superseded = s.score(mu, now=_NOW, graph_penalty=0.30)
        conflicted = s.score(mu, now=_NOW, graph_penalty=0.10)
        assert conflicted > superseded

    def test_penalty_recorded_in_result(self) -> None:
        s = SalienceScorer()
        r = s.detail(_mu(), now=_NOW, graph_penalty=0.30)
        assert r.graph_penalty == pytest.approx(0.30)

    def test_combined_penalty_capped(self) -> None:
        s = SalienceScorer()
        r = s.detail(_mu(), now=_NOW, graph_penalty=0.50)
        assert r.graph_penalty == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# score_and_update()
# ---------------------------------------------------------------------------


class TestScoreAndUpdate:
    def test_writes_salience_score_to_mu(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        v = s.score_and_update(mu, now=_NOW)
        assert mu.salience_score == pytest.approx(v)

    def test_returned_value_matches_mu_field(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        v = s.score_and_update(mu, now=_NOW)
        assert mu.salience_score == pytest.approx(v)

    def test_idempotent_on_second_call(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        v1 = s.score_and_update(mu, now=_NOW)
        v2 = s.score_and_update(mu, now=_NOW)
        assert v1 == pytest.approx(v2)


# ---------------------------------------------------------------------------
# detail()
# ---------------------------------------------------------------------------


class TestDetail:
    def test_returns_salience_result(self) -> None:
        s = SalienceScorer()
        assert isinstance(s.detail(_mu(), now=_NOW), SalienceResult)

    def test_sub_scores_keys(self) -> None:
        s = SalienceScorer()
        r = s.detail(_mu(), now=_NOW)
        assert set(r.sub_scores.keys()) == {"ebbinghaus", "importance"}

    def test_sub_scores_in_unit_interval(self) -> None:
        s = SalienceScorer()
        r = s.detail(_mu(), now=_NOW)
        for k, v in r.sub_scores.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"

    def test_mu_id_propagated(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        assert s.detail(mu, now=_NOW).mu_id == mu.mu_id

    def test_salience_consistent_with_score(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        assert s.detail(mu, now=_NOW).salience == pytest.approx(s.score(mu, now=_NOW))


# ---------------------------------------------------------------------------
# rank()
# ---------------------------------------------------------------------------


class TestRank:
    def test_rank_by_salience_descending(self) -> None:
        s = SalienceScorer()
        low = _mu(importance=0.1)
        mid = _mu(importance=0.5)
        high = _mu(importance=0.9)
        ranked = s.rank([low, high, mid], now=_NOW)
        scores = [s.score(m, now=_NOW) for m in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list_returns_empty(self) -> None:
        assert SalienceScorer().rank([], now=_NOW) == []

    def test_single_element_unchanged(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        assert s.rank([mu], now=_NOW) == [mu]

    def test_penalties_applied_in_rank(self) -> None:
        s = SalienceScorer()
        mu_a = _mu(importance=0.9)
        mu_b = _mu(importance=0.5)
        # Without penalty mu_a ranks first
        assert s.rank([mu_a, mu_b], now=_NOW)[0] is mu_a
        # With heavy penalty on mu_a it should drop below mu_b
        ranked = s.rank(
            [mu_a, mu_b],
            now=_NOW,
            penalties={mu_a.mu_id: 0.40},
        )
        assert ranked[0] is mu_b


# ---------------------------------------------------------------------------
# candidates_for_compression()
# ---------------------------------------------------------------------------


class TestCandidatesForCompression:
    def test_below_threshold_included(self) -> None:
        s = SalienceScorer()
        # Force low score: very old, never retrieved, low importance
        low = _mu(importance=0.1, created_at=_NOW - timedelta(days=60))
        high = _mu(importance=0.9, created_at=_NOW)
        low_score = s.score(low, now=_NOW)
        high_score = s.score(high, now=_NOW)
        threshold = (low_score + high_score) / 2
        candidates = s.candidates_for_compression([low, high], threshold=threshold, now=_NOW)
        assert low in candidates
        assert high not in candidates

    def test_pinned_mu_excluded_regardless_of_score(self) -> None:
        s = SalienceScorer()
        old = _NOW - timedelta(days=60)
        pinned = _mu(importance=0.05, user_pinned=True, created_at=old)
        unpinned = _mu(importance=0.05, user_pinned=False, created_at=old)
        candidates = s.candidates_for_compression([pinned, unpinned], threshold=1.0, now=_NOW)
        assert pinned not in candidates
        assert unpinned in candidates

    def test_sorted_ascending_lowest_first(self) -> None:
        s = SalienceScorer()
        old = _NOW - timedelta(days=90)
        very_low = _mu(importance=0.05, created_at=old)
        low = _mu(importance=0.20, created_at=old)
        mid = _mu(importance=0.38, created_at=old)
        candidates = s.candidates_for_compression([mid, very_low, low], threshold=1.0, now=_NOW)
        scores = [s.score(m, now=_NOW) for m in candidates]
        assert scores == sorted(scores)

    def test_empty_input(self) -> None:
        assert SalienceScorer().candidates_for_compression([], threshold=0.5, now=_NOW) == []

    def test_threshold_zero_returns_nothing(self) -> None:
        result = SalienceScorer().candidates_for_compression([_mu()], threshold=0.0, now=_NOW)
        assert result == []

    def test_threshold_one_returns_all_unpinned(self) -> None:
        s = SalienceScorer()
        mu1 = _mu(importance=0.9)
        mu2 = _mu(importance=0.1)
        pinned = _mu(user_pinned=True)
        result = s.candidates_for_compression([mu1, mu2, pinned], threshold=1.0, now=_NOW)
        assert pinned not in result
        assert mu1 in result
        assert mu2 in result

    def test_invalid_threshold_raises(self) -> None:
        s = SalienceScorer()
        with pytest.raises(ValueError, match="threshold"):
            s.candidates_for_compression([_mu()], threshold=1.5, now=_NOW)
        with pytest.raises(ValueError, match="threshold"):
            s.candidates_for_compression([_mu()], threshold=-0.1, now=_NOW)

    def test_penalties_lower_candidate_threshold(self) -> None:
        s = SalienceScorer()
        # mu_a would normally score above 0.5 threshold
        mu_a = _mu(importance=0.8, created_at=_NOW)
        score_no_penalty = s.score(mu_a, now=_NOW)
        assert score_no_penalty > 0.5
        # With superseded penalty it should fall below
        result = s.candidates_for_compression(
            [mu_a], threshold=0.9, now=_NOW,
            penalties={mu_a.mu_id: 0.40},
        )
        assert mu_a in result


# ---------------------------------------------------------------------------
# Timezone-naive datetime guard
# ---------------------------------------------------------------------------


class TestTimezoneHandling:
    def test_naive_created_at_handled_gracefully(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        object.__setattr__(mu, "created_at", datetime(2024, 1, 1))
        v = s.score(mu, now=_NOW)
        assert 0.0 <= v <= 1.0
