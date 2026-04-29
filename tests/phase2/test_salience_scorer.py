"""Tests for the Salience Scorer (Phase 2 Milestone 4).

All tests are deterministic: they pass an explicit ``now`` datetime to every
scoring call so results never depend on the wall clock.

Coverage:
- SalienceWeights construction and validation
- Per-dimension sub-score behaviour
- Composite score properties (clamped, weights respected)
- score_and_update() writes back to MU
- detail() returns full SalienceResult breakdown
- utility() penalises long claims
- rank() and rank(by_utility=True)
- candidates_for_compression() with threshold, pinned exclusion, sort order
- Constructor validation (half_life_days, retrieval_normalization)
- Edge cases: zero retrieval, just-created MU, last_accessed override
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from locomo_memory.phase2.salience import SalienceResult, SalienceScorer, SalienceWeights
from locomo_memory.phase2.schemas import MemoryUnit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _mu(
    *,
    claim: str = "Alice works at Acme Corp.",
    importance: float = 0.5,
    confidence: float = 0.9,
    recency_weight: float = 1.0,
    retrieval_count: int = 0,
    user_pinned: bool = False,
    uniqueness: float = 1.0,
    created_at: datetime | None = None,
    last_accessed: datetime | None = None,
) -> MemoryUnit:
    created = created_at or _NOW
    return MemoryUnit(
        conversation_id="conv_1",
        session_id="session_1",
        claim=claim,
        importance=importance,
        confidence=confidence,
        recency_weight=recency_weight,
        retrieval_count=retrieval_count,
        user_pinned=user_pinned,
        uniqueness=uniqueness,
        created_at=created,
        last_accessed=last_accessed,
    )


def _scorer(**kwargs) -> SalienceScorer:
    return SalienceScorer(**kwargs)


# ---------------------------------------------------------------------------
# SalienceWeights validation
# ---------------------------------------------------------------------------


class TestSalienceWeights:
    def test_defaults_are_positive(self) -> None:
        w = SalienceWeights()
        assert w.total > 0

    def test_total_reflects_all_dims(self) -> None:
        w = SalienceWeights(importance=1.0, confidence=0.0, recency=0.0,
                            retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0)
        assert w.total == pytest.approx(1.0)

    def test_custom_weights(self) -> None:
        w = SalienceWeights(importance=0.5, confidence=0.5, recency=0.0,
                            retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0)
        assert w.total == pytest.approx(1.0)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match=">="):
            SalienceWeights(importance=-0.1)

    def test_all_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="least one"):
            SalienceWeights(
                importance=0, confidence=0, recency=0,
                retrieval_frequency=0, user_pinned=0, uniqueness=0,
            )

    def test_single_nonzero_weight_valid(self) -> None:
        w = SalienceWeights(
            importance=1.0, confidence=0.0, recency=0.0,
            retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0,
        )
        assert w.total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SalienceScorer construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default(self) -> None:
        s = SalienceScorer()
        assert s.half_life_days == pytest.approx(30.0)
        assert s.retrieval_normalization == 10
        assert s.weights == SalienceWeights()

    def test_custom_half_life(self) -> None:
        s = SalienceScorer(half_life_days=7.0)
        assert s.half_life_days == pytest.approx(7.0)

    def test_zero_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life_days"):
            SalienceScorer(half_life_days=0.0)

    def test_negative_half_life_raises(self) -> None:
        with pytest.raises(ValueError):
            SalienceScorer(half_life_days=-1.0)

    def test_zero_retrieval_normalization_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_normalization"):
            SalienceScorer(retrieval_normalization=0)

    def test_custom_weights_stored(self) -> None:
        w = SalienceWeights(importance=1.0, confidence=0.0, recency=0.0,
                            retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0)
        s = SalienceScorer(weights=w)
        assert s.weights is w


# ---------------------------------------------------------------------------
# score() — basic properties
# ---------------------------------------------------------------------------


class TestScore:
    def test_returns_float_in_unit_interval(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        v = s.score(mu, now=_NOW)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_pinned_mu_scores_higher_than_unpinned(self) -> None:
        s = SalienceScorer()
        base = _mu(importance=0.3, user_pinned=False)
        pinned = _mu(importance=0.3, user_pinned=True)
        assert s.score(pinned, now=_NOW) > s.score(base, now=_NOW)

    def test_high_importance_boosts_score(self) -> None:
        s = SalienceScorer()
        low = _mu(importance=0.1)
        high = _mu(importance=0.9)
        assert s.score(high, now=_NOW) > s.score(low, now=_NOW)

    def test_high_confidence_boosts_score(self) -> None:
        s = SalienceScorer()
        low = _mu(confidence=0.1)
        high = _mu(confidence=0.9)
        assert s.score(high, now=_NOW) > s.score(low, now=_NOW)

    def test_high_uniqueness_boosts_score(self) -> None:
        s = SalienceScorer()
        low = _mu(uniqueness=0.1)
        high = _mu(uniqueness=0.9)
        assert s.score(high, now=_NOW) > s.score(low, now=_NOW)

    def test_high_retrieval_count_boosts_score(self) -> None:
        s = SalienceScorer()
        none = _mu(retrieval_count=0)
        many = _mu(retrieval_count=50)
        assert s.score(many, now=_NOW) > s.score(none, now=_NOW)

    def test_fresh_mu_scores_higher_than_old(self) -> None:
        s = SalienceScorer()
        fresh = _mu(created_at=_NOW)
        old = _mu(created_at=_NOW - timedelta(days=90))
        assert s.score(fresh, now=_NOW) > s.score(old, now=_NOW)

    def test_last_accessed_overrides_created_at_for_recency(self) -> None:
        s = SalienceScorer()
        # Old creation date but accessed recently
        recently_accessed = _mu(
            created_at=_NOW - timedelta(days=90),
            last_accessed=_NOW,
        )
        not_accessed = _mu(created_at=_NOW - timedelta(days=90))
        assert s.score(recently_accessed, now=_NOW) > s.score(not_accessed, now=_NOW)

    def test_score_without_now_does_not_raise(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        v = s.score(mu)
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------


class TestRecency:
    def test_half_life_at_half_life_days(self) -> None:
        half_life = 14.0
        s = SalienceScorer(
            weights=SalienceWeights(
                importance=0, confidence=0, recency=1.0,
                retrieval_frequency=0, user_pinned=0, uniqueness=0,
            ),
            half_life_days=half_life,
        )
        mu = _mu(created_at=_NOW - timedelta(days=half_life))
        recency_sub = s.detail(mu, now=_NOW).sub_scores["recency"]
        assert recency_sub == pytest.approx(0.5, abs=1e-6)

    def test_age_zero_recency_is_one(self) -> None:
        s = SalienceScorer()
        mu = _mu(created_at=_NOW)
        sub = s.detail(mu, now=_NOW).sub_scores["recency"]
        assert sub == pytest.approx(1.0, abs=1e-6)

    def test_older_mu_has_lower_recency(self) -> None:
        s = SalienceScorer()
        newer = _mu(created_at=_NOW - timedelta(days=5))
        older = _mu(created_at=_NOW - timedelta(days=60))
        r_new = s.detail(newer, now=_NOW).sub_scores["recency"]
        r_old = s.detail(older, now=_NOW).sub_scores["recency"]
        assert r_new > r_old


# ---------------------------------------------------------------------------
# Retrieval frequency
# ---------------------------------------------------------------------------


class TestRetrievalFrequency:
    def test_zero_retrieval_gives_zero(self) -> None:
        s = SalienceScorer()
        mu = _mu(retrieval_count=0)
        sub = s.detail(mu, now=_NOW).sub_scores["retrieval_frequency"]
        assert sub == pytest.approx(0.0)

    def test_at_norm_constant_gives_half(self) -> None:
        norm = 20
        s = SalienceScorer(retrieval_normalization=norm)
        mu = _mu(retrieval_count=norm)
        sub = s.detail(mu, now=_NOW).sub_scores["retrieval_frequency"]
        assert sub == pytest.approx(0.5, abs=1e-6)

    def test_very_high_retrieval_approaches_one(self) -> None:
        s = SalienceScorer(retrieval_normalization=10)
        mu = _mu(retrieval_count=10_000)
        sub = s.detail(mu, now=_NOW).sub_scores["retrieval_frequency"]
        assert sub > 0.99


# ---------------------------------------------------------------------------
# score_and_update()
# ---------------------------------------------------------------------------


class TestScoreAndUpdate:
    def test_writes_salience_score_to_mu(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        original = mu.salience_score
        v = s.score_and_update(mu, now=_NOW)
        assert mu.salience_score == pytest.approx(v)
        # The update should change the stored field.
        assert mu.salience_score != original or v == original  # may coincidentally equal

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
        mu = _mu()
        r = s.detail(mu, now=_NOW)
        assert isinstance(r, SalienceResult)

    def test_sub_scores_keys_present(self) -> None:
        s = SalienceScorer()
        r = s.detail(_mu(), now=_NOW)
        expected = {"importance", "confidence", "recency",
                    "retrieval_frequency", "user_pinned", "uniqueness"}
        assert set(r.sub_scores.keys()) == expected

    def test_sub_scores_in_unit_interval(self) -> None:
        s = SalienceScorer()
        r = s.detail(_mu(), now=_NOW)
        for k, v in r.sub_scores.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"

    def test_mu_id_propagated(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        r = s.detail(mu, now=_NOW)
        assert r.mu_id == mu.mu_id

    def test_salience_consistent_with_score(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        assert s.detail(mu, now=_NOW).salience == pytest.approx(s.score(mu, now=_NOW))


# ---------------------------------------------------------------------------
# utility()
# ---------------------------------------------------------------------------


class TestUtility:
    def test_short_claim_higher_utility_than_long(self) -> None:
        s = SalienceScorer()
        short = _mu(claim="Alice is a doctor.", importance=0.8)
        long = _mu(
            claim="Alice is a doctor " + "who has worked in various hospitals " * 10,
            importance=0.8,
        )
        assert s.utility(short, now=_NOW) > s.utility(long, now=_NOW)

    def test_utility_at_baseline_equals_salience(self) -> None:
        # A 100-char claim has storage_cost_factor = 1.0, so utility == salience.
        claim = "x" * 100
        s = SalienceScorer()
        mu = _mu(claim=claim)
        assert s.utility(mu, now=_NOW) == pytest.approx(s.score(mu, now=_NOW), abs=1e-6)

    def test_utility_non_negative(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        assert s.utility(mu, now=_NOW) >= 0.0

    def test_storage_cost_factor_in_result(self) -> None:
        s = SalienceScorer()
        # 200-char claim → factor = 2.0
        mu = _mu(claim="y" * 200)
        r = s.detail(mu, now=_NOW)
        assert r.storage_cost_factor == pytest.approx(2.0, abs=1e-6)

    def test_storage_cost_never_below_one(self) -> None:
        s = SalienceScorer()
        mu = _mu(claim="Hi.")
        r = s.detail(mu, now=_NOW)
        assert r.storage_cost_factor >= 1.0


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

    def test_rank_by_utility_descending(self) -> None:
        s = SalienceScorer()
        # Same importance; different claim lengths → different utilities.
        short = _mu(claim="a" * 50, importance=0.8)
        long = _mu(claim="a" * 400, importance=0.8)
        ranked = s.rank([long, short], now=_NOW, by_utility=True)
        assert ranked[0] is short
        assert ranked[1] is long

    def test_empty_list_returns_empty(self) -> None:
        s = SalienceScorer()
        assert s.rank([], now=_NOW) == []

    def test_single_element_unchanged(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        assert s.rank([mu], now=_NOW) == [mu]


# ---------------------------------------------------------------------------
# candidates_for_compression()
# ---------------------------------------------------------------------------


class TestCandidatesForCompression:
    def test_below_threshold_included(self) -> None:
        s = SalienceScorer(
            weights=SalienceWeights(
                importance=1.0, confidence=0.0, recency=0.0,
                retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0,
            )
        )
        low = _mu(importance=0.1)
        high = _mu(importance=0.9)
        candidates = s.candidates_for_compression([low, high], threshold=0.5, now=_NOW)
        assert low in candidates
        assert high not in candidates

    def test_pinned_mu_excluded_regardless_of_score(self) -> None:
        s = SalienceScorer(
            weights=SalienceWeights(
                importance=1.0, confidence=0.0, recency=0.0,
                retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0,
            )
        )
        pinned_low = _mu(importance=0.05, user_pinned=True)
        unpinned_low = _mu(importance=0.05, user_pinned=False)
        candidates = s.candidates_for_compression(
            [pinned_low, unpinned_low], threshold=0.5, now=_NOW
        )
        assert pinned_low not in candidates
        assert unpinned_low in candidates

    def test_sorted_ascending_lowest_first(self) -> None:
        s = SalienceScorer(
            weights=SalienceWeights(
                importance=1.0, confidence=0.0, recency=0.0,
                retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0,
            )
        )
        very_low = _mu(importance=0.05)
        low = _mu(importance=0.2)
        mid = _mu(importance=0.38)
        candidates = s.candidates_for_compression(
            [mid, very_low, low], threshold=0.5, now=_NOW
        )
        scores = [s.score(m, now=_NOW) for m in candidates]
        assert scores == sorted(scores)

    def test_empty_input(self) -> None:
        s = SalienceScorer()
        assert s.candidates_for_compression([], threshold=0.5, now=_NOW) == []

    def test_threshold_zero_returns_nothing(self) -> None:
        s = SalienceScorer()
        mu = _mu()
        # No MU can have salience < 0.
        result = s.candidates_for_compression([mu], threshold=0.0, now=_NOW)
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

    def test_by_utility_mode(self) -> None:
        s = SalienceScorer()
        # Very long claim → high storage cost → low utility despite medium salience
        long = _mu(claim="z" * 500, importance=0.5)
        # Short claim → utility close to salience
        short = _mu(claim="z" * 50, importance=0.5)
        # With threshold tuned to utility, long claim should be a candidate.
        long_utility = s.utility(long, now=_NOW)
        short_utility = s.utility(short, now=_NOW)
        threshold = (long_utility + short_utility) / 2
        candidates = s.candidates_for_compression(
            [long, short], threshold=threshold, now=_NOW, by_utility=True
        )
        assert long in candidates
        assert short not in candidates


# ---------------------------------------------------------------------------
# Weight-only mode (single dimension)
# ---------------------------------------------------------------------------


class TestSingleDimensionWeights:
    def test_importance_only_score_equals_importance(self) -> None:
        s = SalienceScorer(
            weights=SalienceWeights(
                importance=1.0, confidence=0.0, recency=0.0,
                retrieval_frequency=0.0, user_pinned=0.0, uniqueness=0.0,
            )
        )
        mu = _mu(importance=0.73)
        assert s.score(mu, now=_NOW) == pytest.approx(0.73, abs=1e-6)

    def test_pinned_only_weight(self) -> None:
        s = SalienceScorer(
            weights=SalienceWeights(
                importance=0.0, confidence=0.0, recency=0.0,
                retrieval_frequency=0.0, user_pinned=1.0, uniqueness=0.0,
            )
        )
        pinned = _mu(user_pinned=True)
        unpinned = _mu(user_pinned=False)
        assert s.score(pinned, now=_NOW) == pytest.approx(1.0, abs=1e-6)
        assert s.score(unpinned, now=_NOW) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Timezone-naive datetime guard
# ---------------------------------------------------------------------------


class TestTimezoneHandling:
    def test_naive_created_at_handled_gracefully(self) -> None:
        # MemoryUnit validator produces tz-aware datetimes, but if a naive dt
        # slips through (e.g. via direct field assignment on a test object),
        # the scorer should not crash.
        s = SalienceScorer()
        mu = _mu()
        # Force-set a tz-naive datetime by bypassing Pydantic validation.
        object.__setattr__(mu, "created_at", datetime(2024, 1, 1))
        v = s.score(mu, now=_NOW)
        assert 0.0 <= v <= 1.0
