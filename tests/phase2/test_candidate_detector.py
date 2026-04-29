"""Tests for the Memory Candidate Detector.

Covers: composite-score range, sub-score correctness for each signal,
threshold semantics, custom-weight validation, edge cases (empty, very long,
questions, opinions), and that ``is_candidate`` matches ``score().is_candidate``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from locomo_memory.phase2.ingestion.candidate_detector import (
    CandidateDetector,
    CandidateScore,
    CandidateWeights,
)


# ---------------------------------------------------------------------------
# Construction & weights
# ---------------------------------------------------------------------------


class TestCandidateWeights:
    def test_default_weights_sum_to_one(self) -> None:
        w = CandidateWeights()
        total = (
            w.has_named_entity + w.verb_density + w.is_factual_statement
            + w.has_concrete_topic_marker + w.length_normalized
            + w.has_specific_number_or_date
        )
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError):
            CandidateWeights(
                has_named_entity=0.5,
                verb_density=0.5,
                is_factual_statement=0.5,
                has_concrete_topic_marker=0.0,
                length_normalized=0.0,
                has_specific_number_or_date=0.0,
            )

    def test_custom_weights_accepted_when_summing_to_one(self) -> None:
        w = CandidateWeights(
            has_named_entity=0.40,
            verb_density=0.20,
            is_factual_statement=0.15,
            has_concrete_topic_marker=0.15,
            length_normalized=0.05,
            has_specific_number_or_date=0.05,
        )
        assert w.has_named_entity == 0.40

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CandidateWeights(has_named_entity=-0.1)


class TestDetectorConstruction:
    def test_default_threshold(self) -> None:
        d = CandidateDetector()
        assert d.threshold == 0.35

    def test_custom_threshold(self) -> None:
        d = CandidateDetector(threshold=0.5)
        assert d.threshold == 0.5

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 5.0])
    def test_invalid_threshold_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError):
            CandidateDetector(threshold=bad)

    def test_zero_threshold_accepts_everything(self) -> None:
        d = CandidateDetector(threshold=0.0)
        # Even an empty string scores 0 ≥ 0
        assert d.is_candidate("") is True

    def test_one_threshold_rejects_almost_everything(self) -> None:
        d = CandidateDetector(threshold=1.0)
        # A single all-positive-signal text would need every sub-score = 1.0,
        # which is hard to construct; chitchat definitely does not qualify.
        assert d.is_candidate("hello") is False


# ---------------------------------------------------------------------------
# Score range and structure
# ---------------------------------------------------------------------------


class TestScoreShape:
    def test_score_in_unit_interval(self) -> None:
        d = CandidateDetector()
        score = d.score(
            "Caroline left Google in March 2024 and joined Microsoft on Monday."
        )
        assert 0.0 <= score.score <= 1.0
        for field in (
            "has_named_entity", "verb_density", "is_factual_statement",
            "has_concrete_topic_marker", "length_normalized",
            "has_specific_number_or_date",
        ):
            sub = getattr(score, field)
            assert 0.0 <= sub <= 1.0, f"{field} out of [0,1]: {sub}"

    def test_is_candidate_matches_threshold(self) -> None:
        d = CandidateDetector(threshold=0.4)
        s = d.score("Caroline got promoted to senior engineer in March 2024.")
        assert s.is_candidate == (s.score >= 0.4)

    def test_helper_matches_score(self) -> None:
        d = CandidateDetector()
        text = "I went to the doctor yesterday for a checkup."
        assert d.is_candidate(text) == d.score(text).is_candidate

    def test_text_preview_truncated(self) -> None:
        d = CandidateDetector()
        long = "A" * 500
        s = d.score(long)
        assert len(s.text_preview) <= 120

    def test_returns_pydantic_model(self) -> None:
        d = CandidateDetector()
        s = d.score("test")
        assert isinstance(s, CandidateScore)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_text(self) -> None:
        d = CandidateDetector()
        s = d.score(None)
        assert s.score == 0.0
        assert s.is_candidate is False

    def test_empty_string(self) -> None:
        d = CandidateDetector()
        s = d.score("")
        assert s.score == 0.0
        assert s.is_candidate is False

    def test_whitespace_only(self) -> None:
        d = CandidateDetector()
        s = d.score("   \n\t  ")
        assert s.score == 0.0
        assert s.is_candidate is False

    def test_single_word(self) -> None:
        d = CandidateDetector()
        s = d.score("hello")
        assert s.score < 0.35  # below default threshold

    def test_pure_greeting_below_threshold(self) -> None:
        d = CandidateDetector()
        for greeting in ("Hi!", "Hey there", "haha ok cool", "yeah sure"):
            s = d.score(greeting)
            assert s.is_candidate is False, f"greeting passed: {greeting!r}"


# ---------------------------------------------------------------------------
# Signal-by-signal sub-score behaviour
# ---------------------------------------------------------------------------


class TestNamedEntitySubscore:
    def test_proper_nouns_score_high(self) -> None:
        d = CandidateDetector()
        s = d.score("Caroline went to Google with Jake last week.")
        assert s.has_named_entity == 1.0

    def test_no_proper_nouns_score_zero(self) -> None:
        d = CandidateDetector()
        s = d.score("i went to the store and bought some bread")
        assert s.has_named_entity == 0.0

    def test_sentence_initial_pronouns_not_counted(self) -> None:
        d = CandidateDetector()
        # "The store ..." starts with "The" which is a common starter and
        # should not count as a proper noun by itself.
        s = d.score("The store was busy. I bought milk.")
        assert s.has_named_entity == 0.0


class TestVerbDensitySubscore:
    def test_action_verbs_score_high(self) -> None:
        d = CandidateDetector()
        s = d.score("She went got bought sold ate met called")
        # 7 verbs / 7 words = 1.0 ratio → clamped at 1.0
        assert s.verb_density == 1.0

    def test_no_verbs(self) -> None:
        d = CandidateDetector()
        s = d.score("apple banana cherry")
        assert s.verb_density == 0.0

    def test_empty_words(self) -> None:
        d = CandidateDetector()
        s = d.score("!!!")
        assert s.verb_density == 0.0


class TestFactualStatementSubscore:
    def test_question_scores_zero(self) -> None:
        d = CandidateDetector()
        s = d.score("What did you do yesterday?")
        assert s.is_factual_statement == 0.0

    def test_question_starter_without_qmark_still_zero(self) -> None:
        d = CandidateDetector()
        s = d.score("Do you remember last summer")
        assert s.is_factual_statement == 0.0

    def test_opinion_marker_reduces_score(self) -> None:
        d = CandidateDetector()
        s_factual = d.score("Caroline joined Microsoft in March 2024.")
        s_opinion = d.score("I think Caroline maybe joined Microsoft.")
        assert s_factual.is_factual_statement > s_opinion.is_factual_statement
        assert s_opinion.is_factual_statement < 1.0

    def test_plain_statement_full_score(self) -> None:
        d = CandidateDetector()
        s = d.score("Caroline left her job last month.")
        assert s.is_factual_statement == 1.0


class TestTopicMarkerSubscore:
    def test_two_or_more_markers_full_score(self) -> None:
        d = CandidateDetector()
        s = d.score("She quit her job and moved to a new apartment.")
        # quit, job, moved, apartment → 4 markers → full
        assert s.has_concrete_topic_marker == 1.0

    def test_no_markers_zero(self) -> None:
        d = CandidateDetector()
        s = d.score("apple banana cherry")
        assert s.has_concrete_topic_marker == 0.0

    def test_one_marker_partial(self) -> None:
        d = CandidateDetector()
        s = d.score("xyz xyz xyz job xyz xyz")
        assert 0.0 < s.has_concrete_topic_marker < 1.0


class TestLengthSubscore:
    def test_too_short_zero(self) -> None:
        d = CandidateDetector()
        # 4 words
        s = d.score("one two three four")
        assert s.length_normalized == 0.0

    def test_sweet_spot_full(self) -> None:
        d = CandidateDetector()
        words = " ".join(["apple"] * 50)
        s = d.score(words)
        assert s.length_normalized == 1.0

    def test_too_long_decays(self) -> None:
        d = CandidateDetector()
        words = " ".join(["apple"] * 200)
        s = d.score(words)
        assert 0.0 < s.length_normalized < 1.0

    def test_extremely_long_zero(self) -> None:
        d = CandidateDetector()
        words = " ".join(["apple"] * 500)
        s = d.score(words)
        assert s.length_normalized == 0.0


class TestNumberDateSubscore:
    @pytest.mark.parametrize("text", [
        "She joined in 2024",
        "On the 3rd of June",
        "It cost $42.50",
        "We waited 3 hours",
        "Population is 1,234,567",
        "Meeting at 12:30",
    ])
    def test_positive(self, text: str) -> None:
        d = CandidateDetector()
        s = d.score(text)
        assert s.has_specific_number_or_date == 1.0

    @pytest.mark.parametrize("text", [
        "no numbers here at all",
        "a quick brown fox",
        "",
    ])
    def test_negative(self, text: str) -> None:
        d = CandidateDetector()
        s = d.score(text)
        assert s.has_specific_number_or_date == 0.0


# ---------------------------------------------------------------------------
# Custom weights affect composite
# ---------------------------------------------------------------------------


class TestCustomWeights:
    def test_zero_out_named_entity(self) -> None:
        # If we put 100% weight on length_normalized, only length matters.
        d = CandidateDetector(
            threshold=0.5,
            weights=CandidateWeights(
                has_named_entity=0.0,
                verb_density=0.0,
                is_factual_statement=0.0,
                has_concrete_topic_marker=0.0,
                length_normalized=1.0,
                has_specific_number_or_date=0.0,
            ),
        )
        # 50 words, no entities → length subscore = 1.0 → composite 1.0
        text = " ".join(["xyz"] * 50)
        s = d.score(text)
        assert s.score == 1.0


# ---------------------------------------------------------------------------
# Realistic LoCoMo-shaped examples
# ---------------------------------------------------------------------------


class TestRealisticExamples:
    def test_factual_chunk_passes(self) -> None:
        d = CandidateDetector()
        text = (
            "Caroline quit her job at Google in March 2024. "
            "She started at Microsoft on the 15th. "
            "Her new role pays $15000 more per year."
        )
        s = d.score(text)
        assert s.is_candidate is True
        assert s.score > 0.5

    def test_chitchat_chunk_skipped(self) -> None:
        d = CandidateDetector()
        text = "haha yeah right ok cool sure whatever"
        s = d.score(text)
        assert s.is_candidate is False

    def test_pure_question_chunk_skipped(self) -> None:
        d = CandidateDetector()
        text = "What did you do yesterday? How was the weather?"
        s = d.score(text)
        assert s.is_candidate is False
