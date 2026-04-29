"""Tests for the rule-based TopicImportanceEstimator.

Verifies that:
- High-importance topic signals (employment, health, relationships, etc.)
  return scores in the high bucket (0.85).
- Medium signals (interests, plans, preferences) return 0.55.
- Hedged opinions and meta-speech return 0.30.
- Claims with no recognised signal return the default (0.45).
- detect_topic() returns the expected label.
- extract_entities() picks up proper nouns.
- Empty / whitespace input is handled gracefully.
"""

from __future__ import annotations

import pytest

from locomo_memory.phase2.ingestion.importance import TopicImportanceEstimator


@pytest.fixture
def est() -> TopicImportanceEstimator:
    return TopicImportanceEstimator()


# ---------------------------------------------------------------------------
# estimate() — high importance
# ---------------------------------------------------------------------------


class TestHighImportance:
    @pytest.mark.parametrize("claim", [
        "Alice works at Google.",
        "She joined Microsoft on Monday.",
        "Bob was hired by the startup last week.",
        "Caroline resigned from her position.",
        "He was promoted to senior engineer.",
        "She quit her job at Amazon.",
    ])
    def test_employment_is_high(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.85)

    @pytest.mark.parametrize("claim", [
        "John lives in New York.",
        "She moved to London last month.",
        "They relocated to Berlin for work.",
        "He is based in Singapore.",
    ])
    def test_location_is_high(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.85)

    @pytest.mark.parametrize("claim", [
        "Alice and Bob got married in June.",
        "She divorced her husband last year.",
        "They are engaged to be married.",
        "He is separated from his wife.",
    ])
    def test_relationship_is_high(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.85)

    @pytest.mark.parametrize("claim", [
        "She was diagnosed with diabetes.",
        "He had surgery on his knee.",
        "The patient was hospitalized for two weeks.",
        "She found out she was pregnant.",
        "His father passed away in March.",
    ])
    def test_health_is_high(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.85)

    @pytest.mark.parametrize("claim", [
        "She graduated from MIT last year.",
        "He received a degree in computer science.",
        "She enrolled in the PhD program.",
    ])
    def test_education_is_high(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# estimate() — medium importance
# ---------------------------------------------------------------------------


class TestMediumImportance:
    @pytest.mark.parametrize("claim", [
        "She enjoys hiking on weekends.",
        "He loves playing the guitar.",
        "They are interested in astronomy.",
        "She has a hobby of painting.",
    ])
    def test_interests_is_medium(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.55)

    @pytest.mark.parametrize("claim", [
        "She plans to visit Paris next year.",
        "He is going to start a new project.",
        "They have a scheduled appointment on Friday.",
    ])
    def test_plans_is_medium(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.55)

    @pytest.mark.parametrize("claim", [
        "She prefers tea over coffee.",
        "His favorite book is Dune.",
        "She has a favourite restaurant downtown.",
    ])
    def test_preferences_is_medium(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.55)

    @pytest.mark.parametrize("claim", [
        "She grew up in a small town in Ohio.",
        "He was born in Seoul.",
        "Her background is in mechanical engineering.",
    ])
    def test_background_is_medium(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# estimate() — low importance
# ---------------------------------------------------------------------------


class TestLowImportance:
    @pytest.mark.parametrize("claim", [
        "She thinks the project will succeed.",
        "He feels that the team did a good job.",
        "She believes the economy will improve.",
        "Maybe he will join the team next month.",
        "It will probably be fine.",
    ])
    def test_opinion_is_low(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.30)

    @pytest.mark.parametrize("claim", [
        "She mentioned that the meeting was rescheduled.",
        "He told me about the new policy.",
        "They talked about the upcoming event.",
    ])
    def test_meta_speech_is_low(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# estimate() — default
# ---------------------------------------------------------------------------


class TestDefaultImportance:
    @pytest.mark.parametrize("claim", [
        "The meeting was interesting.",
        "The coffee was good today.",
        "That was a long day.",
    ])
    def test_no_signal_is_default(self, est: TopicImportanceEstimator, claim: str) -> None:
        assert est.estimate(claim) == pytest.approx(0.45)

    def test_empty_string(self, est: TopicImportanceEstimator) -> None:
        assert est.estimate("") == pytest.approx(0.45)

    def test_whitespace_only(self, est: TopicImportanceEstimator) -> None:
        assert est.estimate("   ") == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Tier precedence: high beats medium beats low
# ---------------------------------------------------------------------------


class TestTierPrecedence:
    def test_high_beats_medium(self, est: TopicImportanceEstimator) -> None:
        # "plans to" (medium) + "works at" (high) → high
        claim = "She plans to start working at Acme Corp next week."
        assert est.estimate(claim) == pytest.approx(0.85)

    def test_high_beats_low(self, est: TopicImportanceEstimator) -> None:
        # "thinks" (low) + "married" (high) → high
        claim = "He thinks they should get married soon."
        assert est.estimate(claim) == pytest.approx(0.85)

    def test_medium_beats_low(self, est: TopicImportanceEstimator) -> None:
        # "maybe" (low) + "enjoys" (medium) → medium
        claim = "She maybe enjoys hiking on weekends."
        assert est.estimate(claim) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# detect_topic()
# ---------------------------------------------------------------------------


class TestDetectTopic:
    @pytest.mark.parametrize("claim,expected_topic", [
        ("Alice works at Google.", "employment"),
        ("She lives in New York.", "location"),
        ("They are married.", "relationships"),
        ("He was diagnosed with cancer.", "health"),
        ("She graduated from MIT.", "education"),
        ("He enjoys hiking.", "lifestyle"),
        ("She plans to travel to Japan.", "plans"),
        ("He thinks it will rain.", "opinion"),
        ("He owns a Tesla.", "ownership"),
        ("The coffee was fine.", "general"),
        ("", "general"),
    ])
    def test_topic_detection(
        self, est: TopicImportanceEstimator, claim: str, expected_topic: str
    ) -> None:
        assert est.detect_topic(claim) == expected_topic


# ---------------------------------------------------------------------------
# extract_entities()
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_extracts_proper_nouns(self, est: TopicImportanceEstimator) -> None:
        entities = est.extract_entities("Alice works at Google in New York.")
        assert "Alice" in entities
        assert "Google" in entities

    def test_empty_string(self, est: TopicImportanceEstimator) -> None:
        assert est.extract_entities("") == []

    def test_no_entities_in_lowercase(self, est: TopicImportanceEstimator) -> None:
        entities = est.extract_entities("the quick brown fox.")
        assert entities == []

    def test_deduplicates(self, est: TopicImportanceEstimator) -> None:
        entities = est.extract_entities("Alice and Alice both work at Google and Google.")
        assert entities.count("Alice") == 1
        assert entities.count("Google") == 1

    def test_returns_list(self, est: TopicImportanceEstimator) -> None:
        result = est.extract_entities("Some text here.")
        assert isinstance(result, list)
