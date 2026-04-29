"""Memory Candidate Detector — cheap pre-LLM filter.

Sits between the semantic chunker and the LLM-based fact extractor. Decides
whether a chunk is worth sending to the LLM by combining six lightweight
rule-based signals into a 0–1 composite score. If the score is below the
configured threshold, the chunk is skipped (its turns are still archived as
provenance — they just don't go through expensive extraction).

Together with the trivial filter this is expected to skip ~35–45% of
candidate chunks before any LLM is called, with no measurable accuracy loss.

Signals (weights from PHASE2_METHODOLOGY.md §5 Step 3):
    0.30 has_named_entity            — capitalized non-stopword tokens
    0.20 verb_density                — common action verbs per word
    0.15 is_factual_statement        — penalises questions / opinion hedges
    0.15 has_concrete_topic_marker   — work / family / health / time vocab
    0.10 length_normalized           — peak score for 30–150 word chunks
    0.10 has_specific_number_or_date — digits / dates / years / amounts

Implementation notes:
- Pure stdlib + Pydantic. No spaCy, no NLTK, no model load.
- Stateless and thread-safe; multiple threads may share one instance.
- All sub-scores returned for transparency and debugging.
- Tunable via ``CandidateWeights`` and the ``threshold`` argument.
"""

from __future__ import annotations

import re
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Curated word lists (small, high-precision)
# ---------------------------------------------------------------------------


_TOPIC_MARKERS: Final[frozenset[str]] = frozenset({
    # Work / career
    "work", "works", "worked", "working",
    "job", "jobs", "career", "company", "boss", "office", "promotion",
    "salary", "interview", "hired", "fired", "quit", "team", "colleague",
    "manager", "employee", "client", "project",
    # Family
    "family", "spouse", "wife", "husband", "partner", "kid", "kids",
    "child", "children", "son", "daughter", "mother", "father", "mom",
    "dad", "sibling", "brother", "sister", "married", "divorce",
    "wedding", "engaged", "engagement", "fiance", "fiancee",
    # Health
    "doctor", "hospital", "sick", "ill", "illness", "medication", "surgery",
    "diagnosis", "diagnosed", "appointment", "pain", "treatment", "medicine",
    "therapy", "recovery", "patient", "prescription", "symptom", "symptoms",
    # Location / travel
    "moved", "moving", "relocate", "apartment", "house", "city", "country",
    "trip", "travel", "vacation", "flight", "visit", "visited", "abroad",
    # Education
    "school", "college", "university", "degree", "graduate", "graduated",
    "student", "class", "course", "exam",
    # Concrete time markers
    "yesterday", "today", "tomorrow", "tonight", "weekend",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
})


_ACTION_VERBS: Final[frozenset[str]] = frozenset({
    # Auxiliaries / state
    "is", "was", "are", "were", "has", "have", "had",
    "do", "does", "did", "been", "being",
    # Movement / change
    "went", "going", "came", "got", "gave", "took", "made", "make",
    "moved", "started", "stopped", "finished", "began",
    # Communication
    "told", "said", "asked", "answered", "called", "wrote", "read",
    "spoke", "talked", "mentioned", "explained",
    # Decision / state change
    "decided", "chose", "picked", "joined", "left", "quit",
    "married", "divorced", "graduated",
    "hired", "fired", "promoted", "diagnosed", "recovered",
    "bought", "sold", "paid", "earned",
    # Daily action
    "ate", "drank", "saw", "met", "visited", "stayed",
    "worked", "slept", "watched", "played",
})


_OPINION_MARKERS: Final[frozenset[str]] = frozenset({
    "think", "thinks", "thought", "feel", "feels", "felt",
    "guess", "maybe", "perhaps", "probably", "possibly",
    "suppose", "wish", "hope", "wonder", "imagine",
    "seems", "seem", "seemed",
})


_COMMON_CAP_STARTERS: Final[frozenset[str]] = frozenset({
    # Pronouns and very common sentence-initial words that look capitalised
    # but are not proper nouns. Used to strip false positives from the
    # named-entity heuristic.
    "I", "The", "A", "An", "But", "And", "So", "Then", "Now", "Yes", "No",
    "Oh", "Hey", "Well", "Maybe", "Yeah", "Yep", "Sure", "Okay", "Ok",
    "This", "That", "These", "Those", "It", "We", "You", "He", "She", "They",
    "My", "Your", "His", "Her", "Their", "Our",
    "When", "Where", "What", "Why", "How", "Who", "Which",
    "Do", "Did", "Have", "Has", "Is", "Are", "Was", "Were",
})


# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------


_NUMBER_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"\b\d{4}\b"                                   # year (1900-2999)
    r"|\b\d+(?:st|nd|rd|th)\b"                     # 1st, 2nd, 21st...
    r"|\b\d{1,2}[:/-]\d{1,2}(?:[:/-]\d{2,4})?\b"   # date / time 1/2/2024, 12:30
    r"|\b\d+\s*(?:years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b"
    r"|\b\d+\s*(?:dollars?|miles?|km|kg|lbs?|pounds?|hours?)\b"
    r"|\$\d+(?:\.\d+)?"                            # $42, $42.50
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"           # 1,000  1,234,567
    r")",
    re.IGNORECASE,
)

_CAPITALIZED_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z][a-zA-Z'-]+\b")
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\b[a-zA-Z]+\b")
_SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"[.!?]+\s*")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CandidateWeights(BaseModel):
    """Weights for the six candidate-score signals.

    Default values reproduce the recipe in PHASE2_METHODOLOGY.md §5 Step 3.
    Weights must be non-negative and sum to ~1.0 (enforced loosely with a 1e-3
    tolerance).
    """

    model_config = ConfigDict(validate_assignment=True)

    has_named_entity: float = Field(default=0.30, ge=0.0, le=1.0)
    verb_density: float = Field(default=0.20, ge=0.0, le=1.0)
    is_factual_statement: float = Field(default=0.15, ge=0.0, le=1.0)
    has_concrete_topic_marker: float = Field(default=0.15, ge=0.0, le=1.0)
    length_normalized: float = Field(default=0.10, ge=0.0, le=1.0)
    has_specific_number_or_date: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> CandidateWeights:
        total = (
            self.has_named_entity
            + self.verb_density
            + self.is_factual_statement
            + self.has_concrete_topic_marker
            + self.length_normalized
            + self.has_specific_number_or_date
        )
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"CandidateWeights must sum to 1.0 (±1e-3); got {total:.4f}"
            )
        return self


class CandidateScore(BaseModel):
    """Diagnostic record of a single candidate-detector evaluation.

    All sub-scores are in [0, 1] and the composite is a weighted sum that is
    clamped to [0, 1]. ``is_candidate`` is True iff ``score >= threshold``
    that was active at evaluation time.
    """

    model_config = ConfigDict(validate_assignment=True)

    text_preview: str
    score: float = Field(ge=0.0, le=1.0)
    is_candidate: bool

    has_named_entity: float = Field(ge=0.0, le=1.0)
    verb_density: float = Field(ge=0.0, le=1.0)
    is_factual_statement: float = Field(ge=0.0, le=1.0)
    has_concrete_topic_marker: float = Field(ge=0.0, le=1.0)
    length_normalized: float = Field(ge=0.0, le=1.0)
    has_specific_number_or_date: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CandidateDetector:
    """Fast rule-based 'is this worth LLM extraction?' filter.

    Args:
        threshold: minimum composite score for ``is_candidate`` to be True.
            Default 0.35. Pass 0.0 to accept everything (ablation: detector off).
            Pass 1.0 to reject everything.
        weights: optional :class:`CandidateWeights` override.

    The detector is stateless and thread-safe; multiple threads may share a
    single instance.
    """

    DEFAULT_THRESHOLD: ClassVar[float] = 0.35

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        weights: CandidateWeights | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.threshold: float = threshold
        self.weights: CandidateWeights = weights or CandidateWeights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str | None) -> CandidateScore:
        """Compute the composite candidate score and all sub-scores."""
        text = text or ""
        words = _WORD_RE.findall(text)

        ne = self._named_entity_score(text)
        vd = self._verb_density_score(words)
        fs = self._factual_statement_score(text)
        tm = self._topic_marker_score(words)
        ln = self._length_score(len(words))
        nd = 1.0 if _NUMBER_DATE_RE.search(text) else 0.0

        w = self.weights
        composite = (
            w.has_named_entity * ne
            + w.verb_density * vd
            + w.is_factual_statement * fs
            + w.has_concrete_topic_marker * tm
            + w.length_normalized * ln
            + w.has_specific_number_or_date * nd
        )
        composite = max(0.0, min(1.0, composite))

        return CandidateScore(
            text_preview=text[:120],
            score=composite,
            is_candidate=composite >= self.threshold,
            has_named_entity=ne,
            verb_density=vd,
            is_factual_statement=fs,
            has_concrete_topic_marker=tm,
            length_normalized=ln,
            has_specific_number_or_date=nd,
        )

    def is_candidate(self, text: str | None) -> bool:
        """Convenience: True iff the text scores at or above the threshold."""
        return self.score(text).is_candidate

    # ------------------------------------------------------------------
    # Sub-scoring helpers (pure functions, stateless)
    # ------------------------------------------------------------------

    @staticmethod
    def _named_entity_score(text: str) -> float:
        """Count capitalised tokens that are not common sentence-starters.

        3 or more proper-noun candidates → full score; gradient below.
        """
        cap_tokens = _CAPITALIZED_TOKEN_RE.findall(text)
        proper_nouns = [t for t in cap_tokens if t not in _COMMON_CAP_STARTERS]
        return min(1.0, len(proper_nouns) / 3.0)

    @staticmethod
    def _verb_density_score(words: list[str]) -> float:
        """Action verb count / word count, normalised so 1 verb per 8 words → 1.0."""
        if not words:
            return 0.0
        verb_count = sum(1 for w in words if w.lower() in _ACTION_VERBS)
        ratio = verb_count / len(words)
        return min(1.0, ratio * 8.0)

    @staticmethod
    def _factual_statement_score(text: str) -> float:
        """Penalise questions and opinion hedges."""
        text_stripped = text.strip()
        if not text_stripped:
            return 0.0
        if text_stripped.endswith("?"):
            return 0.0

        text_lower = text_stripped.lower()
        question_starters = (
            "do you", "did you", "does ", "are you", "is it", "have you",
            "could you", "would you", "should ", "can you",
            "what do", "what is", "what are", "what was", "what were",
            "how do", "how is", "how was", "how are",
            "where do", "where is", "where are", "where was",
            "when do", "when is", "when was", "when did",
            "why do", "why is", "why are", "why was",
            "who is", "who are", "who was", "who did",
        )
        if text_lower.startswith(question_starters):
            return 0.0

        words = text_lower.split()
        opinion_hits = sum(1 for w in words if w.strip(".,!") in _OPINION_MARKERS)
        if opinion_hits > 0:
            return max(0.0, 1.0 - opinion_hits * 0.3)
        return 1.0

    @staticmethod
    def _topic_marker_score(words: list[str]) -> float:
        """Count topic-marker keywords; 2+ → full score."""
        if not words:
            return 0.0
        marker_hits = sum(1 for w in words if w.lower() in _TOPIC_MARKERS)
        return min(1.0, marker_hits / 2.0)

    @staticmethod
    def _length_score(word_count: int) -> float:
        """Peak score for chunks of 30–150 words; falls off either side."""
        if word_count < 5:
            return 0.0
        if word_count < 30:
            return word_count / 30.0
        if word_count <= 150:
            return 1.0
        if word_count <= 300:
            return 1.0 - (word_count - 150) / 150.0
        return 0.0


__all__ = [
    "CandidateDetector",
    "CandidateScore",
    "CandidateWeights",
]
