"""Rule-based topic importance estimator for atomic claim text.

Used by :class:`~locomo_memory.phase2.ingestion.fact_extractor.FactExtractor`
to populate ``mu.importance`` at ingestion time instead of using a constant.

Design
------
Pattern tiers map to importance buckets:

    High   (0.85) — employment, location/residence, relationships, health/
                    life events, education, major ownership.
    Medium (0.55) — interests/hobbies, plans/appointments, preferences,
                    personal background.
    Low    (0.30) — hedged opinions (think/feel/believe), meta-speech
                    (said, mentioned, discussed).
    Default(0.45) — no recognized topic signal.

Tier precedence is strictly high > medium > low > default.
Hedging words (maybe, probably) are matched as low-importance signals and
only suppressed if a higher-tier signal appears in the same claim.

``detect_topic`` returns a short human-readable label used by the
Lifecycle Engine's label builder when generating CompressedLabel objects.
Topic patterns are evaluated in the order defined in ``_TOPIC_PATTERNS``;
earlier entries win ties.

No LLM call, no external dependencies.
"""

from __future__ import annotations

import re
from typing import Final


# ---------------------------------------------------------------------------
# Compiled pattern tables
# ---------------------------------------------------------------------------

_I = re.IGNORECASE


# High-importance topic matchers (0.85)
_HIGH_PATTERNS: list[re.Pattern[str]] = [
    # Employment / career
    re.compile(
        r"\b(works?\s+at|working\s+at|employed\s+by|hired|quit|resigned|fired|"
        r"promoted|joined|started\s+(at|working)|career|new\s+job|got\s+a\s+job)\b",
        _I,
    ),
    # Location / residence
    re.compile(
        r"\b(lives?\s+in|moved?\s+to|relocated|based\s+in|living\s+in|"
        r"staying\s+in|new\s+home|moved\s+house)\b",
        _I,
    ),
    # Relationships / family
    re.compile(
        r"\b(married|divorced|engaged|separated|broke\s+up|spouse|husband|wife|"
        r"boyfriend|girlfriend|fianc[eé]e?|wedding)\b",
        _I,
    ),
    # Health / major life events
    re.compile(
        r"\b(diagnosed|surgery|hospitali[sz]ed?|cancer|pregnant|pregnanc|"
        r"died|death|passed\s+away|ill(ness)?|disease|treatment|recovering|"
        r"gave\s+birth|had\s+a\s+baby|accident|injury)\b",
        _I,
    ),
    # Education
    re.compile(
        r"\b(graduated|graduation|degree|PhD|masters?|bachelors?|"
        r"university|college|enrolled|accepted\s+(to|into))\b",
        _I,
    ),
    # Major ownership / purchase
    re.compile(
        r"\b(owns?\s+(a|an|the)|bought\s+(a|an|the)|purchased\s+(a|an|the)|"
        r"sold\s+(a|an|the)|house|apartment|condo|car|vehicle)\b",
        _I,
    ),
]

# Medium-importance topic matchers (0.55)
# Note: standalone "will" is excluded because hedged sentences like
# "Maybe he will join" should remain low-importance.
_MEDIUM_PATTERNS: list[re.Pattern[str]] = [
    # Interests / hobbies — broad match (likes, loves, enjoys + any continuation)
    re.compile(
        r"\b(likes?\b|loves?\b|enjoys?\b|hobby|hobbies|"
        r"passionate\s+about|fan\s+of|interested\s+in)\b",
        _I,
    ),
    # Future plans / appointments — specific intent markers only
    re.compile(
        r"\b(plans?\s+to|going\s+to|intends?\s+to|scheduled\s+(to|for)|"
        r"appointment)\b",
        _I,
    ),
    # Preferences
    re.compile(
        r"\b(prefers?|preference|favorite|favourite|best\s+(part|thing|place))\b",
        _I,
    ),
    # Personal background / origin
    re.compile(
        r"\b(grew\s+up|childhood|originally\s+from|background|born\s+in|"
        r"raised\s+in|hometown)\b",
        _I,
    ),
]

# Low-importance topic matchers (0.30)
_LOW_PATTERNS: list[re.Pattern[str]] = [
    # Hedged opinions / uncertain modality
    re.compile(
        r"\b(thinks?|feels?\s+(that|like)|believes?|assumes?|guesses?|"
        r"\bmaybe\b|\bprobably\b|\bperhaps\b|\bmight\b)\b",
        _I,
    ),
    # Meta-speech acts
    re.compile(
        r"\b(said\s+that|mentioned|told\s+(me|him|her|us|them)|"
        r"talked\s+about|discussed|asked\s+about)\b",
        _I,
    ),
]


# ---------------------------------------------------------------------------
# Topic-label patterns (for CompressedLabel.topic)
# Evaluated in order — first match wins.
# opinion and plans come before lifestyle so that hedged/plan claims are
# classified correctly before the broader lifestyle pattern fires.
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("employment",    re.compile(
        r"\b(works?\s+(at|for|in|as)|worked\s+(at|for|in|as)|"
        r"working\s+(at|for|in|as)|"
        r"started\s+working|began\s+working|now\s+works?|currently\s+works?|"
        r"employed|hired|quit|resigned|fired|promoted|"
        r"joined|job|career|company|startup|office|"
        r"engineer|developer|scientist|manager|analyst|designer|director|"
        r"researcher|consultant|programmer|architect|lead|intern)\b", _I)),
    ("location",      re.compile(
        r"\b(lives?\s+in|lived\s+in|living\s+in|"
        r"resides?\s+in|residing\s+in|"
        r"moved?\s+to|relocated|"
        r"based\s+in|located\s+in|stays?\s+in|staying\s+in|"
        r"city|country|home|apartment|house|neighborhood|hometown)\b", _I)),
    ("relationships", re.compile(
        r"\b(married|divorced|engaged|separated|spouse|husband|wife|"
        r"boyfriend|girlfriend|fianc[eé]e?|partner|family|wedding|"
        r"dating|broke\s+up|in\s+a\s+relationship|seeing\s+someone)\b", _I)),
    ("health",        re.compile(
        r"\b(diagnosed|surgery|hospital|cancer|pregnant|died|death|ill(ness)?|"
        r"disease|treatment|medicine|doctor|baby)\b", _I)),
    ("education",     re.compile(
        r"\b(graduated|graduation|degree|school|university|college|"
        r"studi(ed|es|ing)|student|enrolled|accepted\s+(to|into)|"
        r"academic|thesis|dissertation|course|class|campus|major|minor|"
        r"doctorate|postgrad|undergrad|alumni|alumna)\b", _I)),
    ("ownership",     re.compile(
        r"\b(owns?|bought|purchased|sold|house|car|vehicle|property)\b", _I)),
    # opinion before plans so hedged claims ("thinks it will rain") resolve correctly
    ("opinion",       re.compile(
        r"\b(thinks?|feels?|believes?|opinions?|prefers?|guesses?|"
        r"\bmaybe\b|\bprobably\b)\b", _I)),
    # plans before lifestyle so "plans to travel" resolves to plans, not lifestyle
    ("plans",         re.compile(
        r"\b(plans?\s+to|going\s+to|intends?\s+to|scheduled|appointment)\b", _I)),
    ("lifestyle",     re.compile(
        r"\b(hobby|hobbies|interested?\s+in|likes?\b|loves?\b|enjoys?\b|"
        r"food|travel|music|sport|exercise|fitness)\b", _I)),
]

_DEFAULT_TOPIC: Final[str] = "general"

# Proper-noun entity heuristic: capitalized tokens (Title Case or ALL-CAPS 2+)
# that do NOT immediately follow sentence-ending punctuation.
_ENTITY_RE = re.compile(
    r"(?<![.!?\n])\b("
    r"[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*"   # Title Case words
    r"|[A-Z]{2,}"                                 # ALL-CAPS acronyms like IBM, AI
    r")\b"
)

# Words that are capitalized but are NOT named entities — filtered out of
# extract_entities so they don't create false entity-overlap between facts.
# LLM-generated claims often start with "The user …" which makes "The" appear
# as a shared entity across every fact, polluting the overlap signal.
_NON_ENTITY_WORDS: frozenset[str] = frozenset({
    # Articles / determiners
    "The", "A", "An",
    # Prepositions / conjunctions
    "In", "On", "At", "By", "For", "To", "Of", "With", "From", "Into",
    "About", "Through", "During", "Before", "After", "Between", "Under",
    "And", "Or", "But", "So", "As", "If", "Then", "When", "While",
    "Although", "Because", "However", "Therefore", "Despite",
    # Personal pronouns / possessives
    "I", "You", "He", "She", "It", "We", "They",
    "Me", "Him", "Her", "Us", "Them",
    "My", "Your", "His", "Its", "Our", "Their",
    # Demonstratives / relatives
    "This", "That", "These", "Those",
    "What", "Which", "Who", "Whose", "Whom", "When", "Where", "How", "Why",
    # Auxiliary / copula verbs (sometimes capitalised mid-sentence in quoted text)
    "Is", "Are", "Was", "Were", "Has", "Have", "Had",
    "Do", "Does", "Did", "Be", "Been", "Being",
    "Will", "Would", "Could", "Should", "May", "Might",
    # Generic subject nouns used in LLM fact-extraction output
    "User", "Person", "Someone", "Anyone", "Everyone", "Nobody", "People",
    # Common adverbs / quantifiers
    "No", "Not", "Now", "Also", "Just", "Still", "Even", "Only",
    "Both", "All", "More", "Most", "Very", "Much", "Many",
})

# Topic → importance mapping
_TOPIC_IMPORTANCE: Final[dict[str, float]] = {
    "employment": 0.85,
    "location": 0.85,
    "relationships": 0.85,
    "health": 0.85,
    "education": 0.80,
    "ownership": 0.80,
    "lifestyle": 0.55,
    "plans": 0.55,
    "opinion": 0.30,
    _DEFAULT_TOPIC: 0.45,
}


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class TopicImportanceEstimator:
    """Assign an importance score to an atomic claim using topic rules.

    The estimator is stateless and safe to share across threads.

    Methods
    -------
    estimate(claim) → float
        Returns importance in [0.0, 1.0].
    detect_topic(claim) → str
        Returns a short topic label (used by the Lifecycle Engine's label
        builder when generating CompressedLabel objects).
    extract_entities(claim) → list[str]
        Heuristic proper-noun extraction for CompressedLabel.key_entities.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, claim: str) -> float:
        """Return importance in [0, 1] based on topic keyword patterns.

        Tier precedence: high > medium > low > default.
        """
        if not claim or not claim.strip():
            return _TOPIC_IMPORTANCE[_DEFAULT_TOPIC]

        if any(p.search(claim) for p in _HIGH_PATTERNS):
            return 0.85

        if any(p.search(claim) for p in _MEDIUM_PATTERNS):
            return 0.55

        if any(p.search(claim) for p in _LOW_PATTERNS):
            return 0.30

        return _TOPIC_IMPORTANCE[_DEFAULT_TOPIC]

    def detect_topic(self, claim: str) -> str:
        """Return the first matching topic label, or ``'general'``."""
        if not claim:
            return _DEFAULT_TOPIC
        for label, pattern in _TOPIC_PATTERNS:
            if pattern.search(claim):
                return label
        return _DEFAULT_TOPIC

    def extract_entities(self, claim: str) -> list[str]:
        """Extract candidate proper nouns from the claim.

        Uses a simple heuristic: capitalized multi-character tokens that do
        *not* immediately follow a sentence-ending punctuation mark (to avoid
        picking up sentence-initial capitalisation).

        Filters out determiners, pronouns, auxiliary verbs, and generic nouns
        (``_NON_ENTITY_WORDS``) so that common sentence-starter words like
        ``"The"`` do not create false entity-overlap between unrelated facts.
        Single-character tokens are also dropped.
        """
        if not claim:
            return []
        matches = _ENTITY_RE.findall(claim)
        seen: set[str] = set()
        out: list[str] = []
        for m in matches:
            # Drop single-char tokens and known non-entity words
            if len(m) < 2 or m in _NON_ENTITY_WORDS:
                continue
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out


__all__ = ["TopicImportanceEstimator"]
