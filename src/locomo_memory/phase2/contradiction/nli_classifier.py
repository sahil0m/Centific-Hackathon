"""NLI-based classifier for contradiction detection — Phase 2.

Research basis
--------------
Natural Language Inference (NLI) / Textual Entailment is the standard NLP
task for determining whether one sentence (premise) *entails*, *contradicts*,
or is *neutral* with respect to another (hypothesis).

    Bowman et al. (2015) — Stanford NLI corpus (SNLI)
    Williams et al. (2018) — Multi-genre NLI corpus (MultiNLI)
    He et al. (2021) — DeBERTa: fine-tuned cross-encoder for NLI

Model: ``cross-encoder/nli-deberta-v3-large``
    - DeBERTa-v3-large fine-tuned jointly on SNLI + MultiNLI
    - Output label order: [contradiction, entailment, neutral]
    - State-of-the-art zero-shot contradiction detection accuracy

Why NLI beats pure token overlap for contradiction:
    Token-overlap (Jaccard) can't distinguish "He graduated from MIT" from
    "He *never* graduated from MIT" because they share all content tokens.
    A cross-encoder NLI model reads both claims jointly, capturing negation
    and semantic polarity.

Usage (production)::

    clf = NLIContradictionClassifier()
    scores = clf.classify("Alice works at Google", "Alice no longer works at Google")
    # scores.contradiction ≈ 0.92

Usage (tests — no model download)::

    clf = FakeNLIClassifier()
    scores = clf.classify(...)
"""

from __future__ import annotations

from dataclasses import dataclass

from locomo_memory.phase2.contradiction.resolver import (
    _NEGATION_RE,
    _jaccard,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class NLIScores:
    """Probabilities for each NLI label (must sum to ~1.0).

    Attributes:
        contradiction: probability that claim_b *contradicts* claim_a.
        entailment: probability that claim_a *entails* claim_b.
        neutral: probability of no logical relationship.
    """

    contradiction: float
    entailment: float
    neutral: float


# ---------------------------------------------------------------------------
# Real NLI classifier (lazy model loading)
# ---------------------------------------------------------------------------


class NLIContradictionClassifier:
    """DeBERTa-v3-large NLI cross-encoder for contradiction detection.

    The model is lazy-loaded on the first :meth:`classify` call to avoid
    import-time cost.  Tests should use :class:`FakeNLIClassifier` instead
    to avoid downloading a ~1.5 GB model.

    Args:
        model_name: HuggingFace model identifier.
            The default is the highest-accuracy publicly available NLI
            cross-encoder at time of writing.
    """

    # Label order matches cross-encoder/nli-* output: [contradiction, entailment, neutral]
    _IDX_CONTRADICTION = 0
    _IDX_ENTAILMENT = 1
    _IDX_NEUTRAL = 2

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
    ) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for NLIContradictionClassifier. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(self._model_name)

    def classify(self, claim_a: str, claim_b: str) -> NLIScores:
        """Score the NLI relationship between claim_a (premise) and claim_b (hypothesis).

        Args:
            claim_a: existing / older claim (premise).
            claim_b: incoming / newer claim (hypothesis).

        Returns:
            :class:`NLIScores` with contradiction, entailment, neutral probabilities.
        """
        self._load()
        probs = self._model.predict(
            [(claim_a, claim_b)],
            apply_softmax=True,
        )[0].tolist()
        return NLIScores(
            contradiction=float(probs[self._IDX_CONTRADICTION]),
            entailment=float(probs[self._IDX_ENTAILMENT]),
            neutral=float(probs[self._IDX_NEUTRAL]),
        )


# ---------------------------------------------------------------------------
# Deterministic fake classifier for tests
# ---------------------------------------------------------------------------


class FakeNLIClassifier:
    """Deterministic NLI classifier for unit tests — no model download.

    Uses token Jaccard and negation patterns to approximate what a real NLI
    model would produce on the claims used in the test suite.

    Decision rules (deterministic):
        1. Asymmetric negation (B negates A) + Jaccard ≥ 0.25  → contradiction ≈ 0.85
        2. High lexical overlap (Jaccard ≥ 0.60)               → entailment ≈ (0.87–0.95)
        3. Otherwise                                            → neutral (0.10 / 0.10 / 0.80)
    """

    def classify(self, claim_a: str, claim_b: str) -> NLIScores:
        tok_a = _tokenize(claim_a)
        tok_b = _tokenize(claim_b)
        j = _jaccard(tok_a, tok_b)

        has_neg_b = bool(_NEGATION_RE.search(claim_b))
        has_neg_a = bool(_NEGATION_RE.search(claim_a))

        # Asymmetric negation: B explicitly negates something A stated positively.
        # A real NLI model strongly predicts contradiction in this case.
        if has_neg_b and not has_neg_a and j >= 0.25:
            return NLIScores(contradiction=0.85, entailment=0.05, neutral=0.10)

        # Near-duplicate / high lexical overlap: a real NLI model predicts entailment.
        # Scale entailment with j: j=1.0 → 0.95, j=0.60 → 0.87.
        if j >= 0.60:
            ent = round(0.75 + 0.20 * j, 4)
            con = 0.05
            neu = round(1.0 - ent - con, 4)
            return NLIScores(contradiction=con, entailment=ent, neutral=neu)

        # Default: neutral — no strong semantic signal in either direction.
        return NLIScores(contradiction=0.10, entailment=0.10, neutral=0.80)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "FakeNLIClassifier",
    "NLIContradictionClassifier",
    "NLIScores",
]
