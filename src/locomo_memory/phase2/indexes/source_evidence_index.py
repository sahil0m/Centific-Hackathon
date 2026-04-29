"""Source Evidence Index — Phase 2 Milestone 13.

Multi-Granularity Provenance Evidence Retrieval: indexes original dialogue
turns so the hybrid retriever has a 5th lane that searches at the raw source
level rather than abstracted MemoryUnit claims.

Key properties:
- Each entry preserves dia_id as the canonical evidence identifier.
- Optional ±N context window broadens the search text without changing dia_id.
- Linked MU map enables RRF boosting of MemoryUnits via source provenance.
- Turns with no linked MU can still be returned as transient evidence hits.
- Deleted memory is never returned (enforced at HybridMemoryRetriever level).
- Gold evidence is never used: search is query-only.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

from loguru import logger

# ---------------------------------------------------------------------------
# Tokenization — same stopwords as MemoryBM25Index for consistency
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "through", "about", "and",
    "or", "but", "not", "what", "when", "where", "who", "which", "how",
    "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "my", "your", "his", "her", "our", "their",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceEvidenceEntry:
    """One dialogue turn stored in the source evidence index."""

    dia_id: str
    conversation_id: str
    session_id: str
    speaker: str
    text: str
    timestamp: str | None
    turn_index: int
    linked_mu_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceEvidenceHit:
    """One result from a source evidence BM25 search."""

    entry: SourceEvidenceEntry
    context_text: str
    """Text of the central turn plus optional ±window context turns."""
    score: float
    rank: int


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class SourceEvidenceIndex:
    """BM25 index over raw dialogue turns for provenance retrieval.

    Provides a 5th retrieval lane in :class:`HybridMemoryRetriever` that
    searches at the raw source level rather than abstracted MemoryUnit claims.

    Indexed at the turn (dia_id) level.  Each entry preserves the original
    turn's ``dia_id`` so evidence-recall metrics remain comparable.

    Linked MU map: when a MemoryUnit's ``source_dia_ids`` reference a turn, a
    link is registered via :meth:`link_mu` or :meth:`build_links_from_mus`.
    Source evidence hits then carry these ``linked_mu_ids`` so the retriever
    can boost the corresponding MemoryUnits via RRF.

    Turns not linked to any MU (e.g. turns that were too short to ingest, or
    whose MU was deleted) are tracked separately and can be returned as
    transient evidence hits by the retriever.
    """

    def __init__(self) -> None:
        # dia_id → SourceEvidenceEntry
        self._entries: dict[str, SourceEvidenceEntry] = {}
        # conversation_id → ordered list of dia_ids (turn order)
        self._conv_turn_order: dict[str, list[str]] = {}
        # BM25 corpus (dia_id, conv_id, tokens)
        self._corpus: list[tuple[str, str, list[str]]] = []
        self._active: set[str] = set()
        self._dirty: bool = False
        # Parallel lists populated by _rebuild_bm25
        self._bm25 = None
        self._live_dia_ids: list[str] = []
        self._live_conv_ids: list[str] = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._active)

    def size(self) -> int:
        """Number of indexed turns."""
        return len(self._active)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def add_turns(
        self,
        turns,  # list[Turn] — typed generically to avoid circular import
        *,
        dia_to_mu_ids: dict[str, list[str]] | None = None,
    ) -> int:
        """Index dialogue turns from a conversation.

        Args:
            turns: iterable of Turn-like objects with attributes
                ``dia_id``, ``conversation_id``, ``session_id``,
                ``speaker``, ``text``, ``timestamp``, ``turn_index``.
            dia_to_mu_ids: optional pre-built dia_id → [mu_id] map for
                bulk linking at add time.  Individual links can also be
                set later via :meth:`link_mu` or :meth:`build_links_from_mus`.

        Returns the number of new turns indexed.
        """
        added = 0
        for i, turn in enumerate(turns):
            dia_id = getattr(turn, "dia_id", None) or getattr(turn, "dialog_id", None)
            if not dia_id:
                continue
            conv_id = getattr(turn, "conversation_id", "")
            entry = SourceEvidenceEntry(
                dia_id=dia_id,
                conversation_id=conv_id,
                session_id=getattr(turn, "session_id", ""),
                speaker=getattr(turn, "speaker", ""),
                text=getattr(turn, "text", ""),
                timestamp=getattr(turn, "timestamp", None),
                turn_index=getattr(turn, "turn_index", i),
                linked_mu_ids=list((dia_to_mu_ids or {}).get(dia_id, [])),
            )
            self._entries[dia_id] = entry
            if conv_id not in self._conv_turn_order:
                self._conv_turn_order[conv_id] = []
            if dia_id not in self._conv_turn_order[conv_id]:
                self._conv_turn_order[conv_id].append(dia_id)

            if dia_id not in self._active:
                self._corpus.append((dia_id, conv_id, _tokenize(entry.text)))
                self._active.add(dia_id)
                added += 1
        if added:
            self._dirty = True
        logger.debug(
            "SourceEvidenceIndex: added {} turns ({} total)",
            added, len(self._active),
        )
        return added

    def link_mu(self, dia_id: str, mu_id: str) -> None:
        """Register that a MemoryUnit cites this dia_id as source evidence."""
        entry = self._entries.get(dia_id)
        if entry is None:
            return
        if mu_id not in entry.linked_mu_ids:
            entry.linked_mu_ids.append(mu_id)

    def build_links_from_mus(self, mus) -> int:
        """Batch-link dia_ids from MemoryUnit.source_dia_ids.

        Iterates the given MU list and calls :meth:`link_mu` for every
        dia_id in ``mu.source_dia_ids``.

        Returns the number of (dia_id, mu_id) link pairs registered.
        """
        count = 0
        for mu in mus:
            for dia_id in mu.source_dia_ids:
                self.link_mu(dia_id, mu.mu_id)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Context window
    # ------------------------------------------------------------------

    def get_context_text(self, dia_id: str, window: int = 0) -> str:
        """Return turn text with ±window adjacent turns.

        The central dia_id's text is always included.  Context turns are
        from the same conversation and in turn order.  The dia_id itself
        is unchanged — only the search text is broadened.

        Args:
            dia_id: central turn identifier.
            window: number of turns before/after the central turn to include.

        Returns a single space-joined string, or empty string if not found.
        """
        entry = self._entries.get(dia_id)
        if entry is None:
            return ""
        if window == 0:
            return entry.text

        turn_order = self._conv_turn_order.get(entry.conversation_id, [])
        try:
            center_idx = turn_order.index(dia_id)
        except ValueError:
            return entry.text

        start = max(0, center_idx - window)
        end = min(len(turn_order), center_idx + window + 1)

        parts: list[str] = []
        for dia in turn_order[start:end]:
            e = self._entries.get(dia)
            if e:
                part = f"{e.speaker}: {e.text}" if e.speaker else e.text
                parts.append(part)
        return " ".join(parts)

    def get_linked_mu_ids(self, dia_id: str) -> list[str]:
        """Return the list of mu_ids linked to this dia_id."""
        entry = self._entries.get(dia_id)
        return list(entry.linked_mu_ids) if entry else []

    # ------------------------------------------------------------------
    # BM25 internals
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore[import]

        live = [(did, cid, toks) for did, cid, toks in self._corpus if did in self._active]
        self._corpus = live
        self._live_dia_ids = [did for did, cid, toks in live]
        self._live_conv_ids = [cid for did, cid, toks in live]
        live_tokens = [toks for did, cid, toks in live]
        self._bm25 = BM25Okapi(live_tokens) if any(live_tokens) else None
        self._dirty = False

    # ------------------------------------------------------------------
    # Search — query only, no gold evidence
    # ------------------------------------------------------------------

    def search_bm25(
        self,
        query: str,
        top_n: int,
        *,
        conversation_id: str | None = None,
        context_window: int = 0,
    ) -> list[SourceEvidenceHit]:
        """BM25 search over indexed dialogue turns.

        Searches using *query* only.  Gold evidence IDs are never accepted
        or used.

        Args:
            query: natural-language query string.
            top_n: maximum results.
            conversation_id: if given, restrict to this conversation.
            context_window: adjacent turns to include in ``context_text``.
                The central dia_id remains the canonical evidence identifier.

        Returns ranked :class:`SourceEvidenceHit` list (best first).
        """
        if top_n <= 0 or not self._active:
            return []
        if self._dirty:
            self._rebuild_bm25()
        if self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        candidates: list[tuple[float, str]] = []
        for i, (dia_id, conv_id) in enumerate(
            zip(self._live_dia_ids, self._live_conv_ids)
        ):
            if conversation_id is not None and conv_id != conversation_id:
                continue
            candidates.append((float(scores[i]), dia_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:top_n]

        hits: list[SourceEvidenceHit] = []
        for rank, (score, dia_id) in enumerate(candidates, start=1):
            entry = self._entries[dia_id]
            ctx = self.get_context_text(dia_id, window=context_window)
            hits.append(SourceEvidenceHit(
                entry=entry,
                context_text=ctx,
                score=score,
                rank=rank,
            ))
        return hits


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "SourceEvidenceEntry",
    "SourceEvidenceHit",
    "SourceEvidenceIndex",
]
