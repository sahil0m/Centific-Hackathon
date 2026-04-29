"""BM25 sparse index over MemoryUnit claims — Phase 2 Milestone 8B.

Wraps ``rank_bm25.BM25Okapi`` and provides a corpus management layer that
mirrors the ``MemoryFAISSIndex`` interface (add_mu, remove_mu, rebuild,
rebuild_from_store, search).

Design notes
------------
- The BM25 corpus is rebuilt lazily on the next search after any mutation.
- Soft-deletion via an ``_active`` set avoids immediate corpus reshuffling.
- On lazy rebuild the corpus is compacted to only live entries.
- ``rebuild_from_store`` accepts a ``status`` parameter so the same class
  can serve both the active-claim lane and the forgotten-fallback lane.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from loguru import logger

from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Tokenization
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
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BM25SearchResult:
    """One hit from a BM25 search."""

    mu_id: str
    score: float
    rank: int
    conversation_id: str


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class MemoryBM25Index:
    """BM25 sparse index over MemoryUnit claims.

    Corpus entries are stored as ``(mu_id, conversation_id, tokenized_claim)``
    tuples.  Removal marks ``mu_id`` as inactive; the corpus is compacted into
    a fresh BM25Okapi object on the next :meth:`search` call.

    Args:
        None – the index starts empty; populate via :meth:`add_mu` or
            :meth:`rebuild_from_store`.
    """

    def __init__(self) -> None:
        # Raw entries: may contain stale records after remove_mu
        self._entries: list[tuple[str, str, list[str]]] = []  # (mu_id, conv_id, tokens)
        # Currently live mu_ids
        self._active: set[str] = set()
        # Dirty flag — triggers lazy BM25 rebuild
        self._dirty: bool = False
        # Built BM25 model and parallel lists (reset together on rebuild)
        self._bm25 = None
        self._live_mu_ids: list[str] = []
        self._live_conv_ids: list[str] = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._active)

    def __repr__(self) -> str:
        return f"MemoryBM25Index(size={len(self._active)}, dirty={self._dirty})"

    def size(self) -> int:
        """Number of live (non-deleted) entries."""
        return len(self._active)

    def mu_ids(self) -> list[str]:
        """Return mu_ids of all live entries."""
        return list(self._active)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_mu(self, mu: MemoryUnit) -> None:
        """Add or re-add a single MU. Re-adds replace the old entry."""
        if mu.mu_id in self._active:
            self.remove_mu(mu.mu_id)
        tokens = _tokenize(mu.claim)
        self._entries.append((mu.mu_id, mu.conversation_id, tokens))
        self._active.add(mu.mu_id)
        self._dirty = True

    def add_mus(self, mus: list[MemoryUnit]) -> None:
        """Batch add MUs."""
        for mu in mus:
            self.add_mu(mu)

    def remove_mu(self, mu_id: str) -> bool:
        """Soft-delete a MU.  Returns ``True`` if found, ``False`` otherwise."""
        if mu_id not in self._active:
            return False
        self._active.discard(mu_id)
        self._dirty = True
        return True

    def rebuild(self, mus: list[MemoryUnit]) -> None:
        """Replace the entire index with a new set of MUs."""
        self._entries = []
        self._active = set()
        self._dirty = False
        self._bm25 = None
        self._live_mu_ids = []
        self._live_conv_ids = []
        for mu in mus:
            self._entries.append((mu.mu_id, mu.conversation_id, _tokenize(mu.claim)))
            self._active.add(mu.mu_id)
        if self._active:
            self._rebuild_bm25()

    def rebuild_from_store(
        self,
        store: MemoryStore,
        *,
        conversation_id: str | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> int:
        """Rebuild from the store filtered by *status* (default: ACTIVE).

        Setting ``status=MemoryStatus.FORGOTTEN`` powers the forgotten-fallback
        lane in :class:`~locomo_memory.phase2.retrieval.hybrid_retriever.HybridMemoryRetriever`.

        Returns the number of MUs indexed.
        """
        if conversation_id is not None:
            mus = store.list_by_status(conversation_id, status)
        else:
            with store.reader() as conn:
                rows = conn.execute(
                    "SELECT mu_id, conversation_id, claim FROM memory_units WHERE status = ?",
                    (status.value,),
                ).fetchall()
            mus = [
                MemoryUnit(
                    mu_id=r["mu_id"],
                    conversation_id=r["conversation_id"],
                    session_id="__bm25__",
                    claim=r["claim"],
                )
                for r in rows
            ]
        self.rebuild(mus)
        logger.debug(
            "MemoryBM25Index: rebuilt {} MU(s) (status={}, conv={})",
            len(mus), status.value, conversation_id or "all",
        )
        return len(mus)

    # ------------------------------------------------------------------
    # Internal BM25 rebuild
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore[import]

        # Compact: drop stale entries and rebuild parallel lists
        live = [(mid, cid, toks) for mid, cid, toks in self._entries if mid in self._active]
        self._entries = live
        self._live_mu_ids = [mid for mid, cid, toks in live]
        self._live_conv_ids = [cid for mid, cid, toks in live]
        live_tokens = [toks for mid, cid, toks in live]

        if any(live_tokens):
            self._bm25 = BM25Okapi(live_tokens)
        else:
            self._bm25 = None
        self._dirty = False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int,
        *,
        conversation_id: str | None = None,
    ) -> list[BM25SearchResult]:
        """BM25 search over live entries.

        Args:
            query: natural-language query string.
            top_k: maximum number of results.
            conversation_id: if given, restrict results to this conversation.

        Returns ranked :class:`BM25SearchResult` list (best first).
        """
        if top_k <= 0 or not self._active:
            return []

        if self._dirty:
            self._rebuild_bm25()

        if self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        candidates: list[tuple[float, str, str]] = []
        for i, (mu_id, conv_id) in enumerate(
            zip(self._live_mu_ids, self._live_conv_ids)
        ):
            if conversation_id is not None and conv_id != conversation_id:
                continue
            candidates.append((float(scores[i]), mu_id, conv_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:top_k]

        return [
            BM25SearchResult(mu_id=mid, score=score, rank=rank, conversation_id=conv)
            for rank, (score, mid, conv) in enumerate(candidates, start=1)
        ]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = ["BM25SearchResult", "MemoryBM25Index"]
