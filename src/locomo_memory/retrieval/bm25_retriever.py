"""
BM25 sparse retriever using rank-bm25 (BM25Okapi).

One BM25 index is built per conversation, mirroring the FAISS per-conversation
design so retrieval is always scoped to a single conversation.
"""

from __future__ import annotations

import logging
import re
import string

from locomo_memory.data.schemas import Chunk

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
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
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t and t not in _STOPWORDS]


class BM25ConversationIndex:
    """BM25 index for one conversation's chunks."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self._chunks: list[Chunk] = []
        self._index = None

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        # Guard against all-empty token lists (BM25Okapi crashes on empty corpus)
        if not any(tokenized):
            self._index = None
            return
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self._index is None or not self._chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(self._chunks[i], float(s)) for i, s in ranked if s > 0]


class MultiBM25Index:
    """Manages one BM25ConversationIndex per conversation."""

    def __init__(self) -> None:
        self._indices: dict[str, BM25ConversationIndex] = {}

    def build_all(self, chunks_by_conv: dict[str, list[Chunk]]) -> None:
        for conv_id, chunks in chunks_by_conv.items():
            if not chunks:
                continue
            idx = BM25ConversationIndex(conversation_id=conv_id)
            idx.build(chunks)
            self._indices[conv_id] = idx
        logger.info("Built BM25 indices for %d conversations", len(self._indices))

    def search(
        self, conversation_id: str, query: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        idx = self._indices.get(conversation_id)
        if idx is None:
            return []
        return idx.search(query, top_k)
