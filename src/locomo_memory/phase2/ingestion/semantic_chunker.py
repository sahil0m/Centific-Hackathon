"""Semantic Chunker — group consecutive turns by topic similarity.

Replaces fixed-size sliding windows (Phase 1's ``window3``) with variable-size
chunks whose boundaries follow real topic shifts in the conversation. A turn
that semantically continues the previous one extends the current chunk; a
turn whose embedding diverges starts a new chunk.

Algorithm (PHASE2_METHODOLOGY.md §5 Step 2):

    1. Embed every turn individually (one batched call to the embedder).
    2. Walk the turns in order. For each turn t_i:
         a. If this is the first turn of the chunk, append it.
         b. Otherwise compute cosine(t_i, reference). If similarity >=
            threshold, append; else close current chunk and start a new one.
       The reference is either the immediately previous turn's embedding
       ("previous_turn", default — matches the methodology spec) or the
       running mean of the current chunk's embeddings ("chunk_centroid",
       more robust to slow topic drift).
    3. Close the final chunk after the last turn.

Output: Phase 1 ``Chunk`` objects with ``chunk_strategy="semantic"``. This
means downstream code (FAISS index, BM25 index, retriever, evaluator) can
consume semantic chunks without modification.

The embedder is injected via the ``TurnEmbedder`` Protocol so tests can
substitute a deterministic fake without loading sentence-transformers.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from locomo_memory.data.schemas import Chunk, Conversation, Turn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder protocol (decouples from EmbeddingGenerator for testability)
# ---------------------------------------------------------------------------


@runtime_checkable
class TurnEmbedder(Protocol):
    """Anything with ``embed_texts(list[str]) -> ndarray[(N, D), float32]``."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


_ComparisonMode = Literal["previous_turn", "chunk_centroid"]


class SemanticChunker:
    """Topic-boundary chunker driven by sentence-embedding similarity.

    Args:
        embedder: implements :class:`TurnEmbedder`. Embeddings are expected
            to be L2-normalised — the chunker does not re-normalise. (BGE
            models normalise by default; pass ``normalize=True`` to
            ``EmbeddingGenerator``.)
        similarity_threshold: cosine similarity at or above which two turns
            are considered same-topic. Default 0.65 from the methodology.
            Higher values produce more, smaller chunks.
        comparison_mode: ``"previous_turn"`` (compare to last turn — the
            spec) or ``"chunk_centroid"`` (compare to the running mean of
            the current chunk).
        max_chunk_size: hard cap on turns per chunk; ``None`` means no cap.
            Useful so a single long monologue does not absorb everything.
        min_chunk_size: minimum turns per chunk before considering a split.
            Default 1 (allow single-turn chunks).
        include_speaker / include_timestamp / include_session_id: chunk-text
            formatting flags that mirror the Phase 1 chunkers.

    Threading: a single ``SemanticChunker`` is safe to share across threads
    only if the underlying embedder is thread-safe. Calls do not mutate
    instance state.
    """

    def __init__(
        self,
        embedder: TurnEmbedder,
        similarity_threshold: float = 0.65,
        comparison_mode: _ComparisonMode = "previous_turn",
        max_chunk_size: int | None = None,
        min_chunk_size: int = 1,
        include_speaker: bool = True,
        include_timestamp: bool = True,
        include_session_id: bool = True,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [-1, 1], got {similarity_threshold}"
            )
        if comparison_mode not in ("previous_turn", "chunk_centroid"):
            raise ValueError(
                f"comparison_mode must be 'previous_turn' or 'chunk_centroid'; "
                f"got {comparison_mode!r}"
            )
        if max_chunk_size is not None and max_chunk_size < 1:
            raise ValueError(f"max_chunk_size must be >= 1 or None; got {max_chunk_size}")
        if min_chunk_size < 1:
            raise ValueError(f"min_chunk_size must be >= 1; got {min_chunk_size}")
        if max_chunk_size is not None and min_chunk_size > max_chunk_size:
            raise ValueError(
                "min_chunk_size cannot exceed max_chunk_size "
                f"(min={min_chunk_size}, max={max_chunk_size})"
            )

        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.comparison_mode = comparison_mode
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_speaker = include_speaker
        self.include_timestamp = include_timestamp
        self.include_session_id = include_session_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_conversations(
        self, conversations: list[Conversation]
    ) -> list[Chunk]:
        """Chunk every conversation; deduplicate any colliding chunk_ids."""
        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()
        for conv in conversations:
            for chunk in self.chunk_conversation(conv):
                if chunk.chunk_id in seen_ids:
                    logger.warning(
                        "Duplicate chunk_id detected: %s — skipping", chunk.chunk_id
                    )
                    continue
                seen_ids.add(chunk.chunk_id)
                all_chunks.append(chunk)
        logger.info(
            "Built %d semantic chunks from %d conversations (threshold=%.2f, mode=%s)",
            len(all_chunks),
            len(conversations),
            self.similarity_threshold,
            self.comparison_mode,
        )
        return all_chunks

    def chunk_conversation(self, conv: Conversation) -> list[Chunk]:
        """Chunk a single conversation. Filters synthetic 'summary' turns."""
        turns = [t for t in conv.turns if t.speaker.lower() != "summary"]
        return self._chunk_turns(turns, conv.conversation_id, conv.sample_id)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _chunk_turns(
        self, turns: list[Turn], conversation_id: str, sample_id: str
    ) -> list[Chunk]:
        if not turns:
            return []

        if len(turns) == 1:
            return [self._build_chunk(turns, conversation_id, sample_id)]

        # Embed all turns in one shot — the embedder is responsible for batching.
        texts = [t.text for t in turns]
        embeddings = self.embedder.embed_texts(texts)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(turns):
            raise RuntimeError(
                f"embedder returned {embeddings.shape[0]} vectors for "
                f"{len(turns)} turns"
            )

        chunks: list[Chunk] = []
        current_indices: list[int] = [0]

        for i in range(1, len(turns)):
            ref_vec = self._reference_vector(embeddings, current_indices)
            sim = float(np.dot(ref_vec, embeddings[i]))

            should_extend = sim >= self.similarity_threshold
            if (
                self.max_chunk_size is not None
                and len(current_indices) >= self.max_chunk_size
            ):
                should_extend = False

            if should_extend:
                current_indices.append(i)
            else:
                # Honour min_chunk_size: if the current chunk is too small,
                # extend regardless. This avoids producing 0-turn artefacts.
                if len(current_indices) < self.min_chunk_size:
                    current_indices.append(i)
                    continue
                chunks.append(
                    self._build_chunk(
                        [turns[j] for j in current_indices],
                        conversation_id,
                        sample_id,
                    )
                )
                current_indices = [i]

        # Close the final chunk
        chunks.append(
            self._build_chunk(
                [turns[j] for j in current_indices],
                conversation_id,
                sample_id,
            )
        )
        return chunks

    def _reference_vector(
        self, embeddings: np.ndarray, current_indices: list[int]
    ) -> np.ndarray:
        if self.comparison_mode == "previous_turn":
            return embeddings[current_indices[-1]]
        # chunk_centroid: mean of current chunk's vectors, re-normalised so
        # cosine semantics are preserved.
        centroid = embeddings[current_indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0.0:
            return centroid
        return centroid / norm

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------

    def _build_chunk(
        self, turns: list[Turn], conversation_id: str, sample_id: str
    ) -> Chunk:
        if not turns:
            raise ValueError("_build_chunk called with empty turn list")
        start_idx = turns[0].turn_index
        end_idx = turns[-1].turn_index
        chunk_id = f"{conversation_id}#semantic#{start_idx}-{end_idx}"

        return Chunk(
            chunk_id=chunk_id,
            conversation_id=conversation_id,
            sample_id=sample_id,
            session_id=turns[0].session_id,
            turn_index_start=start_idx,
            turn_index_end=end_idx,
            dia_ids=[t.dia_id for t in turns if t.dia_id],
            speakers=[t.speaker for t in turns],
            timestamps=[t.timestamp for t in turns],
            text=self._format_chunk_text(turns, conversation_id),
            chunk_strategy="semantic",
        )

    def _format_chunk_text(
        self, turns: list[Turn], conversation_id: str
    ) -> str:
        first = turns[0]
        header_parts = [f"Conversation: {conversation_id}"]
        if self.include_session_id:
            header_parts.append(f"Session: {first.session_id}")
        dia_ids_present = [t.dia_id for t in turns if t.dia_id]
        if dia_ids_present:
            header_parts.append(f"Dialog IDs: {','.join(dia_ids_present)}")
        header = "[" + " | ".join(header_parts) + "]"

        body_lines: list[str] = []
        for t in turns:
            prefix_parts: list[str] = []
            if self.include_speaker and t.speaker:
                prefix_parts.append(t.speaker)
            if self.include_timestamp and t.timestamp:
                prefix_parts.append(f"[{t.timestamp}]")
            prefix = " ".join(prefix_parts)
            line = f"{prefix}: {t.text}" if prefix else t.text
            body_lines.append(line)

        return header + "\n" + "\n".join(body_lines)


__all__ = ["SemanticChunker", "TurnEmbedder"]
