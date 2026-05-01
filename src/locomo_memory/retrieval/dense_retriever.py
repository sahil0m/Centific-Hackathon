"""
Dense retriever: embeds a question, searches the per-conversation FAISS index,
returns top-k chunks with scores and retrieval latency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass


from locomo_memory.data.schemas import Chunk, RetrievedChunk
from locomo_memory.indexing.embeddings import EmbeddingGenerator
from locomo_memory.indexing.vector_index import MultiConversationIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    conversation_id: str
    qa_id: str
    question: str
    retrieved: list[RetrievedChunk]
    retrieval_latency_ms: float


class DenseRetriever:
    def __init__(
        self,
        index: MultiConversationIndex,
        embedder: EmbeddingGenerator,
        top_k: int = 5,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self,
        conversation_id: str,
        qa_id: str,
        question: str,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(question)
        hits = self.index.search(conversation_id, query_vec, self.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved: list[RetrievedChunk] = []
        for chunk, score in hits:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    dia_ids=chunk.dia_ids,
                    session_id=chunk.session_id,
                    speaker=", ".join(chunk.speakers),
                    text=chunk.text,
                    score=score,
                )
            )

        logger.debug(
            "qa_id=%s | retrieved %d chunks in %.1f ms",
            qa_id,
            len(retrieved),
            latency_ms,
        )
        return RetrievalResult(
            conversation_id=conversation_id,
            qa_id=qa_id,
            question=question,
            retrieved=retrieved,
            retrieval_latency_ms=latency_ms,
        )

    def retrieve_batch(
        self,
        queries: list[tuple[str, str, str]],
    ) -> list[RetrievalResult]:
        """
        queries: list of (conversation_id, qa_id, question)
        Returns results in the same order.
        """
        results = []
        for conv_id, qa_id, question in queries:
            results.append(self.retrieve(conv_id, qa_id, question))
        return results
