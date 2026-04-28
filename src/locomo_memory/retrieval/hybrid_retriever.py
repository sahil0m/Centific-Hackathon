"""
Hybrid retriever: Dense (FAISS) + Sparse (BM25) fused with
Reciprocal Rank Fusion (RRF).

RRF formula (Cormack et al., 2009):
    score(d) = Σ_i  1 / (k + rank_i(d))
where k=60 is the standard constant that dampens the impact of very high ranks.

Each retriever independently fetches a candidate pool of size `candidate_k`
(default 3×top_k). Their ranked lists are merged via RRF and the final top_k
is returned. This means the correct chunk only needs to appear in EITHER the
dense or BM25 list — significantly improving recall vs either alone.
"""

from __future__ import annotations

import logging
import time

from locomo_memory.data.schemas import RetrievedChunk
from locomo_memory.indexing.embeddings import EmbeddingGenerator
from locomo_memory.indexing.vector_index import MultiConversationIndex
from locomo_memory.retrieval.bm25_retriever import MultiBM25Index
from locomo_memory.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant from the original RRF paper


def _rrf_fuse(
    rankings: list[list[tuple]],   # each list is [(chunk, score), ...]
    top_k: int,
) -> list[tuple]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Returns up to top_k (chunk, rrf_score) pairs, highest score first.
    """
    doc_scores: dict[str, tuple] = {}  # chunk_id -> (chunk, accumulated_rrf)

    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking):
            cid = chunk.chunk_id
            rrf_contrib = 1.0 / (_RRF_K + rank + 1)
            if cid not in doc_scores:
                doc_scores[cid] = (chunk, 0.0)
            prev_chunk, prev_score = doc_scores[cid]
            doc_scores[cid] = (prev_chunk, prev_score + rrf_contrib)

    fused = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


class HybridRetriever:
    """
    Combines FAISS dense retrieval and BM25 sparse retrieval via RRF.

    Args:
        dense_index:  MultiConversationIndex (FAISS)
        bm25_index:   MultiBM25Index
        embedder:     EmbeddingGenerator for query embedding
        top_k:        Final number of chunks to return
        candidate_k:  Candidate pool size per retriever (default = 3 * top_k).
                      Larger pool means more chances the gold chunk appears in
                      at least one list before fusion.
    """

    def __init__(
        self,
        dense_index: MultiConversationIndex,
        bm25_index: MultiBM25Index,
        embedder: EmbeddingGenerator,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> None:
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.top_k = top_k
        self.candidate_k = candidate_k or max(top_k * 3, 20)

    def retrieve(
        self,
        conversation_id: str,
        qa_id: str,
        question: str,
    ) -> RetrievalResult:
        t0 = time.perf_counter()

        # Dense retrieval
        query_vec = self.embedder.embed_query(question)
        dense_hits = self.dense_index.search(
            conversation_id, query_vec, self.candidate_k
        )

        # BM25 retrieval
        bm25_hits = self.bm25_index.search(
            conversation_id, question, self.candidate_k
        )

        # RRF fusion
        fused = _rrf_fuse([dense_hits, bm25_hits], top_k=self.top_k)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                dia_ids=chunk.dia_ids,
                session_id=chunk.session_id,
                speaker=", ".join(chunk.speakers),
                text=chunk.text,
                score=rrf_score,
            )
            for chunk, rrf_score in fused
        ]

        logger.debug(
            "qa_id=%s | dense=%d BM25=%d fused=%d | %.1f ms",
            qa_id, len(dense_hits), len(bm25_hits), len(retrieved), latency_ms,
        )
        return RetrievalResult(
            conversation_id=conversation_id,
            qa_id=qa_id,
            question=question,
            retrieved=retrieved,
            retrieval_latency_ms=latency_ms,
        )

    def retrieve_batch(
        self, queries: list[tuple[str, str, str]]
    ) -> list[RetrievalResult]:
        return [self.retrieve(cid, qid, q) for cid, qid, q in queries]
