"""
Evidence Recall@k: fraction of gold evidence IDs found in the retrieved chunks.

A gold evidence ID matches if it appears in any retrieved chunk's dia_ids list.
If no gold evidence IDs are provided, returns None (not 0).
"""

from __future__ import annotations

from locomo_memory.data.schemas import RetrievedChunk


def evidence_recall_at_k(
    gold_evidence_ids: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> float | None:
    """
    Returns fraction of gold IDs covered by retrieved chunks, or None if
    gold_evidence_ids is empty.
    """
    if not gold_evidence_ids:
        return None

    retrieved_dia_ids: set[str] = set()
    for chunk in retrieved_chunks:
        for dia_id in chunk.dia_ids:
            retrieved_dia_ids.add(dia_id)

    hits = sum(1 for gid in gold_evidence_ids if gid in retrieved_dia_ids)
    return round(hits / len(gold_evidence_ids), 4)


def compute_mean_evidence_recall(
    recalls: list[float | None],
) -> float | None:
    valid = [r for r in recalls if r is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)
