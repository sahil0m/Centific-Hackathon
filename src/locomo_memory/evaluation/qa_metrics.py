"""
QA metrics: token-level F1 and exact match.

Normalization follows the SQuAD evaluation convention:
- lowercase
- remove punctuation
- remove articles (a, an, the)
- normalize whitespace
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def compute_metrics_for_batch(
    predictions: list[str],
    gold_answers: list[str],
) -> dict[str, float]:
    assert len(predictions) == len(gold_answers)
    f1_scores = [token_f1(p, g) for p, g in zip(predictions, gold_answers)]
    em_scores = [1.0 if exact_match(p, g) else 0.0 for p, g in zip(predictions, gold_answers)]
    return {
        "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "exact_match": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
    }


def compute_category_metrics(
    predictions: list[str],
    gold_answers: list[str],
    categories: list[str],
) -> dict[str, dict[str, float]]:
    category_data: dict[str, list[tuple[str, str]]] = {}
    for pred, gold, cat in zip(predictions, gold_answers, categories):
        category_data.setdefault(cat, []).append((pred, gold))

    result: dict[str, dict[str, float]] = {}
    for cat, pairs in category_data.items():
        preds, golds = zip(*pairs)
        result[cat] = compute_metrics_for_batch(list(preds), list(golds))
        result[cat]["count"] = len(pairs)
    return result


def compute_latency_percentiles(
    latency_ms_list: list[float],
) -> dict[str, float]:
    if not latency_ms_list:
        return {"p50": 0.0, "p95": 0.0}
    import statistics
    sorted_lat = sorted(latency_ms_list)
    n = len(sorted_lat)
    p50_idx = max(0, int(0.50 * n) - 1)
    p95_idx = max(0, int(0.95 * n) - 1)
    return {
        "p50": round(sorted_lat[p50_idx], 2),
        "p95": round(sorted_lat[p95_idx], 2),
        "mean": round(sum(sorted_lat) / n, 2),
    }
