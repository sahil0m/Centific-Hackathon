"""Tests for QA metrics: F1, exact match, latency percentiles."""

import pytest

from locomo_memory.evaluation.qa_metrics import (
    normalize_answer,
    token_f1,
    exact_match,
    compute_metrics_for_batch,
    compute_category_metrics,
    compute_latency_percentiles,
)


class TestNormalize:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"

    def test_removes_articles(self):
        assert normalize_answer("The cat sat on a mat") == "cat sat on mat"
        assert normalize_answer("An apple a day") == "apple day"

    def test_normalizes_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"


class TestTokenF1:
    def test_exact(self):
        assert token_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert token_f1("foo bar", "baz qux") == 0.0

    def test_partial_overlap(self):
        f1 = token_f1("hello world foo", "hello world bar")
        assert 0 < f1 < 1.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_prediction_empty(self):
        assert token_f1("", "hello") == 0.0

    def test_gold_empty(self):
        assert token_f1("hello", "") == 0.0

    def test_case_insensitive(self):
        assert token_f1("Hello", "hello") == 1.0

    def test_articles_ignored(self):
        f1 = token_f1("the cat", "cat")
        assert f1 == 1.0

    def test_no_info_answer(self):
        f1 = token_f1("No information available.", "John")
        assert f1 == 0.0


class TestExactMatch:
    def test_exact(self):
        assert exact_match("hello world", "hello world") is True

    def test_case_insensitive(self):
        assert exact_match("Hello", "hello") is True

    def test_no_match(self):
        assert exact_match("foo", "bar") is False

    def test_article_stripped(self):
        assert exact_match("the cat", "cat") is True


class TestBatchMetrics:
    def test_all_correct(self):
        m = compute_metrics_for_batch(["hello", "world"], ["hello", "world"])
        assert m["avg_f1"] == 1.0
        assert m["exact_match"] == 1.0

    def test_all_wrong(self):
        m = compute_metrics_for_batch(["foo", "bar"], ["baz", "qux"])
        assert m["avg_f1"] == 0.0
        assert m["exact_match"] == 0.0

    def test_empty(self):
        m = compute_metrics_for_batch([], [])
        assert m["avg_f1"] == 0.0


class TestCategoryMetrics:
    def test_basic(self):
        preds = ["hello", "world", "foo"]
        golds = ["hello", "baz", "foo"]
        cats = ["single_hop", "multi_hop", "single_hop"]
        cat_m = compute_category_metrics(preds, golds, cats)
        assert "single_hop" in cat_m
        assert "multi_hop" in cat_m
        assert cat_m["single_hop"]["count"] == 2
        assert cat_m["multi_hop"]["count"] == 1


class TestLatencyPercentiles:
    def test_empty(self):
        r = compute_latency_percentiles([])
        assert r["p50"] == 0.0
        assert r["p95"] == 0.0

    def test_single(self):
        r = compute_latency_percentiles([100.0])
        assert r["p50"] == 100.0
        assert r["p95"] == 100.0

    def test_ordering(self):
        latencies = [10.0, 20.0, 30.0, 100.0, 500.0]
        r = compute_latency_percentiles(latencies)
        assert r["p50"] <= r["p95"]
