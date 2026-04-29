"""Tests for the LLM disk cache.

Verifies key-construction determinism, isolation across (model,
template_version, input) axes, set/get roundtrip, persistence across
process boundaries, and corruption recovery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from locomo_memory.phase2.llm.cache import LLMCache, LLMCacheEntry


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


class TestKeyConstruction:
    def test_deterministic_for_same_inputs(self) -> None:
        k1 = LLMCache.make_key("m", "v1", "hello")
        k2 = LLMCache.make_key("m", "v1", "hello")
        assert k1 == k2
        assert len(k1) == 16

    def test_different_model_isolates(self) -> None:
        k1 = LLMCache.make_key("m1", "v1", "hello")
        k2 = LLMCache.make_key("m2", "v1", "hello")
        assert k1 != k2

    def test_different_template_version_isolates(self) -> None:
        k1 = LLMCache.make_key("m", "v1", "hello")
        k2 = LLMCache.make_key("m", "v2", "hello")
        assert k1 != k2

    def test_different_input_isolates(self) -> None:
        k1 = LLMCache.make_key("m", "v1", "hello")
        k2 = LLMCache.make_key("m", "v1", "world")
        assert k1 != k2

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValueError):
            LLMCache.make_key("", "v1", "x")

    def test_empty_template_version_rejected(self) -> None:
        with pytest.raises(ValueError):
            LLMCache.make_key("m", "", "x")

    def test_empty_input_allowed(self) -> None:
        # Empty input is a legitimate (if rare) prompt; the hash is still well-defined.
        key = LLMCache.make_key("m", "v1", "")
        assert len(key) == 16


# ---------------------------------------------------------------------------
# Set / get / delete
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> LLMCache:
    return LLMCache(tmp_path / "llm_cache")


def _entry(content: str = "hello world") -> LLMCacheEntry:
    return LLMCacheEntry(
        content=content,
        input_tokens=10,
        output_tokens=20,
        latency_ms=123.4,
        model="m",
        template_version="v1",
    )


class TestRoundtrip:
    def test_set_get(self, cache: LLMCache) -> None:
        key = LLMCache.make_key("m", "v1", "input")
        cache.set(key, _entry("answer"))
        hit = cache.get(key)
        assert hit is not None
        assert hit.content == "answer"
        assert hit.input_tokens == 10
        assert hit.output_tokens == 20
        assert hit.latency_ms == 123.4

    def test_miss_returns_none(self, cache: LLMCache) -> None:
        assert cache.get("nonexistent") is None

    def test_has(self, cache: LLMCache) -> None:
        key = LLMCache.make_key("m", "v1", "x")
        assert cache.has(key) is False
        cache.set(key, _entry())
        assert cache.has(key) is True

    def test_contains_operator(self, cache: LLMCache) -> None:
        key = LLMCache.make_key("m", "v1", "x")
        cache.set(key, _entry())
        assert key in cache

    def test_delete(self, cache: LLMCache) -> None:
        key = LLMCache.make_key("m", "v1", "x")
        cache.set(key, _entry())
        assert cache.delete(key) is True
        assert cache.get(key) is None
        assert cache.delete(key) is False

    def test_clear(self, cache: LLMCache) -> None:
        for i in range(5):
            cache.set(LLMCache.make_key("m", "v1", f"k{i}"), _entry())
        assert len(cache) == 5
        cache.clear()
        assert len(cache) == 0

    def test_len(self, cache: LLMCache) -> None:
        assert len(cache) == 0
        cache.set("a", _entry())
        cache.set("b", _entry())
        assert len(cache) == 2


# ---------------------------------------------------------------------------
# Persistence and corruption recovery
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "llm_cache"
        c1 = LLMCache(path)
        key = LLMCache.make_key("m", "v1", "x")
        c1.set(key, _entry("persistent"))
        c1.close()

        c2 = LLMCache(path)
        hit = c2.get(key)
        assert hit is not None
        assert hit.content == "persistent"
        c2.close()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "llm_cache"
        cache = LLMCache(nested)
        assert nested.is_dir()
        cache.close()


class TestCorruptionRecovery:
    def test_corrupt_entry_treated_as_miss(self, cache: LLMCache) -> None:
        # Store invalid JSON directly via the underlying diskcache
        cache._cache.set("bad-key", "not-valid-json-{")
        # Should return None (and silently delete the bad entry)
        assert cache.get("bad-key") is None
        assert cache.has("bad-key") is False


# ---------------------------------------------------------------------------
# Entry validation
# ---------------------------------------------------------------------------


class TestEntryValidation:
    def test_negative_tokens_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMCacheEntry(
                content="x", input_tokens=-1, output_tokens=0,
                latency_ms=0.0, model="m", template_version="v1",
            )

    def test_negative_latency_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMCacheEntry(
                content="x", input_tokens=0, output_tokens=0,
                latency_ms=-1.0, model="m", template_version="v1",
            )
