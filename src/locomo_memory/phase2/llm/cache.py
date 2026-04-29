"""Disk-backed cache for LLM responses.

The cache key follows the spec from PHASE2_METHODOLOGY.md §9.1:

    cache_key = sha256(
        model_name + "|" + prompt_template_version + "|" + sha256(input_text)
    )[:16]

This means:
- changing the model invalidates cache,
- bumping ``prompt_template_version`` invalidates cache (use this when you
  change the system/user prompt or any other parameter that affects the
  output, e.g. ``max_facts``),
- changing the input text invalidates cache.

Because keys are deterministic, multiple processes can share the same cache
directory safely (``diskcache`` provides file locking).

The stored value is a JSON-serialised :class:`LLMCacheEntry` so we preserve
content, token counts, and original latency on hit. Callers who want to
distinguish cache hits set ``from_cache=True`` on the response they return.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class LLMCacheEntry(BaseModel):
    """One cached LLM response. Stored as JSON in diskcache."""

    model_config = ConfigDict(validate_assignment=True)

    content: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    model: str
    template_version: str


class LLMCache:
    """Disk-backed LLM response cache with strict key discipline.

    Args:
        cache_dir: directory where ``diskcache`` stores its files. Created
            if missing.
        size_limit_bytes: optional eviction cap in bytes. ``None`` = unbounded.
            Default is None — for benchmark reproducibility we never want
            cache misses caused by silent eviction.

    Usage::

        cache = LLMCache("data/processed/phase2_cache/llm")
        key = LLMCache.make_key(model, "extractor_v1", chunk_text)
        hit = cache.get(key)
        if hit is None:
            response = call_llm(...)
            cache.set(key, LLMCacheEntry(...))
    """

    def __init__(
        self,
        cache_dir: str | Path,
        size_limit_bytes: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if size_limit_bytes is not None:
            kwargs["size_limit"] = int(size_limit_bytes)
        self._cache = diskcache.Cache(str(self.cache_dir), **kwargs)
        logger.debug("Opened LLMCache at {}", self.cache_dir)

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(model_name: str, template_version: str, input_text: str) -> str:
        """Deterministic 16-char hex key per the methodology spec.

        Args:
            model_name: full model id, e.g. ``"meta-llama/llama-3.1-8b-instruct"``.
            template_version: opaque version string. Bump when prompts or
                parameters that affect output change.
            input_text: the user-visible content of the request (typically
                the chunk text or question).
        """
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if not template_version:
            raise ValueError("template_version must be non-empty")
        inner = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        outer = hashlib.sha256(
            f"{model_name}|{template_version}|{inner}".encode("utf-8")
        ).hexdigest()
        return outer[:16]

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> LLMCacheEntry | None:
        raw = self._cache.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            return LLMCacheEntry(**data)
        except Exception as exc:
            # Treat corrupt entries as cache miss — but log loudly.
            logger.warning("Corrupt cache entry for key {}: {}", key, exc)
            self._cache.delete(key)
            return None

    def set(self, key: str, entry: LLMCacheEntry) -> None:
        self._cache.set(key, entry.model_dump_json())

    def has(self, key: str) -> bool:
        return key in self._cache

    def delete(self, key: str) -> bool:
        """Delete a single key. Returns True if present, False otherwise."""
        return bool(self._cache.delete(key))

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def close(self) -> None:
        self._cache.close()


__all__ = ["LLMCache", "LLMCacheEntry"]
