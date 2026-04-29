"""LLM access layer for Phase 2.

Two responsibilities, sharply separated:

- :class:`LLMCache` — disk-backed cache keyed by ``(model, template_version, input_hash)``
  per the cache-key spec in PHASE2_METHODOLOGY.md §9.1.
- :class:`OpenRouterClient` — OpenAI-compatible client that talks to OpenRouter
  through a swappable :class:`LLMBackend`. Retry, caching, and request shaping
  live here so callers (Fact Extractor, Contradiction Resolver, Answer
  Generator) only see a clean ``chat_completion`` method.

The backend is a Protocol so tests can substitute deterministic fakes without
touching the network.
"""

from __future__ import annotations

from locomo_memory.phase2.llm.cache import LLMCache
from locomo_memory.phase2.llm.client import (
    LLMBackend,
    LLMResponse,
    OpenAIBackend,
    OpenRouterClient,
    OpenRouterError,
)

__all__ = [
    "LLMBackend",
    "LLMCache",
    "LLMResponse",
    "OpenAIBackend",
    "OpenRouterClient",
    "OpenRouterError",
]
