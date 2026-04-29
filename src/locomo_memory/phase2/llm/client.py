"""OpenRouter LLM client with retries, caching, and a swappable backend.

Design
------
The class hierarchy intentionally splits two concerns:

- :class:`LLMBackend` (Protocol): the raw "send a chat completion to a
  remote service" capability. Defines a tiny interface so tests can plug
  in deterministic fakes.
- :class:`OpenAIBackend`: the production implementation, using the
  ``openai`` SDK pointed at OpenRouter's OpenAI-compatible endpoint.
- :class:`OpenRouterClient`: orchestrates caching (via :class:`LLMCache`),
  retry policy (via tenacity), and request shaping. Callers always go
  through this class.

This makes the unit tests cheap (no network, no SDK mocking) while keeping
the production path simple.

Cache & retry semantics
-----------------------
- Cache key follows the spec in PHASE2_METHODOLOGY.md §9.1.
- A cache hit short-circuits both the HTTP call and the retry loop.
- Retries use exponential backoff: 0.5s, 1s, 2s, ... up to ``max_retries``.
- On final failure :class:`OpenRouterError` is raised — callers decide
  whether to fall back, re-raise, or skip the work.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from locomo_memory.phase2.llm.cache import LLMCache, LLMCacheEntry


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails after all retries.

    Carries the underlying cause as ``__cause__`` (use ``raise X from Y``
    pattern in callers).
    """


# ---------------------------------------------------------------------------
# Response value object
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMResponse:
    """Result of one chat completion (cached or freshly computed)."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    from_cache: bool


# ---------------------------------------------------------------------------
# Backend Protocol + production implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface a chat completions backend must satisfy."""

    def call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        timeout_s: float,
    ) -> tuple[str, int, int]:  # (content, input_tokens, output_tokens)
        ...  # pragma: no cover — protocol


class OpenAIBackend:
    """Production backend: OpenAI Python SDK pointed at OpenRouter."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        http_referer: str | None = None,
        app_name: str | None = None,
    ) -> None:
        # Imported lazily so unit tests do not depend on the openai package
        # being importable, and so the import cost is paid only on first use.
        from openai import OpenAI  # type: ignore[import-untyped]

        default_headers: dict[str, str] = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_name:
            default_headers["X-Title"] = app_name

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )

    def call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        timeout_s: float,
    ) -> tuple[str, int, int]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        if not response.choices:
            raise OpenRouterError("OpenRouter returned no choices")
        content = response.choices[0].message.content or ""
        usage = response.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        return content, in_tok, out_tok


# ---------------------------------------------------------------------------
# OpenRouterClient
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Cached, retrying chat-completion client.

    Args:
        api_key: OpenRouter API key. If None, read from ``OPENROUTER_API_KEY``
            environment variable. Required only when ``backend`` is None.
        base_url: OpenAI-compatible base URL. Default is OpenRouter's.
        cache: optional :class:`LLMCache`. If None, every call hits the
            backend.
        max_retries: number of attempts (1 = no retry).
        timeout_s: per-call HTTP timeout.
        http_referer: optional ``HTTP-Referer`` header for OpenRouter
            analytics. Ignored unless using :class:`OpenAIBackend`.
        app_name: optional ``X-Title`` header. Ignored unless using
            :class:`OpenAIBackend`.
        backend: a custom :class:`LLMBackend`. Test code passes a fake.
            If omitted, :class:`OpenAIBackend` is constructed lazily.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OpenAIBackend.DEFAULT_BASE_URL,
        cache: LLMCache | None = None,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        http_referer: str | None = None,
        app_name: str | None = None,
        backend: LLMBackend | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")

        self.cache = cache
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._backend: LLMBackend | None = backend
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._base_url = base_url
        self._http_referer = http_referer
        self._app_name = app_name

    # ------------------------------------------------------------------
    # Lazy backend
    # ------------------------------------------------------------------

    def _get_backend(self) -> LLMBackend:
        if self._backend is None:
            if not self._api_key:
                raise OpenRouterError(
                    "OPENROUTER_API_KEY not set. Pass api_key=... or set the "
                    "OPENROUTER_API_KEY environment variable."
                )
            self._backend = OpenAIBackend(
                api_key=self._api_key,
                base_url=self._base_url,
                http_referer=self._http_referer,
                app_name=self._app_name,
            )
        return self._backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        prompt_template_version: str,
        cache_input: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Run a chat completion, with cache and retries.

        Args:
            model: full model id (e.g. ``"meta-llama/llama-3.1-8b-instruct"``).
            messages: list of ``{"role": ..., "content": ...}`` dicts.
            prompt_template_version: bump this whenever prompts or non-input
                parameters that affect output change. Used in the cache key.
            cache_input: the input portion of the cache key — typically the
                user message content (chunk text, question, etc.).
            temperature: sampling temperature (default 0.0 for determinism).
            max_tokens: max output tokens.
            response_format: optional, e.g. ``{"type": "json_object"}``.

        Returns:
            :class:`LLMResponse` with content and usage metrics. ``from_cache``
            is True iff the result came from :class:`LLMCache`.

        Raises:
            OpenRouterError: if the backend fails after ``max_retries``.
            ValueError: on invalid arguments.
        """
        if not model:
            raise ValueError("model must be non-empty")
        if not messages:
            raise ValueError("messages must be non-empty")
        if not prompt_template_version:
            raise ValueError("prompt_template_version must be non-empty")

        # Cache lookup
        if self.cache is not None:
            key = LLMCache.make_key(model, prompt_template_version, cache_input)
            hit = self.cache.get(key)
            if hit is not None:
                logger.debug("LLM cache hit: model={} key={}", model, key)
                return LLMResponse(
                    content=hit.content,
                    input_tokens=hit.input_tokens,
                    output_tokens=hit.output_tokens,
                    latency_ms=hit.latency_ms,
                    model=hit.model,
                    from_cache=True,
                )

        # Cache miss — call backend with retry
        backend = self._get_backend()

        def _attempt() -> tuple[str, int, int, float]:
            t0 = time.perf_counter()
            content, in_tok, out_tok = backend.call(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout_s=self.timeout_s,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return content, in_tok, out_tok, latency_ms

        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    content, in_tok, out_tok, latency_ms = _attempt()
        except RetryError as exc:  # pragma: no cover — reraise=True bypasses this
            raise OpenRouterError(f"LLM call failed after retries: {exc}") from exc
        except Exception as exc:
            raise OpenRouterError(f"LLM call failed: {exc}") from exc

        # Cache write (only on success)
        if self.cache is not None:
            key = LLMCache.make_key(model, prompt_template_version, cache_input)
            self.cache.set(
                key,
                LLMCacheEntry(
                    content=content,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                    model=model,
                    template_version=prompt_template_version,
                ),
            )

        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            model=model,
            from_cache=False,
        )


__all__ = [
    "LLMBackend",
    "LLMResponse",
    "OpenAIBackend",
    "OpenRouterClient",
    "OpenRouterError",
]
