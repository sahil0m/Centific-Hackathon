"""Tests for the OpenRouter LLM client.

Uses a deterministic ``FakeBackend`` so tests never touch the network or the
``openai`` SDK. Covers cache hit/miss, retry on transient failures, error
propagation, argument validation, and metric capture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from locomo_memory.phase2.llm.cache import LLMCache
from locomo_memory.phase2.llm.client import (
    LLMResponse,
    OpenRouterClient,
    OpenRouterError,
)


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------


class FakeBackend:
    """Records calls and returns scripted responses (or raises)."""

    def __init__(
        self,
        responses: list[tuple[str, int, int]] | None = None,
        exceptions: list[Exception | None] | None = None,
    ) -> None:
        # If exceptions is provided, exceptions[i] is raised on call i
        # (None means use responses[i] instead).
        self.responses = responses or [("default content", 5, 7)]
        self.exceptions = exceptions or []
        self.calls: list[dict[str, Any]] = []

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
        idx = len(self.calls)
        self.calls.append({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "timeout_s": timeout_s,
        })
        if idx < len(self.exceptions) and self.exceptions[idx] is not None:
            raise self.exceptions[idx]  # type: ignore[misc]
        if idx < len(self.responses):
            return self.responses[idx]
        return self.responses[-1]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults_with_backend(self) -> None:
        client = OpenRouterClient(backend=FakeBackend())
        assert client.max_retries == 3
        assert client.timeout_s == 60.0
        assert client.cache is None

    def test_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError):
            OpenRouterClient(backend=FakeBackend(), max_retries=0)

    def test_invalid_timeout(self) -> None:
        with pytest.raises(ValueError):
            OpenRouterClient(backend=FakeBackend(), timeout_s=0)

    def test_no_api_key_no_backend_raises_only_on_use(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Construction should not raise — we may never call the LLM.
        client = OpenRouterClient()
        # But the first call should fail clearly.
        with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
            client.chat_completion(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                prompt_template_version="v1",
                cache_input="hi",
            )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_empty_model(self) -> None:
        client = OpenRouterClient(backend=FakeBackend())
        with pytest.raises(ValueError):
            client.chat_completion(
                model="",
                messages=[{"role": "user", "content": "x"}],
                prompt_template_version="v1",
                cache_input="x",
            )

    def test_empty_messages(self) -> None:
        client = OpenRouterClient(backend=FakeBackend())
        with pytest.raises(ValueError):
            client.chat_completion(
                model="m", messages=[],
                prompt_template_version="v1", cache_input="x",
            )

    def test_empty_template_version(self) -> None:
        client = OpenRouterClient(backend=FakeBackend())
        with pytest.raises(ValueError):
            client.chat_completion(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                prompt_template_version="",
                cache_input="x",
            )


# ---------------------------------------------------------------------------
# Successful calls
# ---------------------------------------------------------------------------


class TestSuccessfulCall:
    def test_returns_response(self) -> None:
        backend = FakeBackend([("answer", 42, 17)])
        client = OpenRouterClient(backend=backend)
        resp = client.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "ask"}],
            prompt_template_version="v1",
            cache_input="ask",
        )
        assert isinstance(resp, LLMResponse)
        assert resp.content == "answer"
        assert resp.input_tokens == 42
        assert resp.output_tokens == 17
        assert resp.latency_ms >= 0
        assert resp.from_cache is False
        assert resp.model == "m"

    def test_passes_args_to_backend(self) -> None:
        backend = FakeBackend([("ok", 1, 1)])
        client = OpenRouterClient(backend=backend, timeout_s=30.0)
        client.chat_completion(
            model="my-model",
            messages=[{"role": "system", "content": "sys"},
                      {"role": "user", "content": "u"}],
            prompt_template_version="v1",
            cache_input="u",
            temperature=0.5,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        assert backend.calls[0]["model"] == "my-model"
        assert backend.calls[0]["temperature"] == 0.5
        assert backend.calls[0]["max_tokens"] == 200
        assert backend.calls[0]["response_format"] == {"type": "json_object"}
        assert backend.calls[0]["timeout_s"] == 30.0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_hit_avoids_backend(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path / "c")
        backend = FakeBackend([("first", 1, 2)])
        client = OpenRouterClient(backend=backend, cache=cache)

        r1 = client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v1", cache_input="x",
        )
        r2 = client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v1", cache_input="x",
        )
        assert r1.from_cache is False
        assert r2.from_cache is True
        assert r2.content == "first"
        assert r2.input_tokens == 1
        assert r2.output_tokens == 2
        # Backend called only once
        assert len(backend.calls) == 1

    def test_different_input_misses(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path / "c")
        backend = FakeBackend([("a", 1, 1), ("b", 2, 2)])
        client = OpenRouterClient(backend=backend, cache=cache)
        r1 = client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v1", cache_input="x",
        )
        r2 = client.chat_completion(
            model="m", messages=[{"role": "user", "content": "y"}],
            prompt_template_version="v1", cache_input="y",
        )
        assert r1.content == "a"
        assert r2.content == "b"
        assert r2.from_cache is False
        assert len(backend.calls) == 2

    def test_template_version_change_invalidates(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path / "c")
        backend = FakeBackend([("a", 1, 1), ("b", 2, 2)])
        client = OpenRouterClient(backend=backend, cache=cache)
        client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v1", cache_input="x",
        )
        client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v2", cache_input="x",
        )
        assert len(backend.calls) == 2

    def test_no_cache_calls_every_time(self) -> None:
        backend = FakeBackend([("a", 1, 1), ("a", 1, 1)])
        client = OpenRouterClient(backend=backend)  # no cache
        client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v1", cache_input="x",
        )
        client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v1", cache_input="x",
        )
        assert len(backend.calls) == 2


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetry:
    def test_succeeds_after_transient_failures(self) -> None:
        backend = FakeBackend(
            responses=[("", 0, 0), ("", 0, 0), ("recovered", 5, 5)],
            exceptions=[
                ConnectionError("transient 1"),
                ConnectionError("transient 2"),
                None,
            ],
        )
        client = OpenRouterClient(backend=backend, max_retries=3)
        resp = client.chat_completion(
            model="m", messages=[{"role": "user", "content": "x"}],
            prompt_template_version="v1", cache_input="x",
        )
        assert resp.content == "recovered"
        assert len(backend.calls) == 3

    def test_gives_up_after_max_retries(self) -> None:
        backend = FakeBackend(
            exceptions=[ConnectionError("nope")] * 5,
        )
        client = OpenRouterClient(backend=backend, max_retries=2)
        with pytest.raises(OpenRouterError):
            client.chat_completion(
                model="m", messages=[{"role": "user", "content": "x"}],
                prompt_template_version="v1", cache_input="x",
            )
        assert len(backend.calls) == 2

    def test_failed_call_not_cached(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path / "c")
        backend = FakeBackend(
            exceptions=[ConnectionError("nope")] * 10,
        )
        client = OpenRouterClient(backend=backend, cache=cache, max_retries=1)
        with pytest.raises(OpenRouterError):
            client.chat_completion(
                model="m", messages=[{"role": "user", "content": "x"}],
                prompt_template_version="v1", cache_input="x",
            )
        assert len(cache) == 0
