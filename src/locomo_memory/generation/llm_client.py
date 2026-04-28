"""
LLM client with:
- configurable provider (anthropic, openai, ollama)
- temperature=0 by default
- exponential backoff retries via tenacity
- disk-based response cache (keyed on prompt hash)
- generation latency tracking
- approximate token usage
- no hardcoded API keys (reads from .env)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    generation_latency_ms: float
    cache_hit: bool = False


class LLMClient:
    def __init__(
        self,
        provider: str = "anthropic",
        model_name: str = "claude-3-5-sonnet-latest",
        temperature: float = 0.0,
        max_output_tokens: int = 120,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self._client = None

    def generate(self, prompt: str) -> GenerationResult:
        cached = self._load_cache(prompt)
        if cached is not None:
            logger.debug("LLM cache hit for prompt hash %s", self._prompt_key(prompt))
            return cached

        result = self._generate_with_retry(prompt)
        self._save_cache(prompt, result)
        return result

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _generate_with_retry(self, prompt: str) -> GenerationResult:
        from tenacity import (  # type: ignore
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> GenerationResult:
            if self.provider == "anthropic":
                return self._call_anthropic(prompt)
            if self.provider in ("openai", "ollama"):
                return self._call_openai_compatible(prompt)
            raise ValueError(f"Unknown LLM provider: '{self.provider}'")

        return _call()

    def _call_anthropic(self, prompt: str) -> GenerationResult:
        import anthropic  # type: ignore

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=api_key)

        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_latency_ms=latency_ms,
        )

    def _call_openai_compatible(self, prompt: str) -> GenerationResult:
        import openai  # type: ignore

        if self.provider == "ollama":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            api_key = "ollama"
        else:
            base_url = None
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
                )

        if self._client is None:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = response.choices[0].message.content.strip()
        usage = response.usage
        return GenerationResult(
            answer=answer,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            generation_latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _prompt_key(self, prompt: str) -> str:
        return hashlib.sha256(
            f"{self.provider}||{self.model_name}||{self.temperature}||{prompt}".encode()
        ).hexdigest()

    def _cache_path(self, prompt: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = self._prompt_key(prompt)
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, prompt: str) -> GenerationResult | None:
        path = self._cache_path(prompt)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return GenerationResult(
                answer=data["answer"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                generation_latency_ms=data["generation_latency_ms"],
                cache_hit=True,
            )
        except Exception:
            return None

    def _save_cache(self, prompt: str, result: GenerationResult) -> None:
        path = self._cache_path(prompt)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps(
                    {
                        "answer": result.answer,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "generation_latency_ms": result.generation_latency_ms,
                    }
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save LLM cache: %s", exc)
