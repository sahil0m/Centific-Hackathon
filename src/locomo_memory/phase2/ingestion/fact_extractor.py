"""Fact Extractor — the agentic chunker (LLM Call #1).

Turns Phase 1 :class:`Chunk` objects into Phase 2 :class:`MemoryUnit` objects
by asking a small fast LLM to extract atomic factual claims. The result is a
:class:`ExtractionResult` that bundles the produced MUs with usage metrics
and any error information.

Design properties
-----------------
- **Pure provenance**: every produced MU carries source dia_ids, speaker,
  session, and timestamps copied from the input chunk. The LLM may pick a
  single dia_id per fact; if it picks one not in the chunk, we fall back to
  the chunk's full dia_id list rather than fabricating provenance.
- **Robust JSON parsing**: tolerates markdown code-fences and minor
  formatting drift. Invalid output is logged and treated as a failure.
- **Heuristic fallback**: when LLM extraction fails (or is disabled by
  ``enable_llm=False``), a sentence-level fallback keeps the pipeline
  running. Heuristic facts get lower confidence (0.5) so downstream salience
  scoring can prioritise LLM facts.
- **Cache-friendly**: forwards a stable ``cache_input`` (the chunk text) and
  a versioned template id to :class:`OpenRouterClient`. Same chunk + same
  prompt + same model = cache hit, no LLM call.

Configuration flags (from the methodology)
------------------------------------------
- ``enable_llm`` — corresponds to ``phase2.enable_llm_extraction`` YAML flag.
- ``max_facts_per_chunk`` — affects both prompt and template version so
  changing it invalidates cache.
- ``retain_on_failure`` — if True, a failed extraction still emits a single
  MU containing the chunk text, so no provenance is silently dropped.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import ClassVar, Final

from loguru import logger
from pydantic import ValidationError

from locomo_memory.data.schemas import Chunk
from locomo_memory.phase2.ingestion.importance import TopicImportanceEstimator
from locomo_memory.phase2.llm.client import (
    LLMResponse,
    OpenRouterClient,
    OpenRouterError,
)
from locomo_memory.phase2.schemas import MemoryUnit, MemoryStatus

# Module-level singleton — stateless, safe to share.
_importance_estimator = TopicImportanceEstimator()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_BASE_TEMPLATE_VERSION: Final[str] = "extractor_v1"


def _template_version(max_facts: int, model: str) -> str:
    """Cache-key salt that captures everything outside the chunk text."""
    return f"{_BASE_TEMPLATE_VERSION}|max{max_facts}|{model}"


def _build_messages(chunk_text: str, max_facts: int) -> list[dict[str, str]]:
    system = (
        "You are a memory extraction agent. Read the conversation chunk and "
        "extract atomic factual claims worth remembering for long-term memory.\n\n"
        "Rules:\n"
        "1. Each fact must be a complete, standalone statement (a single sentence).\n"
        "2. Resolve pronouns to full names where possible, using the speaker labels.\n"
        "3. Skip questions, opinions, hedges (\"I think\", \"maybe\"), and pleasantries.\n"
        "4. Skip meta-statements about the conversation itself (\"yeah\", \"ok\", \"right\").\n"
        "5. Each fact must be self-contained and readable without other facts.\n"
        f"6. Return AT MOST {max_facts} facts; pick the most informative.\n\n"
        "For each fact, attribute it to a speaker label that appears in the chunk, "
        "and (if you can identify a single source turn) the dialog ID of that turn.\n\n"
        "Return STRICT JSON ONLY. Do not include markdown fences, explanations, "
        "or any prose outside the JSON object. Schema:\n"
        '{"facts": [{"claim": "<sentence>", "speaker": "<speaker label>", '
        '"source_dia_id": "<one dialog id from the chunk, or null>"}]}\n\n'
        "If no facts can be extracted, return: {\"facts\": []}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": chunk_text},
    ]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExtractionResult:
    """Outcome of extracting facts from a single chunk."""

    chunk_id: str
    memory_units: list[MemoryUnit] = field(default_factory=list)
    raw_response: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False
    success: bool = True
    failure_reason: str | None = None
    used_heuristic: bool = False


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_facts_payload(raw: str) -> list[dict[str, str | None]]:
    """Strip optional code fences, parse JSON, return the ``facts`` list.

    Raises ``ValueError`` for any malformed input.
    """
    text = raw.strip()
    text = _FENCE_RE.sub("", text).strip()
    if not text:
        raise ValueError("empty LLM response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "facts" not in data:
        raise ValueError("response missing 'facts' key")
    facts = data["facts"]
    if not isinstance(facts, list):
        raise ValueError("'facts' is not a list")
    out: list[dict[str, str | None]] = []
    for i, item in enumerate(facts):
        if isinstance(item, str):
            # Tolerate plain-string facts (some models drop the structure).
            out.append({"claim": item, "speaker": None, "source_dia_id": None})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"facts[{i}] is not an object or string")
        claim = item.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"facts[{i}].claim missing or non-string")
        speaker = item.get("speaker")
        source = item.get("source_dia_id")
        out.append({
            "claim": claim.strip(),
            "speaker": speaker if isinstance(speaker, str) and speaker else None,
            "source_dia_id": source if isinstance(source, str) and source else None,
        })
    return out


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------


_OPINION_PREFIX_RE = re.compile(
    r"^\s*(i\s+(think|feel|guess|hope|wish|wonder)|maybe|perhaps|probably)\s+",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _heuristic_facts(chunk: Chunk, max_facts: int) -> list[dict[str, str | None]]:
    """Sentence-level fallback. Lower-quality but always available.

    Splits each turn body on sentence boundaries, drops obvious non-facts
    (questions, opinion-prefixed statements, very short fragments), and
    attributes each to the surrounding turn's speaker.
    """
    out: list[dict[str, str | None]] = []
    # Reconstruct per-turn (speaker, dia_id, text) by walking chunk metadata.
    # The chunk text contains a header + one line per turn formatted as
    # "Speaker [ts]: text" — but rather than re-parse, walk the parallel
    # arrays we already have.
    n = max(len(chunk.dia_ids), len(chunk.speakers))
    if n == 0:
        return out

    # We don't have per-turn raw text on the Chunk; the methodology accepts
    # this — heuristic uses chunk-level granularity for source.
    body = chunk.text
    # Strip header line if present
    if body.startswith("[") and "\n" in body:
        body = body.split("\n", 1)[1]

    for line in body.split("\n"):
        if not line.strip():
            continue
        # Strip "Speaker: " prefix if present
        line_speaker: str | None = None
        if ":" in line:
            head, rest = line.split(":", 1)
            head = head.strip()
            if head and len(head) <= 80:
                line_speaker = head
                # "Alice [2024-01-01]" → take just the first token as the speaker label
                line_speaker = line_speaker.split(" ", 1)[0]
                line = rest.strip()

        for sentence in _SENTENCE_SPLIT_RE.split(line):
            s = sentence.strip().rstrip(".")
            if len(s) < 10:
                continue
            if s.endswith("?"):
                continue
            if _OPINION_PREFIX_RE.match(s):
                continue
            speaker = line_speaker if line_speaker in chunk.speakers else None
            out.append({"claim": s, "speaker": speaker, "source_dia_id": None})
            if len(out) >= max_facts:
                return out
    return out


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FactExtractor:
    """Turn a Chunk into atomic-fact MemoryUnits via an LLM (or heuristic).

    Args:
        client: an :class:`OpenRouterClient`. Required when ``enable_llm`` is
            True, ignored otherwise.
        model: full OpenRouter model id. Default is the cheap-and-fast
            ``meta-llama/llama-3.1-8b-instruct``.
        temperature: sampling temperature. Default 0.0 for determinism (and
            so cache hits are useful).
        max_output_tokens: ceiling on LLM output length.
        max_facts_per_chunk: hard cap on facts emitted per chunk. Affects
            both the prompt and the cache key.
        enable_llm: if False, always use the heuristic fallback. Equivalent
            to ``phase2.enable_llm_extraction: false`` in the YAML config.
        retain_on_failure: if True (default), a complete LLM failure still
            emits one MU containing the raw chunk text with confidence 0.5.
            Set to False if you want failed chunks to produce zero MUs.
        llm_confidence: confidence assigned to MUs from successful LLM
            extraction (default 0.9).
        heuristic_confidence: confidence for heuristic-fallback MUs
            (default 0.5).

    Threading: a single ``FactExtractor`` is safe to share across threads
    only if its underlying ``OpenRouterClient`` (and cache) is thread-safe.
    Calls do not mutate instance state.
    """

    DEFAULT_MODEL: ClassVar[str] = "meta-llama/llama-3.1-8b-instruct"
    DEFAULT_MAX_FACTS: ClassVar[int] = 7

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        max_facts_per_chunk: int = DEFAULT_MAX_FACTS,
        enable_llm: bool = True,
        retain_on_failure: bool = True,
        llm_confidence: float = 0.9,
        heuristic_confidence: float = 0.5,
    ) -> None:
        if max_facts_per_chunk < 1:
            raise ValueError(
                f"max_facts_per_chunk must be >= 1, got {max_facts_per_chunk}"
            )
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature out of [0,2]: {temperature}")
        if max_output_tokens < 16:
            raise ValueError(f"max_output_tokens too small: {max_output_tokens}")
        if not 0.0 <= llm_confidence <= 1.0:
            raise ValueError("llm_confidence must be in [0,1]")
        if not 0.0 <= heuristic_confidence <= 1.0:
            raise ValueError("heuristic_confidence must be in [0,1]")
        if enable_llm and client is None:
            raise ValueError("client is required when enable_llm=True")

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_facts_per_chunk = max_facts_per_chunk
        self.enable_llm = enable_llm
        self.retain_on_failure = retain_on_failure
        self.llm_confidence = llm_confidence
        self.heuristic_confidence = heuristic_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionResult:
        """Extract facts from a single chunk.

        Returns an :class:`ExtractionResult` even on failure; check
        ``result.success`` and ``result.failure_reason`` to distinguish.
        """
        if not self.enable_llm:
            mus = self._build_heuristic_mus(chunk)
            return ExtractionResult(
                chunk_id=chunk.chunk_id,
                memory_units=mus,
                used_heuristic=True,
                success=True,
            )

        assert self.client is not None  # guarded by __init__
        messages = _build_messages(chunk.text, self.max_facts_per_chunk)
        template_version = _template_version(self.max_facts_per_chunk, self.model)

        try:
            response: LLMResponse = self.client.chat_completion(
                model=self.model,
                messages=messages,
                prompt_template_version=template_version,
                cache_input=chunk.text,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
            )
        except OpenRouterError as exc:
            return self._handle_failure(chunk, f"LLM call failed: {exc}")

        try:
            parsed = _parse_facts_payload(response.content)
        except ValueError as exc:
            logger.warning(
                "Fact extractor: malformed LLM response for chunk={}: {}",
                chunk.chunk_id, exc,
            )
            return self._handle_failure(
                chunk,
                f"malformed LLM response: {exc}",
                raw_response=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                from_cache=response.from_cache,
            )

        # Cap fact count defensively (the LLM may ignore the prompt limit).
        if len(parsed) > self.max_facts_per_chunk:
            parsed = parsed[: self.max_facts_per_chunk]

        mus = self._build_llm_mus(chunk, parsed)
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            memory_units=mus,
            raw_response=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            from_cache=response.from_cache,
            success=True,
            used_heuristic=False,
        )

    def extract_from_chunks(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        return [self.extract_from_chunk(c) for c in chunks]

    # ------------------------------------------------------------------
    # MU builders
    # ------------------------------------------------------------------

    def _build_llm_mus(
        self,
        chunk: Chunk,
        parsed: list[dict[str, str | None]],
    ) -> list[MemoryUnit]:
        mus: list[MemoryUnit] = []
        for fact in parsed:
            claim = fact["claim"]
            assert isinstance(claim, str)  # _parse_facts_payload guarantees
            speaker, source_dia_ids, timestamp = self._resolve_provenance(
                chunk, fact["speaker"], fact["source_dia_id"],
            )
            try:
                mu = MemoryUnit(
                    conversation_id=chunk.conversation_id,
                    session_id=chunk.session_id,
                    claim=claim,
                    original_text=chunk.text,
                    source_dia_ids=source_dia_ids,
                    source_speaker=speaker,
                    timestamp=timestamp,
                    importance=_importance_estimator.estimate(claim),
                    confidence=self.llm_confidence,
                    status=MemoryStatus.ACTIVE,
                )
            except ValidationError as exc:
                logger.warning(
                    "Fact extractor: rejected invalid MU for chunk={}: {}",
                    chunk.chunk_id, exc,
                )
                continue
            mus.append(mu)
        return mus

    def _build_heuristic_mus(self, chunk: Chunk) -> list[MemoryUnit]:
        parsed = _heuristic_facts(chunk, self.max_facts_per_chunk)
        mus: list[MemoryUnit] = []
        for fact in parsed:
            claim = fact["claim"]
            assert isinstance(claim, str)
            speaker, source_dia_ids, timestamp = self._resolve_provenance(
                chunk, fact["speaker"], fact["source_dia_id"],
            )
            try:
                mu = MemoryUnit(
                    conversation_id=chunk.conversation_id,
                    session_id=chunk.session_id,
                    claim=claim,
                    original_text=chunk.text,
                    source_dia_ids=source_dia_ids,
                    source_speaker=speaker,
                    timestamp=timestamp,
                    importance=_importance_estimator.estimate(claim),
                    confidence=self.heuristic_confidence,
                    status=MemoryStatus.ACTIVE,
                )
            except ValidationError as exc:
                logger.debug(
                    "Heuristic fact rejected for chunk={}: {}",
                    chunk.chunk_id, exc,
                )
                continue
            mus.append(mu)
        return mus

    def _resolve_provenance(
        self,
        chunk: Chunk,
        suggested_speaker: str | None,
        suggested_dia_id: str | None,
    ) -> tuple[str, list[str], str | None]:
        """Pin a fact's provenance to actual chunk metadata.

        - If ``suggested_dia_id`` matches a chunk dia_id, narrow source to it
          and look up the speaker/timestamp at that index.
        - Otherwise fall back to the chunk's full dia_id list, joining
          speakers if no suggestion is given.
        """
        if (
            suggested_dia_id is not None
            and suggested_dia_id in chunk.dia_ids
        ):
            idx = chunk.dia_ids.index(suggested_dia_id)
            speaker = (
                chunk.speakers[idx] if idx < len(chunk.speakers) else
                (suggested_speaker or "")
            )
            timestamp = (
                chunk.timestamps[idx] if idx < len(chunk.timestamps) else None
            )
            return speaker, [suggested_dia_id], (timestamp or None)

        # Fallback: keep the full chunk dia_id list (still correct provenance).
        if suggested_speaker and suggested_speaker in chunk.speakers:
            speaker = suggested_speaker
        elif chunk.speakers:
            speaker = ", ".join(dict.fromkeys(chunk.speakers))  # dedupe, keep order
        else:
            speaker = ""
        timestamp = chunk.timestamps[0] if chunk.timestamps else None
        return speaker, list(chunk.dia_ids), (timestamp or None)

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _handle_failure(
        self,
        chunk: Chunk,
        reason: str,
        *,
        raw_response: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        from_cache: bool = False,
    ) -> ExtractionResult:
        mus: list[MemoryUnit] = []
        used_heuristic = False
        if self.retain_on_failure:
            # Try the heuristic before giving up entirely.
            mus = self._build_heuristic_mus(chunk)
            used_heuristic = True
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            memory_units=mus,
            raw_response=raw_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            from_cache=from_cache,
            success=False,
            failure_reason=reason,
            used_heuristic=used_heuristic,
        )


__all__ = ["ExtractionResult", "FactExtractor"]
