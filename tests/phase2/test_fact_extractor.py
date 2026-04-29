"""Tests for the Fact Extractor (LLM Call #1).

Uses a stub :class:`OpenRouterClient` constructed against a deterministic
:class:`FakeBackend` so no network access is needed. Covers happy-path LLM
extraction, JSON shape variants, malformed responses, retain-on-failure
behaviour, the heuristic fallback, provenance resolution, max-fact capping,
and constructor validation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from locomo_memory.data.schemas import Chunk
from locomo_memory.phase2.ingestion.fact_extractor import (
    ExtractionResult,
    FactExtractor,
)
from locomo_memory.phase2.llm.client import (
    OpenRouterClient,
    OpenRouterError,
)
from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBackend:
    """Records calls, returns scripted ``(content, in_tok, out_tok)`` tuples."""

    def __init__(self, responses: list[tuple[str, int, int]]) -> None:
        self.responses = list(responses)
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
        self.calls.append({"model": model, "messages": messages})
        if not self.responses:
            return ("default", 0, 0)
        return self.responses.pop(0)


def _client(backend: FakeBackend) -> OpenRouterClient:
    return OpenRouterClient(backend=backend, max_retries=1)


def _make_chunk(
    *,
    chunk_id: str = "conv_1#semantic#0-1",
    conversation_id: str = "conv_1",
    sample_id: str = "sample_1",
    session_id: str = "session_1",
    dia_ids: list[str] | None = None,
    speakers: list[str] | None = None,
    timestamps: list[str] | None = None,
    text: str | None = None,
) -> Chunk:
    dia_ids = dia_ids or ["D1:1", "D1:2"]
    speakers = speakers or ["Caroline", "Caroline"]
    timestamps = timestamps or ["2024-03-15", "2024-03-15"]
    text = text or (
        "[Conversation: conv_1 | Session: session_1 | Dialog IDs: D1:1,D1:2]\n"
        "Caroline [2024-03-15]: I quit my job at Google.\n"
        "Caroline [2024-03-15]: Starting at Microsoft on Monday."
    )
    return Chunk(
        chunk_id=chunk_id,
        conversation_id=conversation_id,
        sample_id=sample_id,
        session_id=session_id,
        turn_index_start=0,
        turn_index_end=len(dia_ids) - 1,
        dia_ids=dia_ids,
        speakers=speakers,
        timestamps=timestamps,
        text=text,
        chunk_strategy="semantic",
    )


def _llm_response(facts: list[dict[str, str | None]]) -> tuple[str, int, int]:
    return (json.dumps({"facts": facts}), 100, 50)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default(self) -> None:
        backend = FakeBackend([])
        ex = FactExtractor(_client(backend))
        assert ex.max_facts_per_chunk == 7
        assert ex.enable_llm is True

    def test_no_client_when_llm_enabled_raises(self) -> None:
        with pytest.raises(ValueError, match="client is required"):
            FactExtractor(client=None, enable_llm=True)

    def test_no_client_okay_when_llm_disabled(self) -> None:
        ex = FactExtractor(client=None, enable_llm=False)
        assert ex.client is None

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_max_facts(self, bad: int) -> None:
        with pytest.raises(ValueError):
            FactExtractor(_client(FakeBackend([])), max_facts_per_chunk=bad)

    @pytest.mark.parametrize("bad", [-0.1, 2.1])
    def test_invalid_temperature(self, bad: float) -> None:
        with pytest.raises(ValueError):
            FactExtractor(_client(FakeBackend([])), temperature=bad)

    def test_invalid_max_output_tokens(self) -> None:
        with pytest.raises(ValueError):
            FactExtractor(_client(FakeBackend([])), max_output_tokens=4)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_confidences(self, bad: float) -> None:
        with pytest.raises(ValueError):
            FactExtractor(_client(FakeBackend([])), llm_confidence=bad)
        with pytest.raises(ValueError):
            FactExtractor(_client(FakeBackend([])), heuristic_confidence=bad)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_basic_extraction(self) -> None:
        backend = FakeBackend([
            _llm_response([
                {"claim": "Caroline left Google", "speaker": "Caroline",
                 "source_dia_id": "D1:1"},
                {"claim": "Caroline starts at Microsoft", "speaker": "Caroline",
                 "source_dia_id": "D1:2"},
            ])
        ])
        ex = FactExtractor(_client(backend))
        chunk = _make_chunk()

        result = ex.extract_from_chunk(chunk)

        assert result.success is True
        assert result.failure_reason is None
        assert result.used_heuristic is False
        assert len(result.memory_units) == 2

        m1, m2 = result.memory_units
        assert isinstance(m1, MemoryUnit)
        assert m1.claim == "Caroline left Google"
        assert m1.source_dia_ids == ["D1:1"]
        assert m1.source_speaker == "Caroline"
        assert m1.timestamp == "2024-03-15"
        assert m1.confidence == 0.9
        assert m1.status == MemoryStatus.ACTIVE

        assert m2.claim == "Caroline starts at Microsoft"
        assert m2.source_dia_ids == ["D1:2"]

    def test_empty_facts_array(self) -> None:
        backend = FakeBackend([_llm_response([])])
        ex = FactExtractor(_client(backend))
        result = ex.extract_from_chunk(_make_chunk())
        assert result.success is True
        assert result.memory_units == []

    def test_max_facts_capped(self) -> None:
        backend = FakeBackend([
            _llm_response([
                {"claim": f"fact {i}", "speaker": "Caroline", "source_dia_id": "D1:1"}
                for i in range(20)
            ])
        ])
        ex = FactExtractor(_client(backend), max_facts_per_chunk=3)
        result = ex.extract_from_chunk(_make_chunk())
        assert len(result.memory_units) == 3

    def test_returns_token_usage_and_latency(self) -> None:
        backend = FakeBackend([_llm_response([
            {"claim": "x", "speaker": "Caroline", "source_dia_id": "D1:1"},
        ])])
        ex = FactExtractor(_client(backend))
        result = ex.extract_from_chunk(_make_chunk())
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# JSON shape robustness
# ---------------------------------------------------------------------------


class TestJsonRobustness:
    def test_handles_markdown_code_fence(self) -> None:
        raw = '```json\n{"facts": [{"claim": "x", "speaker": "Caroline", "source_dia_id": "D1:1"}]}\n```'
        backend = FakeBackend([(raw, 1, 1)])
        ex = FactExtractor(_client(backend))
        result = ex.extract_from_chunk(_make_chunk())
        assert result.success is True
        assert len(result.memory_units) == 1

    def test_handles_plain_string_facts(self) -> None:
        # Some models emit ["fact1", "fact2"] instead of objects
        raw = json.dumps({"facts": ["Caroline left Google", "She joined Microsoft"]})
        backend = FakeBackend([(raw, 1, 1)])
        ex = FactExtractor(_client(backend))
        result = ex.extract_from_chunk(_make_chunk())
        assert len(result.memory_units) == 2
        # No source_dia_id was given — provenance falls back to full chunk dia_ids
        assert result.memory_units[0].source_dia_ids == ["D1:1", "D1:2"]


# ---------------------------------------------------------------------------
# Malformed responses
# ---------------------------------------------------------------------------


class TestMalformedResponse:
    def test_invalid_json_falls_back(self) -> None:
        backend = FakeBackend([("not json at all", 1, 1)])
        ex = FactExtractor(_client(backend), retain_on_failure=True)
        result = ex.extract_from_chunk(_make_chunk())
        assert result.success is False
        assert result.failure_reason is not None
        assert result.used_heuristic is True
        # Heuristic produced at least one MU from chunk text
        assert len(result.memory_units) > 0
        assert all(m.confidence == 0.5 for m in result.memory_units)

    def test_missing_facts_key_falls_back(self) -> None:
        backend = FakeBackend([(json.dumps({"items": []}), 1, 1)])
        ex = FactExtractor(_client(backend), retain_on_failure=True)
        result = ex.extract_from_chunk(_make_chunk())
        assert result.success is False
        assert result.used_heuristic is True

    def test_facts_not_a_list_falls_back(self) -> None:
        backend = FakeBackend([(json.dumps({"facts": "not a list"}), 1, 1)])
        ex = FactExtractor(_client(backend), retain_on_failure=True)
        result = ex.extract_from_chunk(_make_chunk())
        assert result.success is False

    def test_retain_off_returns_no_mus(self) -> None:
        backend = FakeBackend([("garbage", 1, 1)])
        ex = FactExtractor(_client(backend), retain_on_failure=False)
        result = ex.extract_from_chunk(_make_chunk())
        assert result.success is False
        assert result.memory_units == []
        assert result.used_heuristic is False


# ---------------------------------------------------------------------------
# LLM call failures
# ---------------------------------------------------------------------------


class TestLlmCallFailure:
    def test_openrouter_error_triggers_fallback(self) -> None:
        # Configure backend to always raise — client gives up after max_retries
        class AlwaysFailingBackend:
            def __init__(self) -> None:
                self.calls: list[Any] = []

            def call(self, **_: Any) -> tuple[str, int, int]:
                self.calls.append(_)
                raise ConnectionError("network down")

        client = OpenRouterClient(backend=AlwaysFailingBackend(), max_retries=1)
        ex = FactExtractor(client, retain_on_failure=True)
        result = ex.extract_from_chunk(_make_chunk())
        assert result.success is False
        assert "LLM call failed" in (result.failure_reason or "")
        assert result.used_heuristic is True


# ---------------------------------------------------------------------------
# Provenance resolution
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_invalid_dia_id_falls_back_to_full_list(self) -> None:
        backend = FakeBackend([_llm_response([
            {"claim": "x", "speaker": "Caroline", "source_dia_id": "D9:99"},
        ])])
        ex = FactExtractor(_client(backend))
        result = ex.extract_from_chunk(_make_chunk())
        mu = result.memory_units[0]
        assert mu.source_dia_ids == ["D1:1", "D1:2"]

    def test_invalid_speaker_falls_back_to_chunk_speakers(self) -> None:
        backend = FakeBackend([_llm_response([
            {"claim": "x", "speaker": "Bob", "source_dia_id": None},
        ])])
        ex = FactExtractor(_client(backend))
        chunk = _make_chunk(speakers=["Alice", "Bob_typo"])
        # "Bob" not in ["Alice", "Bob_typo"] → fall back to joined chunk speakers
        result = ex.extract_from_chunk(chunk)
        mu = result.memory_units[0]
        assert "Alice" in mu.source_speaker
        assert "Bob_typo" in mu.source_speaker

    def test_valid_speaker_kept(self) -> None:
        backend = FakeBackend([_llm_response([
            {"claim": "x", "speaker": "Alice", "source_dia_id": None},
        ])])
        chunk = _make_chunk(speakers=["Alice", "Bob"])
        ex = FactExtractor(_client(backend))
        result = ex.extract_from_chunk(chunk)
        assert result.memory_units[0].source_speaker == "Alice"

    def test_valid_dia_id_narrows_provenance(self) -> None:
        backend = FakeBackend([_llm_response([
            {"claim": "x", "speaker": "Alice", "source_dia_id": "D1:2"},
        ])])
        chunk = _make_chunk(
            dia_ids=["D1:1", "D1:2", "D1:3"],
            speakers=["Alice", "Bob", "Alice"],
            timestamps=["2024-01-01", "2024-01-02", "2024-01-03"],
        )
        ex = FactExtractor(_client(backend))
        result = ex.extract_from_chunk(chunk)
        mu = result.memory_units[0]
        assert mu.source_dia_ids == ["D1:2"]
        assert mu.source_speaker == "Bob"  # picked up from index 1
        assert mu.timestamp == "2024-01-02"


# ---------------------------------------------------------------------------
# Heuristic fallback (enable_llm=False)
# ---------------------------------------------------------------------------


class TestHeuristicMode:
    def test_no_llm_calls_when_disabled(self) -> None:
        backend = FakeBackend([])
        client = _client(backend)
        ex = FactExtractor(client, enable_llm=False)
        chunk = _make_chunk()
        result = ex.extract_from_chunk(chunk)
        assert backend.calls == []
        assert result.used_heuristic is True
        assert result.success is True

    def test_heuristic_extracts_sentences(self) -> None:
        ex = FactExtractor(client=None, enable_llm=False)
        chunk = _make_chunk(text=(
            "[Conversation: conv_1 | Session: session_1 | Dialog IDs: D1:1]\n"
            "Caroline: I quit my job at Google. Starting at Microsoft on Monday."
        ))
        result = ex.extract_from_chunk(chunk)
        assert len(result.memory_units) >= 1
        assert all(m.confidence == 0.5 for m in result.memory_units)

    def test_heuristic_skips_questions(self) -> None:
        ex = FactExtractor(client=None, enable_llm=False)
        chunk = _make_chunk(text=(
            "[Conversation: conv_1]\n"
            "Caroline: What did you do today? I went to the store."
        ))
        result = ex.extract_from_chunk(chunk)
        claims = [m.claim for m in result.memory_units]
        assert all("?" not in c for c in claims)
        # The factual half should be present
        assert any("went to the store" in c for c in claims)

    def test_heuristic_skips_opinion_prefix(self) -> None:
        ex = FactExtractor(client=None, enable_llm=False)
        chunk = _make_chunk(text=(
            "[Conversation: conv_1]\n"
            "Caroline: I think maybe I should switch jobs. I left Google in March."
        ))
        result = ex.extract_from_chunk(chunk)
        claims = [m.claim for m in result.memory_units]
        # The "I think" sentence should be filtered
        assert not any(c.lower().startswith("i think") for c in claims)
        # The factual one should remain
        assert any("Google" in c for c in claims)


# ---------------------------------------------------------------------------
# Caching behaviour (FactExtractor delegates to OpenRouterClient cache)
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    def test_cache_hit_does_not_recall_llm(self, tmp_path) -> None:
        from locomo_memory.phase2.llm.cache import LLMCache

        cache = LLMCache(tmp_path / "c")
        backend = FakeBackend([_llm_response([
            {"claim": "x", "speaker": "Caroline", "source_dia_id": "D1:1"},
        ])])
        client = OpenRouterClient(backend=backend, cache=cache, max_retries=1)
        ex = FactExtractor(client)
        chunk = _make_chunk()

        r1 = ex.extract_from_chunk(chunk)
        r2 = ex.extract_from_chunk(chunk)

        assert len(backend.calls) == 1
        assert r2.from_cache is True
        # Both runs should produce equivalent MUs
        assert r1.memory_units[0].claim == r2.memory_units[0].claim


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_extract_from_chunks(self) -> None:
        backend = FakeBackend([
            _llm_response([{"claim": "fa", "speaker": "Caroline", "source_dia_id": "D1:1"}]),
            _llm_response([{"claim": "fb", "speaker": "Caroline", "source_dia_id": "D1:1"}]),
        ])
        ex = FactExtractor(_client(backend))
        c1 = _make_chunk(chunk_id="conv_1#semantic#0-0")
        c2 = _make_chunk(chunk_id="conv_1#semantic#1-1")
        results = ex.extract_from_chunks([c1, c2])
        assert len(results) == 2
        assert results[0].chunk_id == "conv_1#semantic#0-0"
        assert results[1].chunk_id == "conv_1#semantic#1-1"
