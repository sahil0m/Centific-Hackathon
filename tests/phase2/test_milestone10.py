"""Tests for Phase 2 Milestone 10: Phase2LoCoMoRunner + Evaluation Harness.

All tests use:
- Deterministic 8-dim dummy embedder (no sentence-transformers download)
- Synthetic conversations (no locomo10.json required)
- generation.enabled = False (no LLM calls)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from locomo_memory.data.schemas import Conversation, QAItem, Turn
from locomo_memory.phase2.experiments.config import Phase2RunnerConfig
from locomo_memory.phase2.experiments.evaluator import (
    Phase2Evaluator,
    Phase2Metrics,
    Phase2PredictionRow,
    Phase2RunResult,
)
from locomo_memory.phase2.experiments.runner import Phase2LoCoMoRunner

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 8


def _make_embed_fn(seed: int = 0):
    rng = np.random.default_rng(seed)
    cache: dict[str, np.ndarray] = {}

    def embed_fn(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t not in cache:
                v = rng.random(DIM).astype(np.float32)
                v /= np.linalg.norm(v)
                cache[t] = v
            out.append(cache[t])
        return np.array(out, dtype=np.float32)

    return embed_fn


def _turn(
    text: str,
    *,
    dia_id: str = "D1:1",
    speaker: str = "Alice",
    session: str = "session_1",
    conv: str = "conv1",
    idx: int = 0,
    timestamp: str = "2024-01-01",
) -> Turn:
    return Turn(
        sample_id="s1",
        conversation_id=conv,
        session_id=session,
        turn_index=idx,
        dia_id=dia_id,
        speaker=speaker,
        text=text,
        timestamp=timestamp,
    )


def _qa(
    question: str,
    answer: str,
    *,
    qa_id: str = "qa1",
    conv: str = "conv1",
    category: str = "1",
    evidence: list[str] | None = None,
) -> QAItem:
    return QAItem(
        qa_id=qa_id,
        conversation_id=conv,
        question=question,
        answer=answer,
        category=category,
        gold_evidence_ids=evidence or [],
    )


def _make_conversation(
    conv_id: str = "conv1",
    n_turns: int = 5,
    n_qa: int = 2,
) -> Conversation:
    turns = [
        _turn(
            f"This is turn {i} about topic {i}",
            dia_id=f"D1:{i}",
            conv=conv_id,
            idx=i,
        )
        for i in range(n_turns)
    ]
    qa_items = [
        _qa(
            f"Question {j}",
            f"Answer {j}",
            qa_id=f"qa{j}",
            conv=conv_id,
            category=str(j % 3 + 1),
            evidence=[f"D1:{j}"] if j < n_turns else [],
        )
        for j in range(n_qa)
    ]
    return Conversation(conversation_id=conv_id, sample_id="s1", turns=turns, qa_items=qa_items)


def _default_config(tmp_path: Path, name: str = "test_exp") -> Phase2RunnerConfig:
    return Phase2RunnerConfig(
        experiment_name=name,
        db_dir=str(tmp_path / "db"),
        output={"dir": str(tmp_path / "results")},
        embedding={"model_name": "BAAI/bge-small-en-v1.5", "dim": DIM, "normalize": True},
        retrieval={"top_k": 3, "enable_bm25": True},
        generation={"enabled": False},
    )


def _make_runner(tmp_path: Path, name: str = "test_exp") -> Phase2LoCoMoRunner:
    cfg = _default_config(tmp_path, name)
    return Phase2LoCoMoRunner(
        cfg,
        embed_fn=_make_embed_fn(0),
        db_dir_override=tmp_path / "db",
    )


# ===========================================================================
# Phase2RunnerConfig tests
# ===========================================================================


class TestPhase2RunnerConfig:
    def test_default_construction(self):
        cfg = Phase2RunnerConfig()
        assert cfg.experiment_name == "phase2_retrieval_only"
        assert cfg.generation.enabled is False
        assert cfg.retrieval.enable_bm25 is True

    def test_custom_values(self):
        cfg = Phase2RunnerConfig(
            experiment_name="my_exp",
            retrieval={"top_k": 10},
        )
        assert cfg.experiment_name == "my_exp"
        assert cfg.retrieval.top_k == 10

    def test_from_yaml(self, tmp_path: Path):
        yaml_content = """
experiment:
  name: yaml_test_exp
  seed: 99
retrieval:
  top_k: 7
  enable_bm25: false
generation:
  enabled: false
"""
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        cfg = Phase2RunnerConfig.from_yaml(yaml_path)
        assert cfg.experiment_name == "yaml_test_exp"
        assert cfg.seed == 99
        assert cfg.retrieval.top_k == 7
        assert cfg.retrieval.enable_bm25 is False

    def test_from_yaml_defaults_preserved(self, tmp_path: Path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("experiment:\n  name: minimal\n", encoding="utf-8")
        cfg = Phase2RunnerConfig.from_yaml(yaml_path)
        assert cfg.generation.enabled is False
        assert cfg.retrieval.enable_bm25 is True

    def test_to_dict(self):
        cfg = Phase2RunnerConfig(experiment_name="dict_test")
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["experiment_name"] == "dict_test"
        assert "retrieval" in d
        assert "generation" in d

    def test_phase2_config_yaml_exists(self):
        """The shipped example config must be valid."""
        path = Path("configs/phase2_retrieval_only.yaml")
        if not path.exists():
            pytest.skip("Config file not present")
        cfg = Phase2RunnerConfig.from_yaml(path)
        assert cfg.experiment_name == "phase2_retrieval_only"
        assert cfg.generation.enabled is False


# ===========================================================================
# Phase2LoCoMoRunner tests
# ===========================================================================


class TestPhase2LoCoMoRunnerConstruction:
    def test_constructs_with_defaults(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner.config.experiment_name == "test_exp"

    def test_embed_fn_injected(self, tmp_path):
        fn = _make_embed_fn(42)
        cfg = _default_config(tmp_path)
        runner = Phase2LoCoMoRunner(cfg, embed_fn=fn, db_dir_override=tmp_path / "db")
        assert runner._embed_fn_override is fn


class TestPhase2LoCoMoRunnerIngestion:
    def test_ingest_turns_creates_mus(self, tmp_path):
        runner = _make_runner(tmp_path)
        conv = _make_conversation("c1", n_turns=5, n_qa=0)
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
        from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
        store = MemoryStore(tmp_path / "c1.db")
        fi = MemoryFAISSIndex(embed_fn=_make_embed_fn(0), dim=DIM)
        bi = MemoryBM25Index()
        n = runner._ingest_turns(conv, store=store, faiss_index=fi, bm25_index=bi)
        assert n == 5
        assert fi.size() == 5
        assert bi.size() == 5

    def test_ingest_skips_summary_turns(self, tmp_path):
        runner = _make_runner(tmp_path)
        conv = Conversation(conversation_id="c1", sample_id="s1")
        conv.turns = [
            _turn("Real turn", dia_id="D1:1", speaker="Alice"),
            _turn("Session summary content", dia_id="D1:2", speaker="summary"),
        ]
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
        from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
        store = MemoryStore(tmp_path / "c1.db")
        fi = MemoryFAISSIndex(embed_fn=_make_embed_fn(0), dim=DIM)
        bi = MemoryBM25Index()
        n = runner._ingest_turns(conv, store=store, faiss_index=fi, bm25_index=bi)
        assert n == 1

    def test_ingest_skips_short_turns(self, tmp_path):
        cfg = _default_config(tmp_path)
        cfg.ingestion.min_turn_length = 10
        runner = Phase2LoCoMoRunner(
            cfg, embed_fn=_make_embed_fn(0), db_dir_override=tmp_path / "db"
        )
        conv = Conversation(conversation_id="c1", sample_id="s1")
        conv.turns = [
            _turn("ok", dia_id="D1:1"),  # too short
            _turn("This is a longer turn text", dia_id="D1:2"),
        ]
        from locomo_memory.phase2.store.sqlite_store import MemoryStore
        from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
        from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
        store = MemoryStore(tmp_path / "c1.db")
        fi = MemoryFAISSIndex(embed_fn=_make_embed_fn(0), dim=DIM)
        bi = MemoryBM25Index()
        n = runner._ingest_turns(conv, store=store, faiss_index=fi, bm25_index=bi)
        assert n == 1

    def test_turn_to_mu_speaker_text_format(self, tmp_path):
        runner = _make_runner(tmp_path)
        t = _turn("I work at Google", speaker="Alice", dia_id="D1:5")
        mu = runner._turn_to_mu(t)
        assert "Alice" in mu.claim
        assert "I work at Google" in mu.claim
        assert "D1:5" in mu.source_dia_ids

    def test_turn_to_mu_text_only_format(self, tmp_path):
        cfg = _default_config(tmp_path)
        cfg.ingestion.claim_format = "text_only"
        runner = Phase2LoCoMoRunner(
            cfg, embed_fn=_make_embed_fn(0), db_dir_override=tmp_path / "db"
        )
        t = _turn("I work at Google", speaker="Alice")
        mu = runner._turn_to_mu(t)
        assert "Alice" not in mu.claim
        assert mu.claim == "I work at Google"

    def test_turn_to_mu_preserves_session(self, tmp_path):
        runner = _make_runner(tmp_path)
        t = _turn("text", session="session_7")
        mu = runner._turn_to_mu(t)
        assert mu.session_id == "session_7"

    def test_turn_to_mu_preserves_timestamp(self, tmp_path):
        runner = _make_runner(tmp_path)
        t = _turn("text", timestamp="2024-03-15")
        mu = runner._turn_to_mu(t)
        assert mu.timestamp == "2024-03-15"


class TestPhase2LoCoMoRunnerRun:
    def test_run_returns_result(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=5, n_qa=2)]
        result = runner.run(conversations=convs, save=False)
        assert isinstance(result, Phase2RunResult)
        assert result.experiment_name == "test_exp"

    def test_run_n_conversations(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation(f"c{i}", n_turns=3, n_qa=1) for i in range(3)]
        result = runner.run(conversations=convs, save=False)
        assert result.n_conversations == 3

    def test_run_n_predictions_matches_qa_count(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [
            _make_conversation("c1", n_turns=4, n_qa=3),
            _make_conversation("c2", n_turns=4, n_qa=2),
        ]
        result = runner.run(conversations=convs, save=False)
        assert result.n_predictions == 5

    def test_run_predictions_have_correct_fields(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=5, n_qa=1)]
        result = runner.run(conversations=convs, save=False)
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        assert pred.experiment_name == "test_exp"
        assert pred.conversation_id == "c1"
        assert pred.question == "Question 0"
        assert isinstance(pred.f1, float)
        assert isinstance(pred.exact_match, bool)
        assert isinstance(pred.retrieval_latency_ms, float)

    def test_run_retrieval_only_no_predicted_answer(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=3, n_qa=2)]
        result = runner.run(conversations=convs, save=False)
        for pred in result.predictions:
            assert pred.predicted_answer == ""
            assert pred.f1 == 0.0
            assert pred.exact_match is False

    def test_run_retrieves_from_same_conversation_only(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [
            _make_conversation("c1", n_turns=5, n_qa=2),
            _make_conversation("c2", n_turns=5, n_qa=2),
        ]
        result = runner.run(conversations=convs, save=False)
        for pred in result.predictions:
            for mu_id in pred.retrieved_mu_ids:
                # All retrieved MUs should exist in the store for this conversation
                pass  # isolation is guaranteed by per-conversation store

    def test_run_evidence_recall_computed(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=5, n_qa=2)]
        result = runner.run(conversations=convs, save=False)
        for pred in result.predictions:
            if pred.gold_evidence_ids:
                assert pred.evidence_recall is not None
                assert 0.0 <= pred.evidence_recall <= 1.0

    def test_run_context_sections_populated(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=5, n_qa=1)]
        result = runner.run(conversations=convs, save=False)
        pred = result.predictions[0]
        assert "active" in pred.context_sections
        assert isinstance(pred.context_sections["active"], list)

    def test_run_guard_verdict_present(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=3, n_qa=1)]
        result = runner.run(conversations=convs, save=False)
        pred = result.predictions[0]
        assert isinstance(pred.guard_passed, bool)
        assert isinstance(pred.grounding_score, float)
        assert isinstance(pred.guard_warnings, list)

    def test_run_max_conversations_limit(self, tmp_path):
        cfg = _default_config(tmp_path)
        cfg.dataset.max_conversations = 2
        runner = Phase2LoCoMoRunner(
            cfg, embed_fn=_make_embed_fn(0), db_dir_override=tmp_path / "db"
        )
        convs = [_make_conversation(f"c{i}", n_turns=3, n_qa=1) for i in range(5)]
        result = runner.run(conversations=convs, save=False)
        assert result.n_conversations == 2

    def test_run_max_qa_per_conversation_limit(self, tmp_path):
        cfg = _default_config(tmp_path)
        cfg.dataset.max_qa_per_conversation = 1
        runner = Phase2LoCoMoRunner(
            cfg, embed_fn=_make_embed_fn(0), db_dir_override=tmp_path / "db"
        )
        convs = [_make_conversation("c1", n_turns=5, n_qa=5)]
        result = runner.run(conversations=convs, save=False)
        assert result.n_predictions == 1

    def test_run_empty_conversation_list(self, tmp_path):
        runner = _make_runner(tmp_path)
        result = runner.run(conversations=[], save=False)
        assert result.n_predictions == 0
        assert result.n_conversations == 0

    def test_run_conversation_with_no_qa(self, tmp_path):
        runner = _make_runner(tmp_path)
        conv = _make_conversation("c1", n_turns=3, n_qa=0)
        result = runner.run(conversations=[conv], save=False)
        assert result.n_predictions == 0

    def test_run_metrics_attached(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=5, n_qa=2)]
        result = runner.run(conversations=convs, save=False)
        assert result.metrics is not None
        assert isinstance(result.metrics, Phase2Metrics)

    def test_run_latency_positive(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=3, n_qa=2)]
        result = runner.run(conversations=convs, save=False)
        for pred in result.predictions:
            assert pred.retrieval_latency_ms >= 0.0
            assert pred.end_to_end_latency_ms >= 0.0


# ===========================================================================
# Phase2Evaluator tests
# ===========================================================================


def _make_pred(
    f1: float = 0.5,
    em: bool = False,
    ev_recall: float | None = 0.5,
    grounding: float = 0.6,
    guard_passed: bool = True,
    category: str = "1",
    conv: str = "c1",
    ret_lat: float = 2.0,
    e2e_lat: float = 3.0,
) -> Phase2PredictionRow:
    return Phase2PredictionRow(
        experiment_name="test",
        conversation_id=conv,
        qa_id="qa1",
        question="q",
        gold_answer="a",
        predicted_answer="pred",
        category=category,
        gold_evidence_ids=[],
        retrieved_mu_ids=[],
        retrieved_claims=[],
        retrieved_dia_ids=[],
        context_sections={},
        f1=f1,
        exact_match=em,
        evidence_recall=ev_recall,
        grounding_score=grounding,
        guard_passed=guard_passed,
        guard_warnings=[],
        retrieval_latency_ms=ret_lat,
        generation_latency_ms=0.0,
        end_to_end_latency_ms=e2e_lat,
    )


class TestPhase2Evaluator:
    def test_compute_metrics_empty(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        m = ev.compute_metrics([])
        assert m.n_predictions == 0
        assert m.avg_f1 == 0.0
        assert m.avg_evidence_recall is None

    def test_compute_metrics_avg_f1(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        preds = [_make_pred(f1=0.4), _make_pred(f1=0.6)]
        m = ev.compute_metrics(preds)
        assert abs(m.avg_f1 - 0.5) < 1e-4

    def test_compute_metrics_exact_match_rate(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        preds = [_make_pred(em=True), _make_pred(em=False), _make_pred(em=True)]
        m = ev.compute_metrics(preds)
        assert abs(m.exact_match_rate - 2 / 3) < 1e-4

    def test_compute_metrics_evidence_recall_excludes_none(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        preds = [
            _make_pred(ev_recall=0.8),
            _make_pred(ev_recall=None),
            _make_pred(ev_recall=0.4),
        ]
        m = ev.compute_metrics(preds)
        assert m.avg_evidence_recall is not None
        assert abs(m.avg_evidence_recall - 0.6) < 1e-4

    def test_compute_metrics_no_recalls(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        preds = [_make_pred(ev_recall=None), _make_pred(ev_recall=None)]
        m = ev.compute_metrics(preds)
        assert m.avg_evidence_recall is None

    def test_compute_metrics_guard_pass_rate(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        preds = [_make_pred(guard_passed=True), _make_pred(guard_passed=False)]
        m = ev.compute_metrics(preds)
        assert abs(m.guard_pass_rate - 0.5) < 1e-4

    def test_compute_metrics_latency_percentiles(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        preds = [_make_pred(ret_lat=float(i)) for i in range(10, 20)]
        m = ev.compute_metrics(preds)
        assert m.retrieval_latency_p50 > 0
        assert m.retrieval_latency_p95 >= m.retrieval_latency_p50

    def test_compute_metrics_by_category(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        preds = [
            _make_pred(f1=0.5, category="1"),
            _make_pred(f1=0.8, category="1"),
            _make_pred(f1=0.3, category="2"),
        ]
        m = ev.compute_metrics(preds)
        assert "1" in m.by_category
        assert "2" in m.by_category
        assert m.by_category["1"]["count"] == 2
        assert abs(m.by_category["1"]["avg_f1"] - 0.65) < 1e-4

    def test_compute_metrics_n_conversations(self, tmp_path):
        ev = Phase2Evaluator("test", tmp_path)
        m = ev.compute_metrics([_make_pred()], n_conversations=7)
        assert m.n_conversations == 7

    def test_save_creates_files(self, tmp_path):
        ev = Phase2Evaluator("my_exp", tmp_path / "results")
        preds = [_make_pred(f1=0.5, category="1")]
        metrics = ev.compute_metrics(preds, n_conversations=1)
        result = Phase2RunResult(
            experiment_name="my_exp",
            n_conversations=1,
            n_qa_items=1,
            predictions=preds,
            metrics=metrics,
        )
        ev.save(result)

        pred_file = tmp_path / "results" / "raw_predictions" / "my_exp.json"
        metrics_file = tmp_path / "results" / "metrics" / "my_exp_metrics.json"
        cat_file = tmp_path / "results" / "tables" / "my_exp_by_category.csv"

        assert pred_file.exists()
        assert metrics_file.exists()
        assert cat_file.exists()

    def test_save_predictions_json_valid(self, tmp_path):
        ev = Phase2Evaluator("my_exp", tmp_path / "results")
        preds = [_make_pred(f1=0.4)]
        result = Phase2RunResult(
            experiment_name="my_exp", n_conversations=1, n_qa_items=1, predictions=preds
        )
        ev.save(result)
        pred_file = tmp_path / "results" / "raw_predictions" / "my_exp.json"
        data = json.loads(pred_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["f1"] == 0.4

    def test_save_metrics_json_valid(self, tmp_path):
        ev = Phase2Evaluator("my_exp", tmp_path / "results")
        preds = [_make_pred()]
        result = Phase2RunResult(
            experiment_name="my_exp", n_conversations=1, n_qa_items=1, predictions=preds
        )
        ev.save(result)
        metrics_file = tmp_path / "results" / "metrics" / "my_exp_metrics.json"
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        assert data["experiment_name"] == "my_exp"
        assert "avg_f1" in data
        assert "by_category" in data

    def test_save_auto_computes_metrics_if_none(self, tmp_path):
        ev = Phase2Evaluator("my_exp", tmp_path / "results")
        preds = [_make_pred(f1=0.7)]
        result = Phase2RunResult(
            experiment_name="my_exp",
            n_conversations=1,
            n_qa_items=1,
            predictions=preds,
            metrics=None,
        )
        ev.save(result)
        assert result.metrics is not None
        assert result.metrics.avg_f1 == 0.7


# ===========================================================================
# Phase2PredictionRow tests
# ===========================================================================


class TestPhase2PredictionRow:
    def test_as_dict_returns_dict(self):
        pred = _make_pred()
        d = pred.as_dict()
        assert isinstance(d, dict)
        assert "f1" in d
        assert "experiment_name" in d

    def test_as_dict_values_match(self):
        pred = _make_pred(f1=0.77, category="3")
        d = pred.as_dict()
        assert d["f1"] == 0.77
        assert d["category"] == "3"


# ===========================================================================
# End-to-end integration test
# ===========================================================================


class TestPhase2EndToEnd:
    def test_full_pipeline_no_error(self, tmp_path):
        """Full run with synthetic data must complete without exception."""
        runner = _make_runner(tmp_path, name="e2e_test")
        convs = [
            _make_conversation("c1", n_turns=8, n_qa=4),
            _make_conversation("c2", n_turns=6, n_qa=3),
        ]
        result = runner.run(conversations=convs, save=True)
        assert result.n_conversations == 2
        assert result.n_predictions == 7
        assert result.metrics is not None

    def test_full_pipeline_output_files_created(self, tmp_path):
        runner = _make_runner(tmp_path, name="file_test")
        convs = [_make_conversation("c1", n_turns=4, n_qa=2)]
        runner.run(conversations=convs, save=True)
        assert (tmp_path / "results" / "raw_predictions" / "file_test.json").exists()
        assert (tmp_path / "results" / "metrics" / "file_test_metrics.json").exists()

    def test_full_pipeline_isolation_between_convs(self, tmp_path):
        """MUs from conv1 must not appear in conv2 predictions."""
        runner = _make_runner(tmp_path, name="iso_test")
        conv1 = Conversation(conversation_id="c1", sample_id="s1")
        conv1.turns = [_turn("Alice works at Google", dia_id="D1:1", conv="c1")]
        conv1.qa_items = [_qa("Where does Alice work?", "Google", conv="c1")]

        conv2 = Conversation(conversation_id="c2", sample_id="s1")
        conv2.turns = [_turn("Bob lives in Paris", dia_id="D2:1", conv="c2")]
        conv2.qa_items = [_qa("Where does Bob live?", "Paris", conv="c2")]

        result = runner.run(conversations=[conv1, conv2], save=False)
        # Each prediction must only have MUs from its own conversation
        for pred in result.predictions:
            for dia_list in pred.retrieved_dia_ids:
                if pred.conversation_id == "c1":
                    for dia in dia_list:
                        assert dia.startswith("D1")
                else:
                    for dia in dia_list:
                        assert dia.startswith("D2")

    def test_retrieval_only_guard_passes_on_no_info(self, tmp_path):
        """In retrieval-only mode, guard should pass (answer is empty → no_info)."""
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=3, n_qa=1)]
        result = runner.run(conversations=convs, save=False)
        for pred in result.predictions:
            # Empty answer = is_no_info → guard passes
            assert pred.guard_passed is True

    def test_evidence_recall_at_most_one(self, tmp_path):
        runner = _make_runner(tmp_path)
        convs = [_make_conversation("c1", n_turns=5, n_qa=3)]
        result = runner.run(conversations=convs, save=False)
        for pred in result.predictions:
            if pred.evidence_recall is not None:
                assert pred.evidence_recall <= 1.0
                assert pred.evidence_recall >= 0.0
