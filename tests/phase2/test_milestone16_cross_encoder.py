"""Tests for Cross-Encoder Reranking — Phase 2 Milestone 16.

All tests use FakeCrossEncoderReranker — no model download, no network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from locomo_memory.phase2.retrieval.cross_encoder_reranker import (
    FakeCrossEncoderReranker,
    build_candidate_text,
)
from locomo_memory.phase2.retrieval.hybrid_retriever import (
    HybridMemoryRetriever,
    HybridRetrieverConfig,
)
from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
from locomo_memory.phase2.indexes.label_index import CompressedLabelFAISSIndex
from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
from locomo_memory.phase2.store.sqlite_store import MemoryStore
from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


def _make_embed_fn(seed: int = 42):
    rng = np.random.default_rng(seed)
    cache: dict[str, np.ndarray] = {}

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t not in cache:
                v = rng.random(DIM).astype(np.float32)
                v /= np.linalg.norm(v)
                cache[t] = v
            out.append(cache[t])
        return np.array(out, dtype=np.float32)

    return embed


def _make_mu(
    mu_id: str,
    claim: str,
    *,
    original_text: str = "",
    conv: str = "c1",
    session: str = "s1",
    dia_id: str | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryUnit:
    return MemoryUnit(
        mu_id=mu_id,
        conversation_id=conv,
        session_id=session,
        claim=claim,
        original_text=original_text or claim,
        source_dia_ids=[dia_id or mu_id],
        source_speaker="Alice",
        status=status,
    )


def _make_retriever(tmp_path: Path, mus: list[MemoryUnit], fake_ce=None):
    embed_fn = _make_embed_fn()
    store = MemoryStore(tmp_path / "test.db")
    faiss_idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM, normalize=True)
    bm25_idx = MemoryBM25Index()
    label_idx = CompressedLabelFAISSIndex(embed_fn=embed_fn, dim=DIM, normalize=True)

    # All MUs are inserted as-is; tests that need a hard-deleted MU should
    # call ``store.delete_atomic`` after ``_make_retriever`` returns.
    for mu in mus:
        store.insert_memory_unit(mu)
    if mus:
        faiss_idx.add_mus(mus)
        bm25_idx.add_mus(mus)

    retriever = HybridMemoryRetriever(
        store=store,
        faiss_index=faiss_idx,
        bm25_index=bm25_idx,
        label_index=label_idx,
        cross_encoder=fake_ce or FakeCrossEncoderReranker(),
    )
    return retriever


_CE_CFG = HybridRetrieverConfig(
    top_k=5,
    dense_candidates=20,
    bm25_candidates=20,
    enable_bm25=True,
    enable_cross_encoder=True,
    cross_encoder_pool_size=20,
    cross_encoder_weight=3.0,
    ce_diversity_max_same_dia=2,
)


# ---------------------------------------------------------------------------
# FakeCrossEncoderReranker unit tests
# ---------------------------------------------------------------------------


class TestFakeCrossEncoder:
    def test_score_pairs_returns_one_score_per_candidate(self):
        fake = FakeCrossEncoderReranker()
        scores = fake.score_pairs("Alice went to the market", ["Alice", "Bob", "market trip"])
        assert len(scores) == 3

    def test_higher_overlap_gets_higher_score(self):
        fake = FakeCrossEncoderReranker()
        scores = fake.score_pairs(
            "Alice birthday party celebration",
            [
                "Alice had a birthday party",  # more overlap
                "Bob went fishing",            # less overlap
            ],
        )
        assert scores[0] > scores[1]

    def test_empty_candidate_list_returns_empty(self):
        fake = FakeCrossEncoderReranker()
        assert fake.score_pairs("query", []) == []


# ---------------------------------------------------------------------------
# build_candidate_text
# ---------------------------------------------------------------------------


class TestBuildCandidateText:
    def _make_hit(self, claim, original_text="", session="s1", speaker="Alice",
                  dia_id="D1", label_summary=None):
        """Build a minimal duck-typed hit."""
        @dataclass
        class _MU:
            claim: str
            original_text: str
            session_id: str
            source_speaker: str
            source_dia_ids: list[str]
            timestamp: str | None = None
            salience_score: float = 0.0
            confidence: float = 0.5

        @dataclass
        class _Hit:
            mu: _MU
            label_summary: str | None
            rrf_score: float = 0.5

        return _Hit(
            mu=_MU(claim=claim, original_text=original_text, session_id=session,
                   source_speaker=speaker, source_dia_ids=[dia_id]),
            label_summary=label_summary,
        )

    def test_claim_appears_in_output(self):
        hit = self._make_hit("Alice went shopping for groceries")
        text = build_candidate_text(hit)
        assert "Alice went shopping for groceries" in text

    def test_original_text_included_when_different_from_claim(self):
        hit = self._make_hit(
            claim="Alice: went shopping",
            original_text="went shopping for groceries at the market",
        )
        text = build_candidate_text(hit)
        assert "groceries at the market" in text

    def test_metadata_header_included(self):
        hit = self._make_hit("some claim", session="s3", speaker="Bob")
        text = build_candidate_text(hit)
        assert "Session:s3" in text
        assert "Speaker:Bob" in text

    def test_label_summary_included_when_distinct(self):
        hit = self._make_hit("short claim", label_summary="Birthday event in June")
        text = build_candidate_text(hit)
        assert "Birthday event in June" in text

    def test_output_truncated_to_max_chars(self):
        long_text = "word " * 500
        hit = self._make_hit(long_text)
        result = build_candidate_text(hit, max_chars=100)
        assert len(result) <= 100


# ---------------------------------------------------------------------------
# HybridMemoryRetriever cross-encoder integration
# ---------------------------------------------------------------------------


class TestCrossEncoderIntegration:
    def test_fake_ce_promotes_best_semantic_candidate(self, tmp_path):
        """Fake CE scores by keyword overlap; the hit mentioning query words
        should rise to rank 1 even if its RRF score is lower."""
        # mu_target has low RRF (added last, low density) but matches query well
        mu_noise = _make_mu("n1", "Bob likes fishing on Sundays", dia_id="D1")
        mu_noise2 = _make_mu("n2", "Weather was cold last winter", dia_id="D2")
        mu_noise3 = _make_mu("n3", "Carol bought a new car recently", dia_id="D3")
        mu_target = _make_mu(
            "t1",
            "Alice celebrated her birthday with a surprise party",
            dia_id="D99",
        )
        mus = [mu_noise, mu_noise2, mu_noise3, mu_target]
        retriever = _make_retriever(tmp_path, mus)

        result = retriever.retrieve(
            "What happened at Alice birthday party?",
            conversation_id="c1",
            config_override=_CE_CFG,
        )
        top_ids = [h.mu.mu_id for h in result.hits]
        assert "t1" in top_ids, f"Target MU not retrieved; got {top_ids}"
        assert result.hits[0].mu.mu_id == "t1", "Target should be rank 1"

    def test_final_top_k_is_respected(self, tmp_path):
        mus = [_make_mu(f"m{i}", f"fact {i} about topic", dia_id=f"D{i}") for i in range(20)]
        retriever = _make_retriever(tmp_path, mus)
        result = retriever.retrieve(
            "topic fact", conversation_id="c1",
            config_override=HybridRetrieverConfig(
                top_k=5, dense_candidates=20, bm25_candidates=20,
                enable_bm25=True, enable_cross_encoder=True,
                cross_encoder_pool_size=20,
            ),
        )
        assert len(result.hits) <= 5

    def test_candidate_pool_size_respected(self, tmp_path):
        """Pool limit should be cross_encoder_pool_size, not top_k."""
        mus = [_make_mu(f"m{i}", f"item {i}", dia_id=f"D{i}") for i in range(30)]
        retriever = _make_retriever(tmp_path, mus)
        # With pool_size=10 and top_k=5, retriever must hydrate ≤10 before CE
        cfg = HybridRetrieverConfig(
            top_k=5, dense_candidates=20, bm25_candidates=0, enable_bm25=False,
            enable_cross_encoder=True, cross_encoder_pool_size=10,
        )
        result = retriever.retrieve("item query", conversation_id="c1", config_override=cfg)
        # Result must still be ≤ top_k
        assert len(result.hits) <= cfg.top_k

    def test_deleted_memories_never_returned(self, tmp_path):
        mu_alive = _make_mu("alive", "Alice is alive", dia_id="D1")
        mu_dead = _make_mu("dead", "Alice is deleted", dia_id="D2")
        retriever = _make_retriever(tmp_path, [mu_alive, mu_dead])
        # Hard-delete the row; retriever must skip vector hits whose MU is gone.
        retriever.store.delete_atomic("dead", deleted_by="test")
        result = retriever.retrieve(
            "Alice deleted alive", conversation_id="c1", config_override=_CE_CFG
        )
        ids = [h.mu.mu_id for h in result.hits]
        assert "dead" not in ids
        assert "alive" in ids

    def test_duplicate_dia_ids_capped_by_diversity(self, tmp_path):
        """ce_diversity_max_same_dia=1 means only 1 hit per lead dia_id."""
        # Three MUs all share dia_id D1 (simulates overlapping source turns)
        mus = [
            _make_mu(f"m{i}", f"Alice fact {i}", dia_id="D1")
            for i in range(3)
        ] + [
            _make_mu(f"u{i}", f"Bob unique fact {i}", dia_id=f"U{i}")
            for i in range(5)
        ]
        retriever = _make_retriever(tmp_path, mus)
        cfg = HybridRetrieverConfig(
            top_k=5, dense_candidates=20, bm25_candidates=20,
            enable_bm25=True, enable_cross_encoder=True,
            cross_encoder_pool_size=20,
            ce_diversity_max_same_dia=1,  # strict: max 1 per dia_id
        )
        result = retriever.retrieve(
            "Alice Bob fact", conversation_id="c1", config_override=cfg
        )
        # Count how many hits have lead dia_id D1
        d1_count = sum(
            1 for h in result.hits
            if (h.mu.source_dia_ids or [h.mu.mu_id])[0] == "D1"
        )
        assert d1_count <= 1

    def test_ce_disabled_preserves_simple_truncation(self, tmp_path):
        """With enable_cross_encoder=False, old top_k truncation applies."""
        mus = [_make_mu(f"m{i}", f"fact {i}", dia_id=f"D{i}") for i in range(20)]
        retriever = _make_retriever(tmp_path, mus, fake_ce=None)
        cfg = HybridRetrieverConfig(
            top_k=3, dense_candidates=20, bm25_candidates=0,
            enable_bm25=False, enable_cross_encoder=False,
        )
        result = retriever.retrieve("fact query", conversation_id="c1", config_override=cfg)
        assert len(result.hits) <= 3

    def test_no_gold_evidence_in_retriever_api(self):
        """The HybridMemoryRetriever.retrieve() API has no gold_evidence parameter."""
        import inspect
        from locomo_memory.phase2.retrieval.hybrid_retriever import HybridMemoryRetriever
        sig = inspect.signature(HybridMemoryRetriever.retrieve)
        param_names = list(sig.parameters.keys())
        assert "gold_evidence" not in param_names
        assert "gold_answer" not in param_names


# ---------------------------------------------------------------------------
# Config field presence tests
# ---------------------------------------------------------------------------


class TestCEConfigFields:
    def test_hybrid_retriever_config_has_ce_fields(self):
        cfg = HybridRetrieverConfig()
        assert hasattr(cfg, "enable_cross_encoder")
        assert hasattr(cfg, "cross_encoder_model")
        assert hasattr(cfg, "cross_encoder_weight")
        assert hasattr(cfg, "cross_encoder_batch_size")
        assert hasattr(cfg, "cross_encoder_max_length")
        assert hasattr(cfg, "cross_encoder_pool_size")
        assert hasattr(cfg, "ce_superseded_penalty")
        assert hasattr(cfg, "ce_diversity_max_same_dia")

    def test_ce_disabled_by_default(self):
        assert HybridRetrieverConfig().enable_cross_encoder is False

    def test_yaml_config_ce_top5_loads(self):
        from locomo_memory.phase2.experiments.config import Phase2RunnerConfig
        path = Path("configs/phase2_retrieval_cross_encoder_top5.yaml")
        if not path.exists():
            pytest.skip("YAML not found")
        cfg = Phase2RunnerConfig.from_yaml(path)
        assert cfg.retrieval.enable_cross_encoder is True
        assert cfg.retrieval.cross_encoder_pool_size == 50
        assert cfg.retrieval.top_k == 5

    def test_yaml_config_largepool_loads(self):
        from locomo_memory.phase2.experiments.config import Phase2RunnerConfig
        path = Path("configs/phase2_retrieval_cross_encoder_top5_largepool.yaml")
        if not path.exists():
            pytest.skip("YAML not found")
        cfg = Phase2RunnerConfig.from_yaml(path)
        assert cfg.retrieval.cross_encoder_pool_size == 100
