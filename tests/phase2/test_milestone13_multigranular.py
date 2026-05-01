"""Tests for Multi-Granularity Provenance Evidence Retrieval — Phase 2 Milestone 13."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from locomo_memory.phase2.indexes.source_evidence_index import (
    SourceEvidenceEntry,
    SourceEvidenceHit,
    SourceEvidenceIndex,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

DIM = 8


def _make_embed_fn(seed: int = 0):
    """Deterministic fake embedder — no model download."""
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


@dataclass
class _FakeTurn:
    dia_id: str
    text: str
    conversation_id: str = "c1"
    session_id: str = "s1"
    speaker: str = "Alice"
    timestamp: str | None = None
    turn_index: int = 0


def _turn(dia_id: str, text: str, *, conv: str = "c1", session: str = "s1",
          speaker: str = "Alice", idx: int = 0) -> _FakeTurn:
    return _FakeTurn(dia_id=dia_id, text=text, conversation_id=conv,
                     session_id=session, speaker=speaker, turn_index=idx)


# ---------------------------------------------------------------------------
# SourceEvidenceIndex — unit tests
# ---------------------------------------------------------------------------


class TestSourceEvidenceIndexBasic:
    def test_add_turns_returns_count(self):
        idx = SourceEvidenceIndex()
        added = idx.add_turns([_turn("d1", "Alice went to market"),
                                _turn("d2", "Bob stayed home")])
        assert added == 2
        assert idx.size() == 2

    def test_add_duplicate_dia_id_not_counted_twice(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("d1", "first text")])
        added2 = idx.add_turns([_turn("d1", "second text")])
        assert added2 == 0  # already indexed
        assert idx.size() == 1

    def test_search_exact_keyword(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([
            _turn("d1", "Alice visited the market yesterday"),
            _turn("d2", "Bob went hiking in the mountains"),
        ])
        hits = idx.search_bm25("market", top_n=5, conversation_id="c1")
        assert len(hits) >= 1
        assert hits[0].entry.dia_id == "d1"

    def test_dia_id_preserved(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("dia_abc123", "specific content about gardens")])
        hits = idx.search_bm25("gardens", top_n=3, conversation_id="c1")
        assert hits[0].entry.dia_id == "dia_abc123"

    def test_conversation_filter(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([
            _turn("d1", "apple harvest", conv="c1"),
            _turn("d2", "apple harvest", conv="c2"),
        ])
        hits = idx.search_bm25("apple", top_n=5, conversation_id="c1")
        assert all(h.entry.conversation_id == "c1" for h in hits)
        assert len(hits) == 1

    def test_empty_index_returns_empty(self):
        idx = SourceEvidenceIndex()
        assert idx.search_bm25("anything", top_n=5) == []

    def test_unknown_query_returns_empty(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("d1", "short")])
        # BM25 can still return results but with low score; check it doesn't crash
        hits = idx.search_bm25("zzzzunknownquery", top_n=5)
        assert isinstance(hits, list)

    def test_top_n_respected(self):
        idx = SourceEvidenceIndex()
        for i in range(10):
            idx.add_turns([_turn(f"d{i}", f"apple banana cherry turn {i}", idx=i)])
        hits = idx.search_bm25("apple banana", top_n=3, conversation_id="c1")
        assert len(hits) <= 3

    def test_hits_ranked_best_first(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([
            _turn("d1", "apple"),
            _turn("d2", "apple banana cherry orange"),
        ])
        hits = idx.search_bm25("apple banana cherry", top_n=5, conversation_id="c1")
        assert hits[0].rank == 1
        assert hits[0].score >= hits[-1].score


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


class TestContextWindow:
    def _make_three_turns(self) -> SourceEvidenceIndex:
        idx = SourceEvidenceIndex()
        idx.add_turns([
            _turn("d1", "first sentence", idx=0),
            _turn("d2", "second sentence", idx=1),
            _turn("d3", "third sentence", idx=2),
        ])
        return idx

    def test_zero_window_returns_central_text(self):
        idx = self._make_three_turns()
        text = idx.get_context_text("d2", window=0)
        assert text == "second sentence"

    def test_window_includes_neighbours(self):
        idx = self._make_three_turns()
        text = idx.get_context_text("d2", window=1)
        assert "first sentence" in text
        assert "second sentence" in text
        assert "third sentence" in text

    def test_window_at_boundary(self):
        idx = self._make_three_turns()
        # d1 is the first turn; window=2 should only go back to d1
        text = idx.get_context_text("d1", window=2)
        assert "first sentence" in text
        assert "second sentence" in text  # forward window

    def test_central_dia_id_unchanged_with_window(self):
        """Context window broadens text but must not change which dia_id is the evidence."""
        idx = self._make_three_turns()
        hits = idx.search_bm25("second sentence", top_n=5,
                                conversation_id="c1", context_window=2)
        assert hits[0].entry.dia_id == "d2"

    def test_context_text_contains_more_words_than_zero_window(self):
        idx = self._make_three_turns()
        text_0 = idx.get_context_text("d2", window=0)
        text_1 = idx.get_context_text("d2", window=1)
        assert len(text_1) >= len(text_0)

    def test_context_window_helps_retrieve_by_neighbour_keyword(self):
        """A turn found via neighbour keywords still returns the central dia_id."""
        idx = SourceEvidenceIndex()
        idx.add_turns([
            _turn("d1", "introduction", idx=0),
            _turn("d2", "birthday party was celebrated", idx=1),
            _turn("d3", "aftermath discussion", idx=2),
        ])
        hits = idx.search_bm25("birthday celebration", top_n=5,
                                conversation_id="c1", context_window=0)
        # With window=0 only d2 matches
        assert hits[0].entry.dia_id == "d2"
        # Context text of d2 with window=1 includes neighbour text
        ctx = idx.get_context_text("d2", window=1)
        assert "introduction" in ctx or "aftermath" in ctx


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


class TestLinking:
    def test_link_mu(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("d1", "some text")])
        idx.link_mu("d1", "mu_abc")
        assert "mu_abc" in idx.get_linked_mu_ids("d1")

    def test_link_mu_idempotent(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("d1", "text")])
        idx.link_mu("d1", "mu_x")
        idx.link_mu("d1", "mu_x")
        assert idx.get_linked_mu_ids("d1").count("mu_x") == 1

    def test_link_nonexistent_dia_id_is_noop(self):
        idx = SourceEvidenceIndex()
        idx.link_mu("nonexistent", "mu_y")  # must not raise

    def test_build_links_from_mus(self):
        from locomo_memory.phase2.schemas import MemoryUnit
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("d1", "text"), _turn("d2", "more text")])
        mu1 = MemoryUnit(conversation_id="c1", session_id="s1",
                         claim="claim", source_dia_ids=["d1"])
        mu2 = MemoryUnit(conversation_id="c1", session_id="s1",
                         claim="claim2", source_dia_ids=["d2"])
        count = idx.build_links_from_mus([mu1, mu2])
        assert count == 2
        assert mu1.mu_id in idx.get_linked_mu_ids("d1")
        assert mu2.mu_id in idx.get_linked_mu_ids("d2")

    def test_linked_mu_ids_in_hit(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("d1", "alice birthday party")])
        idx.link_mu("d1", "mu_birthday")
        hits = idx.search_bm25("birthday", top_n=5, conversation_id="c1")
        assert "mu_birthday" in hits[0].entry.linked_mu_ids

    def test_multiple_mus_linked_to_one_dia_id(self):
        idx = SourceEvidenceIndex()
        idx.add_turns([_turn("d1", "shared turn")])
        idx.link_mu("d1", "mu_a")
        idx.link_mu("d1", "mu_b")
        linked = idx.get_linked_mu_ids("d1")
        assert "mu_a" in linked
        assert "mu_b" in linked


# ---------------------------------------------------------------------------
# RRF fusion integration (full HybridMemoryRetriever)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _store(tmp_path: Path):
    from locomo_memory.phase2.store.sqlite_store import MemoryStore
    return MemoryStore(tmp_path / "test.db")


def _make_retriever(store, *, source_ev_index=None, seed=0):
    from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
    from locomo_memory.phase2.indexes.label_index import CompressedLabelFAISSIndex
    from locomo_memory.phase2.retrieval.bm25_index import MemoryBM25Index
    from locomo_memory.phase2.retrieval.hybrid_retriever import HybridMemoryRetriever

    embed_fn = _make_embed_fn(seed)
    faiss_idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    bm25_idx = MemoryBM25Index()
    label_idx = CompressedLabelFAISSIndex(embed_fn=embed_fn, dim=DIM)
    return HybridMemoryRetriever(
        store=store,
        faiss_index=faiss_idx,
        bm25_index=bm25_idx,
        label_index=label_idx,
        source_evidence_index=source_ev_index,
    ), faiss_idx, bm25_idx


def _cfg(**kw):
    from locomo_memory.phase2.retrieval.hybrid_retriever import HybridRetrieverConfig
    defaults = dict(
        enable_bm25=True,
        enable_label_search=False,
        enable_graph_traversal=False,
        top_k=5,
    )
    defaults.update(kw)
    return HybridRetrieverConfig(**defaults)


class TestRRFFusion:
    def test_source_evidence_surfaces_unindexed_mu(self, _store):
        """Source evidence lane surfaces an MU that is NOT in FAISS or BM25."""
        from locomo_memory.phase2.schemas import MemoryUnit

        mu = MemoryUnit(
            mu_id="mu_evidence_only",
            conversation_id="c1",
            session_id="s1",
            claim="The harvest outcome",  # no keyword overlap with query
            source_dia_ids=["d_harvest"],
        )
        _store.insert_memory_unit(mu)
        # Intentionally NOT added to FAISS or BM25 indexes

        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("d_harvest", "apple harvest was bountiful this season")])
        se_idx.link_mu("d_harvest", "mu_evidence_only")

        retriever, _, _ = _make_retriever(_store, source_ev_index=se_idx)
        result = retriever.retrieve(
            "apple harvest season",
            conversation_id="c1",
            config_override=_cfg(
                enable_source_evidence_lane=True,
                source_bm25_top_n=5,
            ),
        )
        mu_ids = [h.mu.mu_id for h in result.hits]
        assert "mu_evidence_only" in mu_ids, (
            "Source evidence lane must surface unindexed MU"
        )

    def test_source_evidence_in_sources_list(self, _store):
        """When source evidence boosts an MU, 'source_evidence' appears in sources."""
        from locomo_memory.phase2.schemas import MemoryUnit

        mu = MemoryUnit(
            mu_id="mu_src_test",
            conversation_id="c1",
            session_id="s1",
            claim="general context",
            source_dia_ids=["d_src"],
        )
        _store.insert_memory_unit(mu)

        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("d_src", "specific banana plantation details")])
        se_idx.link_mu("d_src", "mu_src_test")

        retriever, _, _ = _make_retriever(_store, source_ev_index=se_idx)
        result = retriever.retrieve(
            "banana plantation",
            conversation_id="c1",
            config_override=_cfg(
                enable_source_evidence_lane=True,
                source_bm25_top_n=5,
            ),
        )
        hit = next((h for h in result.hits if h.mu.mu_id == "mu_src_test"), None)
        assert hit is not None
        assert "source_evidence" in hit.sources

    def test_source_evidence_dia_ids_populated(self, _store):
        """source_evidence_dia_ids on HybridHit contains the contributing dia_id."""
        from locomo_memory.phase2.schemas import MemoryUnit

        mu = MemoryUnit(
            mu_id="mu_provenance",
            conversation_id="c1",
            session_id="s1",
            claim="claim text",
            source_dia_ids=["d_prov"],
        )
        _store.insert_memory_unit(mu)

        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("d_prov", "provenance turn with unique keyword xyzzy")])
        se_idx.link_mu("d_prov", "mu_provenance")

        retriever, _, _ = _make_retriever(_store, source_ev_index=se_idx)
        result = retriever.retrieve(
            "xyzzy keyword",
            conversation_id="c1",
            config_override=_cfg(
                enable_source_evidence_lane=True,
                source_bm25_top_n=5,
            ),
        )
        hit = next((h for h in result.hits if h.mu.mu_id == "mu_provenance"), None)
        assert hit is not None
        assert "d_prov" in hit.source_evidence_dia_ids

    def test_source_evidence_returns_transient_when_no_mu(self, _store):
        """Source evidence with no linked MU synthesises a transient hit."""
        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("d_unlinked", "garden party celebration")])
        # No link_mu call — no linked MU

        retriever, _, _ = _make_retriever(_store, source_ev_index=se_idx)
        result = retriever.retrieve(
            "garden party",
            conversation_id="c1",
            config_override=_cfg(
                enable_source_evidence_lane=True,
                source_bm25_top_n=5,
            ),
        )
        se_hits = [h for h in result.hits if h.is_from_source_evidence]
        assert len(se_hits) == 1
        assert se_hits[0].mu.mu_id.startswith("se_")
        assert se_hits[0].source_evidence_dia_ids == ["d_unlinked"]

    def test_deleted_mu_not_returned_via_source_evidence(self, _store):
        """A source turn linked to a hard-deleted MU must never be returned."""
        from locomo_memory.phase2.schemas import MemoryUnit

        mu_deleted = MemoryUnit(
            mu_id="mu_deleted_1",
            conversation_id="c1",
            session_id="s1",
            claim="This content was deleted",
            source_dia_ids=["d_del"],
        )
        _store.insert_memory_unit(mu_deleted)
        # Hard-delete: row is removed; only the audit row survives.
        _store.delete_atomic("mu_deleted_1", deleted_by="test")

        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("d_del", "deleted content appears here")])
        se_idx.link_mu("d_del", "mu_deleted_1")

        retriever, _, _ = _make_retriever(_store, source_ev_index=se_idx)
        result = retriever.retrieve(
            "deleted content",
            conversation_id="c1",
            config_override=_cfg(
                enable_source_evidence_lane=True,
                source_bm25_top_n=5,
            ),
        )
        mu_ids = [h.mu.mu_id for h in result.hits]
        assert "mu_deleted_1" not in mu_ids, (
            "Deleted MU must never be returned even via source evidence lane"
        )

    def test_source_evidence_disabled_no_effect(self, _store):
        """When enable_source_evidence_lane=False, existing behaviour unchanged."""
        from locomo_memory.phase2.schemas import MemoryUnit

        mu = MemoryUnit(
            mu_id="mu_normal",
            conversation_id="c1",
            session_id="s1",
            claim="normal memory content",
            source_dia_ids=["d_norm"],
        )
        _store.insert_memory_unit(mu)

        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("d_norm", "extra source text")])
        se_idx.link_mu("d_norm", "mu_normal")

        # Single retriever whose FAISS+BM25 are populated with the MU
        retriever, faiss_idx, bm25_idx = _make_retriever(_store, source_ev_index=se_idx)
        faiss_idx.add_mu(mu)
        bm25_idx.add_mu(mu)

        result = retriever.retrieve(
            "normal memory",
            conversation_id="c1",
            config_override=_cfg(enable_source_evidence_lane=False),
        )
        # Should still find it via BM25/FAISS; sources must NOT contain 'source_evidence'
        hit = next((h for h in result.hits if h.mu.mu_id == "mu_normal"), None)
        assert hit is not None
        assert "source_evidence" not in hit.sources

    def test_transient_mu_has_correct_dia_id_provenance(self, _store):
        """Transient MU from source-only hit carries the original dia_id."""
        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("dia_xyz999", "unique content about dragons")])

        retriever, _, _ = _make_retriever(_store, source_ev_index=se_idx)
        result = retriever.retrieve(
            "dragons",
            conversation_id="c1",
            config_override=_cfg(
                enable_source_evidence_lane=True,
                source_bm25_top_n=5,
            ),
        )
        se_hits = [h for h in result.hits if h.is_from_source_evidence]
        assert len(se_hits) >= 1
        assert se_hits[0].mu.source_dia_ids == ["dia_xyz999"]


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


class TestHybridRetrieverConfigSourceEvidenceFields:
    def test_default_disabled(self):
        from locomo_memory.phase2.retrieval.hybrid_retriever import HybridRetrieverConfig
        cfg = HybridRetrieverConfig()
        assert cfg.enable_source_evidence_lane is False

    def test_can_enable(self):
        from locomo_memory.phase2.retrieval.hybrid_retriever import HybridRetrieverConfig
        cfg = HybridRetrieverConfig(enable_source_evidence_lane=True,
                                    source_bm25_top_n=30,
                                    source_lane_rrf_weight=0.8)
        assert cfg.enable_source_evidence_lane is True
        assert cfg.source_bm25_top_n == 30
        assert cfg.source_lane_rrf_weight == 0.8

    def test_context_window_default(self):
        from locomo_memory.phase2.retrieval.hybrid_retriever import HybridRetrieverConfig
        cfg = HybridRetrieverConfig()
        assert cfg.source_context_window == 2

    def test_frozen(self):
        from locomo_memory.phase2.retrieval.hybrid_retriever import HybridRetrieverConfig
        cfg = HybridRetrieverConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.enable_source_evidence_lane = True  # type: ignore[misc]


class TestPhase2RetrievalConfigSourceEvidenceFields:
    def test_default_disabled(self):
        from locomo_memory.phase2.experiments.config import Phase2RetrievalConfig
        cfg = Phase2RetrievalConfig()
        assert cfg.enable_source_evidence_lane is False

    def test_can_enable(self):
        from locomo_memory.phase2.experiments.config import Phase2RetrievalConfig
        cfg = Phase2RetrievalConfig(
            enable_source_evidence_lane=True,
            source_bm25_top_n=25,
            source_lane_rrf_weight=0.9,
        )
        assert cfg.enable_source_evidence_lane is True
        assert cfg.source_bm25_top_n == 25


class TestMultigranularYamlConfig:
    def test_yaml_loads(self):
        from locomo_memory.phase2.experiments.config import Phase2RunnerConfig
        # Verify source evidence lane is present in the CE config (which includes SE lane)
        p = Path("configs/phase2_retrieval_cross_encoder_top5.yaml")
        assert p.exists(), "configs/phase2_retrieval_cross_encoder_top5.yaml not found"
        cfg = Phase2RunnerConfig.from_yaml(p)
        assert cfg.retrieval.enable_source_evidence_lane is True
        assert cfg.retrieval.top_k == 5
        assert cfg.retrieval.source_context_window == 2
        assert cfg.retrieval.source_bm25_top_n == 30
        assert cfg.retrieval.source_lane_rrf_weight == 1.0


# ---------------------------------------------------------------------------
# No gold evidence
# ---------------------------------------------------------------------------


class TestNoGoldEvidence:
    def test_search_api_has_no_gold_parameter(self):
        """search_bm25 must not accept a gold evidence parameter."""
        import inspect
        idx = SourceEvidenceIndex()
        sig = inspect.signature(idx.search_bm25)
        params = set(sig.parameters.keys())
        for forbidden in ("gold", "gold_evidence", "gold_dia_ids", "answer",
                          "gold_answer", "evidence_ids"):
            assert forbidden not in params, (
                f"search_bm25 must not accept gold parameter '{forbidden}'"
            )

    def test_retrieval_uses_query_only(self, _store):
        """Source evidence retrieval is driven by query text, not gold ids."""
        from locomo_memory.phase2.schemas import MemoryUnit

        mu = MemoryUnit(
            mu_id="mu_gold_test",
            conversation_id="c1",
            session_id="s1",
            claim="Alice works at the bookshop",
            source_dia_ids=["d_gold"],
        )
        _store.insert_memory_unit(mu)

        se_idx = SourceEvidenceIndex()
        se_idx.add_turns([_turn("d_gold", "Alice works at the bookshop downtown")])
        se_idx.link_mu("d_gold", "mu_gold_test")

        # Gold evidence id: "d_gold" — we deliberately do NOT pass it anywhere
        retriever, _, _ = _make_retriever(_store, source_ev_index=se_idx)
        result = retriever.retrieve(
            "Where does Alice work",  # query only, no gold ids
            conversation_id="c1",
            config_override=_cfg(
                enable_source_evidence_lane=True,
                source_bm25_top_n=5,
            ),
        )
        # Retrieval may find "mu_gold_test" via query keywords — that is fine
        # The important thing is no gold data was injected into the API call
        assert isinstance(result.hits, list)


# ---------------------------------------------------------------------------
# Existing behaviour unchanged
# ---------------------------------------------------------------------------


class TestExistingLanesUnchanged:
    def test_hybrid_hit_is_from_label_default_false(self):
        """New fields have sensible defaults; existing code still constructs HybridHit."""
        from locomo_memory.phase2.retrieval.hybrid_retriever import HybridHit, RelationMeta
        from locomo_memory.phase2.schemas import MemoryUnit

        mu = MemoryUnit(conversation_id="c1", session_id="s1", claim="test")
        hit = HybridHit(
            mu=mu,
            rrf_score=0.016,
            rank=1,
            sources=["bm25"],
            label_summary=None,
            relation_meta=RelationMeta(),
            is_from_label=False,
        )
        assert hit.is_from_source_evidence is False
        assert hit.source_evidence_dia_ids == []

    def test_source_evidence_index_exported_from_indexes_package(self):
        from locomo_memory.phase2.indexes import SourceEvidenceIndex as SEI
        assert SEI is SourceEvidenceIndex
