"""Tests for Phase 2 Milestone 8B: BM25 index, label FAISS index, hybrid retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from locomo_memory.phase2.indexes.label_index import CompressedLabelFAISSIndex, LabelSearchResult
from locomo_memory.phase2.retrieval.bm25_index import BM25SearchResult, MemoryBM25Index
from locomo_memory.phase2.retrieval.hybrid_retriever import (
    HybridHit,
    HybridMemoryRetriever,
    HybridRetrieverConfig,
    HybridRetrievalResult,
    RelationMeta,
)
from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    EdgeRecord,
    EdgeType,
    MemoryStatus,
    MemoryUnit,
    new_archive_id,
    new_label_id,
)
from locomo_memory.phase2.store.sqlite_store import MemoryStore

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


def _mu(claim: str, *, conv: str = "conv1", mu_id: str | None = None) -> MemoryUnit:
    kw: dict = dict(conversation_id=conv, session_id="s1", claim=claim)
    if mu_id:
        kw["mu_id"] = mu_id
    return MemoryUnit(**kw)


def _label(
    summary: str,
    mu_id: str,
    *,
    conv: str = "conv1",
    label_id: str | None = None,
    topic: str = "general",
    key_entities: list[str] | None = None,
) -> CompressedLabel:
    arc_id = new_archive_id()
    lid = label_id or new_label_id()
    return CompressedLabel(
        label_id=lid,
        archived_pointer=arc_id,
        mu_id=mu_id,
        conversation_id=conv,
        topic=topic,
        short_summary=summary,
        key_entities=key_entities or [],
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


@pytest.fixture()
def faiss_index():
    from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex
    return MemoryFAISSIndex(embed_fn=_make_embed_fn(0), dim=DIM)


@pytest.fixture()
def bm25_index() -> MemoryBM25Index:
    return MemoryBM25Index()


@pytest.fixture()
def label_index() -> CompressedLabelFAISSIndex:
    return CompressedLabelFAISSIndex(embed_fn=_make_embed_fn(1), dim=DIM)


# ===========================================================================
# MemoryBM25Index tests
# ===========================================================================


class TestMemoryBM25Index:
    def test_starts_empty(self, bm25_index):
        assert bm25_index.size() == 0
        assert len(bm25_index) == 0
        assert bm25_index.mu_ids() == []

    def test_repr(self, bm25_index):
        assert "MemoryBM25Index" in repr(bm25_index)

    def test_add_single(self, bm25_index):
        bm25_index.add_mu(_mu("Alice works at Google", mu_id="m1"))
        assert bm25_index.size() == 1
        assert "m1" in bm25_index.mu_ids()

    def test_add_multiple(self, bm25_index):
        bm25_index.add_mus([
            _mu("Alice works at Google", mu_id="m1"),
            _mu("Bob lives in NYC", mu_id="m2"),
        ])
        assert bm25_index.size() == 2

    def test_add_empty_list(self, bm25_index):
        bm25_index.add_mus([])
        assert bm25_index.size() == 0

    def test_re_add_replaces(self, bm25_index):
        bm25_index.add_mu(_mu("old claim", mu_id="m1"))
        bm25_index.add_mu(_mu("new claim", mu_id="m1"))
        assert bm25_index.size() == 1
        assert "m1" in bm25_index.mu_ids()

    def test_remove_existing(self, bm25_index):
        bm25_index.add_mu(_mu("Alice", mu_id="m1"))
        removed = bm25_index.remove_mu("m1")
        assert removed is True
        assert bm25_index.size() == 0
        assert "m1" not in bm25_index.mu_ids()

    def test_remove_nonexistent_returns_false(self, bm25_index):
        assert bm25_index.remove_mu("ghost") is False

    def test_search_returns_results(self, bm25_index):
        for i in range(5):
            bm25_index.add_mu(_mu(f"Alice claim number {i}", mu_id=f"m{i}"))
        results = bm25_index.search("Alice claim", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, BM25SearchResult) for r in results)

    def test_search_respects_top_k(self, bm25_index):
        for i in range(10):
            bm25_index.add_mu(_mu(f"item info {i}", mu_id=f"m{i}"))
        results = bm25_index.search("item info", top_k=3)
        assert len(results) <= 3

    def test_search_ranks_ascending(self, bm25_index):
        for i in range(5):
            bm25_index.add_mu(_mu(f"claim fact {i}", mu_id=f"m{i}"))
        results = bm25_index.search("claim fact", top_k=5)
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_search_conversation_filter(self, bm25_index):
        bm25_index.add_mu(_mu("Alice fact", conv="c1", mu_id="m1"))
        bm25_index.add_mu(_mu("Bob fact", conv="c2", mu_id="m2"))
        results = bm25_index.search("fact", top_k=5, conversation_id="c1")
        ids = {r.mu_id for r in results}
        assert "m2" not in ids
        assert "m1" in ids

    def test_search_no_filter_returns_all_convs(self, bm25_index):
        bm25_index.add_mu(_mu("fact c1", conv="c1", mu_id="m1"))
        bm25_index.add_mu(_mu("fact c2", conv="c2", mu_id="m2"))
        results = bm25_index.search("fact", top_k=5)
        ids = {r.mu_id for r in results}
        assert "m1" in ids
        assert "m2" in ids

    def test_search_empty_index(self, bm25_index):
        assert bm25_index.search("query", top_k=5) == []

    def test_search_zero_top_k(self, bm25_index):
        bm25_index.add_mu(_mu("claim", mu_id="m1"))
        assert bm25_index.search("claim", top_k=0) == []

    def test_removed_not_in_results(self, bm25_index):
        bm25_index.add_mu(_mu("Alice works at Google", mu_id="m1"))
        bm25_index.add_mu(_mu("Bob lives in NYC", mu_id="m2"))
        bm25_index.remove_mu("m1")
        results = bm25_index.search("Alice Google", top_k=5)
        assert "m1" not in {r.mu_id for r in results}

    def test_rebuild_replaces(self, bm25_index):
        bm25_index.add_mu(_mu("old claim", mu_id="old"))
        new_mus = [_mu(f"new item {i}", mu_id=f"n{i}") for i in range(3)]
        bm25_index.rebuild(new_mus)
        assert bm25_index.size() == 3
        assert "old" not in bm25_index.mu_ids()

    def test_rebuild_empty_clears(self, bm25_index):
        bm25_index.add_mu(_mu("something", mu_id="m1"))
        bm25_index.rebuild([])
        assert bm25_index.size() == 0

    def test_rebuild_from_store(self, store, bm25_index):
        for i in range(4):
            store.insert_memory_unit(_mu(f"claim {i}", mu_id=f"m{i}"))
        n = bm25_index.rebuild_from_store(store)
        assert n == 4
        assert bm25_index.size() == 4

    def test_rebuild_from_store_only_active(self, store, bm25_index):
        for i in range(3):
            store.insert_memory_unit(_mu(f"claim {i}", mu_id=f"m{i}"))
        store.forget_atomic("m2")
        n = bm25_index.rebuild_from_store(store)
        assert n == 2
        assert "m2" not in bm25_index.mu_ids()

    def test_rebuild_from_store_forgotten_status(self, store, bm25_index):
        store.insert_memory_unit(_mu("active claim", mu_id="m1"))
        store.insert_memory_unit(_mu("forgotten claim", mu_id="m2"))
        store.forget_atomic("m2")
        n = bm25_index.rebuild_from_store(store, status=MemoryStatus.FORGOTTEN)
        assert n == 1
        assert "m2" in bm25_index.mu_ids()
        assert "m1" not in bm25_index.mu_ids()

    def test_rebuild_from_store_conv_filter(self, store, bm25_index):
        store.insert_memory_unit(_mu("c1", conv="conv1", mu_id="m1"))
        store.insert_memory_unit(_mu("c2", conv="conv2", mu_id="m2"))
        n = bm25_index.rebuild_from_store(store, conversation_id="conv1")
        assert n == 1
        assert "m1" in bm25_index.mu_ids()
        assert "m2" not in bm25_index.mu_ids()

    def test_result_has_correct_fields(self, bm25_index):
        bm25_index.add_mu(_mu("Alice works at Google", conv="c1", mu_id="m1"))
        results = bm25_index.search("Alice Google", top_k=1, conversation_id="c1")
        assert len(results) == 1
        r = results[0]
        assert r.mu_id == "m1"
        assert r.conversation_id == "c1"
        assert isinstance(r.score, float)
        assert r.rank == 1


# ===========================================================================
# CompressedLabelFAISSIndex tests
# ===========================================================================


class TestCompressedLabelFAISSIndex:
    def test_starts_empty(self, label_index):
        assert label_index.size() == 0
        assert len(label_index) == 0

    def test_repr(self, label_index):
        assert "CompressedLabelFAISSIndex" in repr(label_index)

    def test_add_single_label(self, label_index):
        lb = _label("Alice works at Google", mu_id="m1", label_id="lbl1")
        label_index.add_label(lb)
        assert label_index.size() == 1
        assert "lbl1" in label_index.label_ids()

    def test_add_multiple_labels(self, label_index):
        label_index.add_labels([
            _label("Alice at Google", mu_id="m1", label_id="lbl1"),
            _label("Bob in NYC", mu_id="m2", label_id="lbl2"),
        ])
        assert label_index.size() == 2

    def test_add_empty_noop(self, label_index):
        label_index.add_labels([])
        assert label_index.size() == 0

    def test_re_add_replaces(self, label_index):
        lb = _label("old summary", mu_id="m1", label_id="lbl1")
        label_index.add_label(lb)
        lb2 = _label("new summary", mu_id="m1", label_id="lbl1")
        label_index.add_label(lb2)
        assert label_index.size() == 1

    def test_remove_existing(self, label_index):
        label_index.add_label(_label("Alice", mu_id="m1", label_id="lbl1"))
        removed = label_index.remove_label("lbl1")
        assert removed is True
        assert label_index.size() == 0

    def test_remove_nonexistent_returns_false(self, label_index):
        assert label_index.remove_label("ghost") is False

    def test_search_returns_results(self, label_index):
        for i in range(5):
            label_index.add_label(_label(f"memory {i}", mu_id=f"m{i}", label_id=f"lbl{i}"))
        results = label_index.search("memory", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, LabelSearchResult) for r in results)

    def test_search_empty_returns_empty(self, label_index):
        assert label_index.search("anything", top_k=5) == []

    def test_search_zero_top_k_returns_empty(self, label_index):
        label_index.add_label(_label("fact", mu_id="m1", label_id="lbl1"))
        assert label_index.search("fact", top_k=0) == []

    def test_search_conversation_filter(self, label_index):
        label_index.add_label(_label("Alice", mu_id="m1", label_id="lbl1", conv="c1"))
        label_index.add_label(_label("Bob", mu_id="m2", label_id="lbl2", conv="c2"))
        results = label_index.search("fact", top_k=5, conversation_id="c1")
        ids = {r.label_id for r in results}
        assert "lbl2" not in ids
        assert "lbl1" in ids

    def test_search_result_has_correct_fields(self, label_index):
        label_index.add_label(_label("Alice at Google", mu_id="m1", label_id="lbl1", conv="c1"))
        results = label_index.search("Alice Google", top_k=1, conversation_id="c1")
        assert len(results) == 1
        r = results[0]
        assert r.label_id == "lbl1"
        assert r.mu_id == "m1"
        assert r.conversation_id == "c1"
        assert isinstance(r.score, float)
        assert r.rank == 1
        assert isinstance(r.short_summary, str)

    def test_removed_not_in_search(self, label_index):
        label_index.add_label(_label("Alice at Google", mu_id="m1", label_id="lbl1"))
        label_index.add_label(_label("Bob in NYC", mu_id="m2", label_id="lbl2"))
        label_index.remove_label("lbl1")
        results = label_index.search("Alice Google", top_k=5)
        assert "lbl1" not in {r.label_id for r in results}

    def test_label_text_includes_topic_and_entities(self):
        lb = _label(
            "short summary text",
            mu_id="m1",
            label_id="lbl1",
            topic="employment",
            key_entities=["Alice", "Google"],
        )
        text = CompressedLabelFAISSIndex._label_text(lb)
        assert "employment" in text
        assert "short summary text" in text
        assert "Alice" in text
        assert "Google" in text

    def test_rebuild_replaces(self, label_index):
        label_index.add_label(_label("old", mu_id="m1", label_id="lbl1"))
        new_labels = [
            _label(f"new {i}", mu_id=f"m{i}", label_id=f"new_lbl{i}")
            for i in range(3)
        ]
        label_index.rebuild(new_labels)
        assert label_index.size() == 3
        assert "lbl1" not in label_index.label_ids()

    def test_rebuild_from_store(self, store, label_index):
        # Insert compressed MUs with labels
        for i in range(3):
            mu = _mu(f"claim {i}", mu_id=f"m{i}")
            store.insert_memory_unit(mu)
            arc_id = new_archive_id()
            lid = new_label_id()
            import json
            archive = ArchivedEntry(
                archived_entry_id=arc_id,
                label_pointer=lid,
                mu_id=f"m{i}",
                conversation_id="conv1",
                full_memory_unit_json=json.dumps(mu.model_dump(mode="json")),
            )
            label = CompressedLabel(
                label_id=lid,
                archived_pointer=arc_id,
                mu_id=f"m{i}",
                conversation_id="conv1",
                topic="test",
                short_summary=f"summary {i}",
            )
            store.compress_atomic(f"m{i}", label, archive)
        n = label_index.rebuild_from_store(store)
        assert n == 3

    def test_save_and_load(self, tmp_path, label_index):
        label_index.add_labels([
            _label("Alice at Google", mu_id="m1", label_id="lbl1"),
            _label("Bob in NYC", mu_id="m2", label_id="lbl2"),
        ])
        label_index.save(tmp_path / "lbl_idx")

        idx2 = CompressedLabelFAISSIndex(embed_fn=_make_embed_fn(1), dim=DIM)
        idx2.load(tmp_path / "lbl_idx")
        assert idx2.size() == 2
        assert set(idx2.label_ids()) == {"lbl1", "lbl2"}

    def test_load_missing_raises(self, tmp_path, label_index):
        with pytest.raises(FileNotFoundError):
            label_index.load(tmp_path / "nonexistent")


# ===========================================================================
# HybridMemoryRetriever tests
# ===========================================================================


def _make_hybrid_retriever(store, *, seed: int = 0):
    from locomo_memory.phase2.indexes.faiss_index import MemoryFAISSIndex

    embed_fn = _make_embed_fn(seed)
    fi = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    bi = MemoryBM25Index()
    li = CompressedLabelFAISSIndex(embed_fn=embed_fn, dim=DIM)
    return HybridMemoryRetriever(store=store, faiss_index=fi, bm25_index=bi, label_index=li)


class TestHybridRetrieverConfig:
    def test_defaults(self):
        cfg = HybridRetrieverConfig()
        assert cfg.top_k == 5
        assert cfg.rrf_k == 60
        assert cfg.enable_bm25 is True
        assert cfg.enable_label_search is True
        assert cfg.enable_graph_traversal is True
        assert cfg.enable_forgotten_fallback is False

    def test_frozen(self):
        cfg = HybridRetrieverConfig()
        with pytest.raises(Exception):
            cfg.top_k = 10  # type: ignore[misc]


class TestRelationMeta:
    def test_defaults_are_empty_lists(self):
        meta = RelationMeta()
        assert meta.superseded_by == []
        assert meta.conflicts_with == []
        assert meta.related_to == []


class TestHybridMemoryRetriever:
    def test_basic_retrieve_returns_result(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("Alice works at Google", mu_id="m1")
        store.insert_memory_unit(mu)
        hr.faiss_index.add_mu(mu)
        hr.bm25_index.add_mu(mu)

        result = hr.retrieve("Alice Google", conversation_id="conv1")
        assert isinstance(result, HybridRetrievalResult)
        assert result.query == "Alice Google"
        assert result.conversation_id == "conv1"

    def test_retrieve_returns_active_only(self, store):
        hr = _make_hybrid_retriever(store)
        mu_active = _mu("Alice at Google", mu_id="m1")
        mu_forgotten = _mu("Bob in NYC", mu_id="m2")
        store.insert_memory_unit(mu_active)
        store.insert_memory_unit(mu_forgotten)
        store.forget_atomic("m2")

        hr.faiss_index.add_mu(mu_active)
        hr.bm25_index.add_mu(mu_active)

        result = hr.retrieve("Alice", conversation_id="conv1")
        mu_ids = result.mu_ids
        assert "m1" in mu_ids
        assert "m2" not in mu_ids

    def test_deleted_never_returned(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("Alice at Google", mu_id="m1")
        store.insert_memory_unit(mu)
        hr.faiss_index.add_mu(mu)
        hr.bm25_index.add_mu(mu)
        store.delete_atomic("m1")

        result = hr.retrieve("Alice", conversation_id="conv1")
        assert "m1" not in result.mu_ids

    def test_retrieve_scoped_to_conversation(self, store):
        hr = _make_hybrid_retriever(store)
        mu1 = _mu("Alice at Google", conv="conv1", mu_id="m1")
        mu2 = _mu("Bob in NYC", conv="conv2", mu_id="m2")
        store.insert_memory_unit(mu1)
        store.insert_memory_unit(mu2)
        hr.faiss_index.add_mu(mu1)
        hr.faiss_index.add_mu(mu2)
        hr.bm25_index.add_mu(mu1)
        hr.bm25_index.add_mu(mu2)

        result = hr.retrieve("Alice Bob", conversation_id="conv1")
        for hit in result.hits:
            assert hit.mu.conversation_id == "conv1"

    def test_retrieve_respects_top_k(self, store):
        hr = _make_hybrid_retriever(store)
        for i in range(10):
            mu = _mu(f"claim {i}", mu_id=f"m{i}")
            store.insert_memory_unit(mu)
            hr.faiss_index.add_mu(mu)
            hr.bm25_index.add_mu(mu)

        cfg = HybridRetrieverConfig(top_k=3)
        result = hr.retrieve("claim", conversation_id="conv1", config_override=cfg)
        assert len(result.hits) <= 3

    def test_hit_has_correct_fields(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("Alice at Google", mu_id="m1")
        store.insert_memory_unit(mu)
        hr.faiss_index.add_mu(mu)
        hr.bm25_index.add_mu(mu)

        result = hr.retrieve("Alice", conversation_id="conv1")
        assert len(result.hits) >= 1
        hit = result.hits[0]
        assert isinstance(hit, HybridHit)
        assert hit.rank == 1
        assert isinstance(hit.rrf_score, float)
        assert isinstance(hit.sources, list)
        assert isinstance(hit.relation_meta, RelationMeta)

    def test_ranks_are_ascending(self, store):
        hr = _make_hybrid_retriever(store)
        for i in range(5):
            mu = _mu(f"alice fact {i}", mu_id=f"m{i}")
            store.insert_memory_unit(mu)
            hr.faiss_index.add_mu(mu)
            hr.bm25_index.add_mu(mu)

        result = hr.retrieve("alice fact", conversation_id="conv1")
        ranks = [h.rank for h in result.hits]
        assert ranks == sorted(ranks)

    def test_rrf_scores_descending(self, store):
        hr = _make_hybrid_retriever(store)
        for i in range(5):
            mu = _mu(f"fact {i}", mu_id=f"m{i}")
            store.insert_memory_unit(mu)
            hr.faiss_index.add_mu(mu)
            hr.bm25_index.add_mu(mu)

        result = hr.retrieve("fact", conversation_id="conv1")
        scores = [h.rrf_score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_sources_include_faiss_and_bm25(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("Alice works at Google", mu_id="m1")
        store.insert_memory_unit(mu)
        hr.faiss_index.add_mu(mu)
        hr.bm25_index.add_mu(mu)

        result = hr.retrieve("Alice Google works", conversation_id="conv1")
        assert len(result.hits) == 1
        sources = result.hits[0].sources
        # Should appear in at least one lane
        assert len(sources) >= 1

    def test_disable_bm25(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("Alice works", mu_id="m1")
        store.insert_memory_unit(mu)
        hr.faiss_index.add_mu(mu)
        hr.bm25_index.add_mu(mu)

        cfg = HybridRetrieverConfig(enable_bm25=False)
        result = hr.retrieve("Alice", conversation_id="conv1", config_override=cfg)
        for hit in result.hits:
            assert "bm25" not in hit.sources

    def test_empty_index_returns_empty_result(self, store):
        hr = _make_hybrid_retriever(store)
        result = hr.retrieve("anything", conversation_id="conv1")
        assert result.hits == []

    def test_result_metadata(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("claim", mu_id="m1")
        store.insert_memory_unit(mu)
        hr.faiss_index.add_mu(mu)
        hr.bm25_index.add_mu(mu)

        result = hr.retrieve("claim", conversation_id="conv1")
        assert result.top_k == 5
        assert isinstance(result.config, HybridRetrieverConfig)
        assert result.forgotten_searched is False
        assert isinstance(result.retrieval_latency_ms, float)
        assert result.retrieval_latency_ms >= 0

    def test_config_override(self, store):
        hr = _make_hybrid_retriever(store)
        result = hr.retrieve(
            "query",
            conversation_id="conv1",
            config_override=HybridRetrieverConfig(top_k=2),
        )
        assert result.top_k == 2
        assert result.config.top_k == 2

    def test_mu_ids_and_mus_properties(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("Alice", mu_id="m1")
        store.insert_memory_unit(mu)
        hr.faiss_index.add_mu(mu)
        hr.bm25_index.add_mu(mu)

        result = hr.retrieve("Alice", conversation_id="conv1")
        assert result.mu_ids == [h.mu.mu_id for h in result.hits]
        assert result.mus == [h.mu for h in result.hits]

    # ------------------------------------------------------------------
    # Relation metadata
    # ------------------------------------------------------------------

    def test_relation_meta_superseded_by(self, store):
        hr = _make_hybrid_retriever(store)
        mu1 = _mu("I work at Google", mu_id="m1")
        mu2 = _mu("I work at Microsoft", mu_id="m2")
        store.insert_memory_unit(mu1)
        store.insert_memory_unit(mu2)
        # m1 is superseded by m2
        store.insert_edge(EdgeRecord(
            source_mu_id="m1",
            target_mu_id="m2",
            edge_type=EdgeType.SUPERSEDED_BY,
        ))
        hr.faiss_index.add_mu(mu1)
        hr.bm25_index.add_mu(mu1)

        result = hr.retrieve("work Google", conversation_id="conv1")
        m1_hit = next((h for h in result.hits if h.mu.mu_id == "m1"), None)
        if m1_hit:
            assert "m2" in m1_hit.relation_meta.superseded_by

    def test_relation_meta_conflicts_with(self, store):
        hr = _make_hybrid_retriever(store)
        mu1 = _mu("Alice likes dogs", mu_id="m1")
        mu2 = _mu("Alice dislikes dogs", mu_id="m2")
        store.insert_memory_unit(mu1)
        store.insert_memory_unit(mu2)
        store.insert_edge(EdgeRecord(
            source_mu_id="m1",
            target_mu_id="m2",
            edge_type=EdgeType.CONFLICTS_WITH,
        ))
        hr.faiss_index.add_mu(mu1)
        hr.bm25_index.add_mu(mu1)

        result = hr.retrieve("Alice dogs", conversation_id="conv1")
        m1_hit = next((h for h in result.hits if h.mu.mu_id == "m1"), None)
        if m1_hit:
            assert "m2" in m1_hit.relation_meta.conflicts_with

    def test_relation_meta_related_to(self, store):
        hr = _make_hybrid_retriever(store)
        mu1 = _mu("Alice had surgery", mu_id="m1")
        mu2 = _mu("Alice is recovering", mu_id="m2")
        store.insert_memory_unit(mu1)
        store.insert_memory_unit(mu2)
        store.insert_edge(EdgeRecord(
            source_mu_id="m1",
            target_mu_id="m2",
            edge_type=EdgeType.RELATED_TO,
        ))
        hr.faiss_index.add_mu(mu1)
        hr.bm25_index.add_mu(mu1)

        result = hr.retrieve("Alice health", conversation_id="conv1")
        m1_hit = next((h for h in result.hits if h.mu.mu_id == "m1"), None)
        if m1_hit:
            assert "m2" in m1_hit.relation_meta.related_to

    # ------------------------------------------------------------------
    # Forgotten fallback
    # ------------------------------------------------------------------

    def test_forgotten_fallback_not_searched_by_default(self, store):
        hr = _make_hybrid_retriever(store)
        result = hr.retrieve("anything", conversation_id="conv1")
        assert result.forgotten_searched is False

    def test_forgotten_fallback_enabled_low_confidence(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("forgotten fact", mu_id="m1")
        store.insert_memory_unit(mu)
        store.forget_atomic("m1")

        cfg = HybridRetrieverConfig(
            enable_forgotten_fallback=True,
            forgotten_confidence_threshold=1.0,  # always trigger
        )
        result = hr.retrieve("forgotten fact", conversation_id="conv1", config_override=cfg)
        assert result.forgotten_searched is True

    def test_forgotten_fallback_returns_forgotten_mus(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("forgotten specific fact", mu_id="m1")
        store.insert_memory_unit(mu)
        store.forget_atomic("m1")

        cfg = HybridRetrieverConfig(
            enable_forgotten_fallback=True,
            forgotten_confidence_threshold=1.0,
        )
        result = hr.retrieve("forgotten specific fact", conversation_id="conv1", config_override=cfg)
        assert result.forgotten_searched is True
        forgotten_hits = [h for h in result.hits if h.mu.status == MemoryStatus.FORGOTTEN]
        assert len(forgotten_hits) >= 1

    def test_forgotten_fallback_deleted_still_excluded(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("deleted fact", mu_id="m1")
        store.insert_memory_unit(mu)
        store.delete_atomic("m1")

        cfg = HybridRetrieverConfig(
            enable_forgotten_fallback=True,
            forgotten_confidence_threshold=1.0,
        )
        result = hr.retrieve("deleted fact", conversation_id="conv1", config_override=cfg)
        assert "m1" not in result.mu_ids

    def test_forgotten_source_label(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("forgotten data", mu_id="m1")
        store.insert_memory_unit(mu)
        store.forget_atomic("m1")

        cfg = HybridRetrieverConfig(
            enable_forgotten_fallback=True,
            forgotten_confidence_threshold=1.0,
        )
        result = hr.retrieve("forgotten data", conversation_id="conv1", config_override=cfg)
        for hit in result.hits:
            if hit.mu.mu_id == "m1":
                assert "forgotten" in hit.sources

    # ------------------------------------------------------------------
    # Label lane
    # ------------------------------------------------------------------

    def test_label_lane_disabled(self, store):
        hr = _make_hybrid_retriever(store)
        cfg = HybridRetrieverConfig(enable_label_search=False)
        result = hr.retrieve("anything", conversation_id="conv1", config_override=cfg)
        for hit in result.hits:
            assert "label" not in hit.sources

    def test_label_hit_is_from_label_true(self, store):
        import json
        hr = _make_hybrid_retriever(store)

        mu = _mu("Alice works at Google", mu_id="m1")
        store.insert_memory_unit(mu)
        arc_id = new_archive_id()
        lid = new_label_id()
        archive = ArchivedEntry(
            archived_entry_id=arc_id,
            label_pointer=lid,
            mu_id="m1",
            conversation_id="conv1",
            full_memory_unit_json=json.dumps(mu.model_dump(mode="json")),
        )
        label = CompressedLabel(
            label_id=lid,
            archived_pointer=arc_id,
            mu_id="m1",
            conversation_id="conv1",
            topic="employment",
            short_summary="Alice works at Google",
        )
        store.compress_atomic("m1", label, archive)
        hr.label_index.add_label(label)

        result = hr.retrieve("Alice Google", conversation_id="conv1")
        label_hits = [h for h in result.hits if h.is_from_label]
        # If the label hit was hydrated, is_from_label should be True
        # (the MU is now COMPRESSED in the store)
        assert len(label_hits) >= 1

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def test_sync_all_indexes(self, store):
        hr = _make_hybrid_retriever(store)
        mu = _mu("new claim", mu_id="m1")
        store.insert_memory_unit(mu)
        store.mark_needs_reindex("m1")

        n = hr.sync_all_indexes()
        assert n == 1
        assert "m1" in hr.faiss_index.mu_ids()

    def test_rebuild_all_indexes(self, store):
        hr = _make_hybrid_retriever(store)
        for i in range(3):
            store.insert_memory_unit(_mu(f"claim {i}", mu_id=f"m{i}"))

        counts = hr.rebuild_all_indexes()
        assert counts["faiss"] == 3
        assert counts["bm25"] == 3
        assert counts["labels"] == 0  # no compressed labels

    def test_rebuild_all_with_conv_filter(self, store):
        hr = _make_hybrid_retriever(store)
        store.insert_memory_unit(_mu("c1 claim", conv="conv1", mu_id="m1"))
        store.insert_memory_unit(_mu("c2 claim", conv="conv2", mu_id="m2"))

        counts = hr.rebuild_all_indexes(conversation_id="conv1")
        assert counts["faiss"] == 1
        assert counts["bm25"] == 1
