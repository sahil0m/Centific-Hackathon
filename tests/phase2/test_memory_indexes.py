"""Tests for Phase 2 Milestone 8: MemoryFAISSIndex."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from locomo_memory.phase2.indexes.faiss_index import FAISSSearchResult, MemoryFAISSIndex
from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Deterministic dummy embedder (no sentence-transformers needed in tests)
# ---------------------------------------------------------------------------

DIM = 8


def _make_embed_fn(seed: int = 0):
    """Return a deterministic embed function that maps text → normalized (N, 8) vecs."""
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


def _index(seed: int = 0, **kwargs) -> MemoryFAISSIndex:
    return MemoryFAISSIndex(embed_fn=_make_embed_fn(seed), dim=DIM, **kwargs)


def _mu(claim: str, *, conv: str = "conv1", mu_id: str | None = None) -> MemoryUnit:
    kw: dict = dict(conversation_id=conv, session_id="s1", claim=claim)
    if mu_id:
        kw["mu_id"] = mu_id
    return MemoryUnit(**kw)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_index_starts_empty() -> None:
    idx = _index()
    assert idx.size() == 0
    assert len(idx) == 0
    assert idx.mu_ids() == []


def test_repr() -> None:
    idx = _index()
    assert "MemoryFAISSIndex" in repr(idx)


# ---------------------------------------------------------------------------
# add_mu / add_mus
# ---------------------------------------------------------------------------


def test_add_single_mu() -> None:
    idx = _index()
    idx.add_mu(_mu("Alice works at Google", mu_id="m1"))
    assert idx.size() == 1
    assert "m1" in idx.mu_ids()


def test_add_multiple_mus() -> None:
    idx = _index()
    idx.add_mus([
        _mu("Alice works at Google", mu_id="m1"),
        _mu("Bob lives in NYC", mu_id="m2"),
        _mu("Carol has a cat", mu_id="m3"),
    ])
    assert idx.size() == 3
    assert set(idx.mu_ids()) == {"m1", "m2", "m3"}


def test_add_empty_list_noop() -> None:
    idx = _index()
    idx.add_mus([])
    assert idx.size() == 0


def test_re_add_replaces_old_entry() -> None:
    idx = _index()
    mu = _mu("original claim", mu_id="m1")
    idx.add_mu(mu)
    assert idx.size() == 1

    mu2 = _mu("updated claim", mu_id="m1")
    idx.add_mu(mu2)
    # Old entry soft-deleted, new one added — size stays 1 (1 deleted + 1 live)
    assert idx.size() == 1
    assert "m1" in idx.mu_ids()


def test_add_preserves_conversation_id() -> None:
    idx = _index()
    idx.add_mu(_mu("claim", conv="conv42", mu_id="m1"))
    results = idx.search("claim", top_k=1)
    assert len(results) == 1
    assert results[0].conversation_id == "conv42"


# ---------------------------------------------------------------------------
# remove_mu
# ---------------------------------------------------------------------------


def test_remove_existing_mu() -> None:
    idx = _index()
    idx.add_mu(_mu("Alice works at Google", mu_id="m1"))
    removed = idx.remove_mu("m1")
    assert removed is True
    assert idx.size() == 0
    assert "m1" not in idx.mu_ids()


def test_remove_nonexistent_mu_returns_false() -> None:
    idx = _index()
    assert idx.remove_mu("ghost") is False


def test_removed_mu_not_in_search_results() -> None:
    idx = _index()
    idx.add_mus([
        _mu("Alice works at Google", mu_id="m1"),
        _mu("Bob lives in NYC", mu_id="m2"),
    ])
    idx.remove_mu("m1")
    results = idx.search("Alice Google", top_k=5)
    result_ids = {r.mu_id for r in results}
    assert "m1" not in result_ids


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_returns_results() -> None:
    idx = _index()
    idx.add_mus([_mu(f"claim number {i}", mu_id=f"m{i}") for i in range(5)])
    results = idx.search("claim number", top_k=3)
    assert len(results) <= 3
    assert all(isinstance(r, FAISSSearchResult) for r in results)


def test_search_respects_top_k() -> None:
    idx = _index()
    for i in range(10):
        idx.add_mu(_mu(f"item {i}", mu_id=f"m{i}"))
    results = idx.search("item", top_k=3)
    assert len(results) <= 3


def test_search_ranks_ascending() -> None:
    idx = _index()
    for i in range(5):
        idx.add_mu(_mu(f"claim {i}", mu_id=f"m{i}"))
    results = idx.search("claim", top_k=5)
    ranks = [r.rank for r in results]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_search_scores_descending() -> None:
    idx = _index()
    for i in range(5):
        idx.add_mu(_mu(f"fact {i}", mu_id=f"m{i}"))
    results = idx.search("fact", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_index_returns_empty() -> None:
    idx = _index()
    assert idx.search("anything", top_k=5) == []


def test_search_zero_top_k_returns_empty() -> None:
    idx = _index()
    idx.add_mu(_mu("a claim", mu_id="m1"))
    assert idx.search("a claim", top_k=0) == []


def test_search_conversation_filter() -> None:
    idx = _index()
    idx.add_mu(_mu("Alice works at Google", conv="conv1", mu_id="m1"))
    idx.add_mu(_mu("Bob lives in NYC", conv="conv2", mu_id="m2"))
    results = idx.search("query", top_k=5, conversation_id="conv1")
    ids = {r.mu_id for r in results}
    assert "m2" not in ids
    assert "m1" in ids


def test_search_no_filter_returns_all_conversations() -> None:
    idx = _index()
    idx.add_mu(_mu("claim1", conv="c1", mu_id="m1"))
    idx.add_mu(_mu("claim2", conv="c2", mu_id="m2"))
    results = idx.search("claim", top_k=5)
    ids = {r.mu_id for r in results}
    assert "m1" in ids
    assert "m2" in ids


def test_search_result_has_correct_fields() -> None:
    idx = _index()
    idx.add_mu(_mu("Alice works at Google", conv="conv1", mu_id="m1"))
    results = idx.search("Alice Google", top_k=1, conversation_id="conv1")
    assert len(results) == 1
    r = results[0]
    assert r.mu_id == "m1"
    assert r.conversation_id == "conv1"
    assert isinstance(r.score, float)
    assert r.rank == 1


# ---------------------------------------------------------------------------
# search_vector
# ---------------------------------------------------------------------------


def test_search_vector_same_as_search_text() -> None:
    embed_fn = _make_embed_fn(seed=99)
    idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    idx.add_mus([_mu(f"doc {i}", mu_id=f"m{i}") for i in range(5)])
    query = "doc 2"
    text_results = idx.search(query, top_k=3)
    vec = embed_fn([query])[0]
    vec_results = idx.search_vector(vec, top_k=3)
    assert [r.mu_id for r in text_results] == [r.mu_id for r in vec_results]


# ---------------------------------------------------------------------------
# rebuild / rebuild_from_store
# ---------------------------------------------------------------------------


def test_rebuild_replaces_index(store: MemoryStore) -> None:
    idx = _index()
    idx.add_mu(_mu("old claim", mu_id="old"))
    new_mus = [_mu(f"new {i}", mu_id=f"n{i}") for i in range(3)]
    idx.rebuild(new_mus)
    assert idx.size() == 3
    ids = set(idx.mu_ids())
    assert "old" not in ids
    assert "n0" in ids


def test_rebuild_empty_clears_index() -> None:
    idx = _index()
    idx.add_mu(_mu("something", mu_id="m1"))
    idx.rebuild([])
    assert idx.size() == 0


def test_rebuild_from_store(store: MemoryStore) -> None:
    for i in range(4):
        store.insert_memory_unit(_mu(f"claim {i}", mu_id=f"m{i}"))
    idx = _index()
    n = idx.rebuild_from_store(store)
    assert n == 4
    assert idx.size() == 4


def test_rebuild_from_store_only_active(store: MemoryStore) -> None:
    for i in range(3):
        store.insert_memory_unit(_mu(f"claim {i}", mu_id=f"m{i}"))
    store.forget_atomic("m2")
    idx = _index()
    n = idx.rebuild_from_store(store)
    assert n == 2  # m0, m1 only
    assert "m2" not in idx.mu_ids()


def test_rebuild_from_store_conversation_filter(store: MemoryStore) -> None:
    store.insert_memory_unit(_mu("c1 claim", conv="conv1", mu_id="m1"))
    store.insert_memory_unit(_mu("c2 claim", conv="conv2", mu_id="m2"))
    idx = _index()
    n = idx.rebuild_from_store(store, conversation_id="conv1")
    assert n == 1
    assert "m1" in idx.mu_ids()
    assert "m2" not in idx.mu_ids()


# ---------------------------------------------------------------------------
# sync_reindex
# ---------------------------------------------------------------------------


def test_sync_reindex_adds_active_mus(store: MemoryStore) -> None:
    mu = _mu("new claim", mu_id="m1")
    store.insert_memory_unit(mu)
    store.mark_needs_reindex("m1")

    idx = _index()
    n = idx.sync_reindex(store)
    assert n == 1
    assert "m1" in idx.mu_ids()
    # Flag should be cleared
    updated = store.get_memory_unit("m1")
    assert updated.needs_reindex is False


def test_sync_reindex_removes_non_active(store: MemoryStore) -> None:
    mu = _mu("claim", mu_id="m1")
    store.insert_memory_unit(mu)
    idx = _index()
    idx.add_mu(mu)

    store.forget_atomic("m1")  # now FORGOTTEN + needs_reindex=1
    n = idx.sync_reindex(store)
    assert n == 1
    assert "m1" not in idx.mu_ids()


# ---------------------------------------------------------------------------
# needs_compact
# ---------------------------------------------------------------------------


def test_needs_compact_false_initially() -> None:
    idx = _index(compact_threshold=0.3)
    for i in range(5):
        idx.add_mu(_mu(f"c{i}", mu_id=f"m{i}"))
    assert idx.needs_compact is False


def test_needs_compact_true_after_many_removals() -> None:
    idx = _index(compact_threshold=0.3)
    for i in range(10):
        idx.add_mu(_mu(f"c{i}", mu_id=f"m{i}"))
    for i in range(4):
        idx.remove_mu(f"m{i}")  # 4/10 = 0.4 > 0.3
    assert idx.needs_compact is True


def test_compact_clears_deleted_metadata() -> None:
    idx = _index(compact_threshold=0.3)
    for i in range(5):
        idx.add_mu(_mu(f"c{i}", mu_id=f"m{i}"))
    idx.remove_mu("m0")
    idx.remove_mu("m1")
    idx.compact()
    # _deleted set should be empty after compact
    assert len(idx._deleted) == 0


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_and_load(tmp_path: Path) -> None:
    idx = _index()
    idx.add_mus([
        _mu("Alice works at Google", conv="c1", mu_id="m1"),
        _mu("Bob lives in NYC", conv="c1", mu_id="m2"),
    ])
    idx.save(tmp_path / "idx")

    idx2 = _index()
    idx2.load(tmp_path / "idx")
    assert idx2.size() == 2
    assert set(idx2.mu_ids()) == {"m1", "m2"}


def test_load_missing_raises(tmp_path: Path) -> None:
    idx = _index()
    with pytest.raises(FileNotFoundError):
        idx.load(tmp_path / "nonexistent")


def test_save_load_preserves_search_results(tmp_path: Path) -> None:
    embed_fn = _make_embed_fn(seed=7)
    idx = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    idx.add_mus([_mu(f"item {i}", mu_id=f"m{i}") for i in range(5)])
    before = [r.mu_id for r in idx.search("item", top_k=3)]

    idx.save(tmp_path / "idx")
    idx2 = MemoryFAISSIndex(embed_fn=embed_fn, dim=DIM)
    idx2.load(tmp_path / "idx")
    after = [r.mu_id for r in idx2.search("item", top_k=3)]
    assert before == after


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")
