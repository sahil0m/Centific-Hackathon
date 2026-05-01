"""
Embedding generator with disk-based caching.

Uses sentence-transformers locally. The cache key is derived from the
model name and the chunk text so embeddings are reused across runs.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_MODELS = {
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
}


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None

        if model_name not in _SUPPORTED_MODELS:
            logger.warning(
                "Model '%s' is not in the supported list %s. Proceeding anyway.",
                model_name,
                _SUPPORTED_MODELS,
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return a (N, D) float32 array of embeddings for the given texts."""
        if not texts:
            return np.empty((0, self._get_dim()), dtype=np.float32)

        results = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = self._embed_batch(uncached_texts)
            for local_idx, global_idx in enumerate(uncached_indices):
                emb = new_embeddings[local_idx]
                results[global_idx] = emb
                self._save_to_cache(texts[global_idx], emb)

        return np.vstack(results).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Return a 1-D float32 embedding for a single query string."""
        return self.embed_texts([query])[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            logger.info("Loading sentence-transformers model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded.")
        return self._model

    def _get_dim(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        logger.info("Embedding %d texts in batches of %d", len(texts), self.batch_size)
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.model_name}||{text}".encode()).hexdigest()
        return h

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.npy"

    def _load_from_cache(self, text: str) -> np.ndarray | None:
        path = self._cache_path(self._cache_key(text))
        if path is None or not path.exists():
            return None
        try:
            return np.load(str(path))
        except Exception:
            return None

    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        path = self._cache_path(self._cache_key(text))
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(str(path), embedding)
        except Exception as exc:
            logger.warning("Failed to cache embedding: %s", exc)
