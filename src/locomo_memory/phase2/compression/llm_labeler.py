"""LLM-powered label builder for the compression pipeline.

Replaces the rule-based truncation in the default LabelBuilder with an
LLM-generated dense summary that preserves key entities, relationships,
and semantics for the compressed label.

The archived_entries record (raw full MU JSON) is still written identically —
this only improves what goes into compressed_labels.short_summary.

Falls back to truncation if the LLM call fails so the pipeline never stalls.
"""

from __future__ import annotations

import hashlib

from loguru import logger

from locomo_memory.phase2.ingestion.importance import TopicImportanceEstimator
from locomo_memory.phase2.lifecycle.engine import LabelBuilder
from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    MemoryUnit,
    new_archive_id,
)

_estimator = TopicImportanceEstimator()
_FALLBACK_MAX_CHARS = 120

_COMPRESSION_PROMPT = """\
Compress the following memory fact into a single dense sentence for long-term storage.

Fact: {claim}
Original context: {original_text}

Requirements:
1. Preserve ALL key information: full names, locations, dates, numbers, roles, relationships
2. State the type of fact (employment, preference, health, event, location, personal info, etc.)
3. Write it so a future search query on this topic will retrieve it
4. Maximum 100 words. Minimum 10 words.

Return ONLY the compressed label sentence. No prefix, no explanation, no quotes."""


class LLMLabeler(LabelBuilder):
    """LabelBuilder that generates compressed labels using an LLM call.

    Used by both CompressionService and LifecycleEngine so every compression
    path — manual and automatic — produces high-quality labels.

    Args:
        client: OpenRouterClient instance (already initialised with API key).
        model: model to use for compression. Should be fast and cheap since
               this runs inline during ingestion at capacity.
    """

    def __init__(
        self,
        client,
        model: str = "anthropic/claude-3-haiku",
    ) -> None:
        self._client = client
        self._model = model

    # ------------------------------------------------------------------
    # LabelBuilder contract
    # ------------------------------------------------------------------

    def build(self, mu: MemoryUnit) -> tuple[CompressedLabel, ArchivedEntry]:
        """Return (label, archive) with LLM-generated short_summary."""
        archive_entry_id = new_archive_id()

        short_summary = self._llm_summary(mu)

        label = CompressedLabel(
            archived_pointer=archive_entry_id,
            mu_id=mu.mu_id,
            conversation_id=mu.conversation_id,
            topic=_estimator.detect_topic(mu.claim),
            short_summary=short_summary,
            key_entities=_estimator.extract_entities(mu.claim),
            time_range=mu.timestamp,
            original_dia_ids=list(mu.source_dia_ids),
        )

        archive = ArchivedEntry(
            archived_entry_id=archive_entry_id,
            label_pointer=label.label_id,
            mu_id=mu.mu_id,
            conversation_id=mu.conversation_id,
            full_memory_unit_json=mu.model_dump_json(),
            full_original_text=mu.original_text,
        )

        return label, archive

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _llm_summary(self, mu: MemoryUnit) -> str:
        """Call the LLM to produce a compressed label; fall back to truncation."""
        try:
            prompt = _COMPRESSION_PROMPT.format(
                claim=mu.claim,
                original_text=(mu.original_text or mu.claim)[:400],
            )
            cache_key = hashlib.sha256(
                (mu.mu_id + mu.claim).encode()
            ).hexdigest()[:20]

            response = self._client.chat_completion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                prompt_template_version="compress_label_v1",
                cache_input=cache_key,
                max_tokens=130,
                temperature=0.0,
            )
            summary = response.content.strip().strip('"').strip("'")
            if len(summary) >= 10:
                logger.debug(
                    "LLMLabeler: mu={} → '{}'", mu.mu_id, summary[:90]
                )
                return summary
            logger.warning(
                "LLMLabeler: LLM returned too-short label for mu={}, using fallback",
                mu.mu_id,
            )
        except Exception as exc:
            logger.warning(
                "LLMLabeler: LLM call failed for mu={}, using truncation fallback: {}",
                mu.mu_id, exc,
            )
        return mu.claim[:_FALLBACK_MAX_CHARS]


__all__ = ["LLMLabeler"]
