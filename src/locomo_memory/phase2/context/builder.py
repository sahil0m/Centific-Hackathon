"""Context Builder — Phase 2 Milestone 9.

Transforms a ranked list of :class:`~locomo_memory.phase2.retrieval.hybrid_retriever.HybridHit`
objects into a structured, LLM-ready prompt context with four labelled sections:

ACTIVE MEMORIES      — current, reliable facts (primary evidence)
HISTORICAL CONTEXT   — facts that have since been superseded by newer ones
CONFLICTING          — facts whose content conflicts with another retrieved fact
RESTORED             — facts that were in the compressed tier and fetched from archive

Section assignment priority (first match wins):
1. ``is_from_label=True`` → RESTORED
2. ``relation_meta.superseded_by`` non-empty → HISTORICAL CONTEXT
3. ``relation_meta.conflicts_with`` non-empty → CONFLICTING
4. Otherwise → ACTIVE MEMORIES

The builder also returns an ``evidence_tokens`` set used by the
:class:`~locomo_memory.phase2.context.guard.ResponseGuard` for grounding checks.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

from locomo_memory.phase2.retrieval.hybrid_retriever import HybridHit
from locomo_memory.phase2.store.sqlite_store import MemoryStore

# ---------------------------------------------------------------------------
# Section names (constants so tests can reference them)
# ---------------------------------------------------------------------------

SECTION_ACTIVE = "active"
SECTION_SUPERSEDED = "superseded"
SECTION_CONFLICTED = "conflicted"
SECTION_RESTORED = "restored"

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are answering a question about a long multi-session conversation using structured memory evidence.

Rules:
1. Use ONLY the evidence provided below.
2. Trust ACTIVE MEMORIES first.
3. For HISTORICAL CONTEXT entries marked SUPERSEDED, prefer the newer fact they were replaced by.
4. For CONFLICTING memories, acknowledge the uncertainty explicitly.
5. For RESTORED entries, use the full claim text provided.
6. If no evidence supports the answer, reply exactly: "No information available."
7. Give a short, direct answer. Do not explain your reasoning or cite evidence IDs.\
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ContextEntry:
    """One indexed evidence line in the built context."""

    index: int
    mu_id: str
    claim: str
    section: str
    """One of SECTION_ACTIVE / SECTION_SUPERSEDED / SECTION_CONFLICTED / SECTION_RESTORED."""

    confidence: float
    source_speaker: str
    source_session: str
    source_timestamp: str | None

    superseded_by_ids: list[str] = field(default_factory=list)
    conflicts_with_ids: list[str] = field(default_factory=list)
    related_to_ids: list[str] = field(default_factory=list)
    label_summary: str | None = None

    @property
    def source_info(self) -> str:
        """Human-readable source line."""
        parts: list[str] = []
        if self.source_speaker:
            parts.append(f"Speaker: {self.source_speaker}")
        if self.source_session:
            parts.append(f"Session: {self.source_session}")
        if self.source_timestamp:
            parts.append(f"Date: {self.source_timestamp}")
        parts.append(f"Confidence: {self.confidence:.2f}")
        return " | ".join(parts)


@dataclass(slots=True)
class BuiltContext:
    """Fully rendered context ready for injection into the LLM prompt."""

    query: str
    entries: list[ContextEntry]
    rendered_text: str
    """The formatted evidence block to be placed in the LLM user message."""
    system_prompt: str
    """The system instruction for the answer LLM."""
    evidence_tokens: frozenset[str]
    """Normalized token set across all evidence claims — used by ResponseGuard."""

    # Convenience flags
    has_active: bool
    has_superseded: bool
    has_conflicted: bool
    has_restored: bool

    @property
    def active_entries(self) -> list[ContextEntry]:
        return [e for e in self.entries if e.section == SECTION_ACTIVE]

    @property
    def superseded_entries(self) -> list[ContextEntry]:
        return [e for e in self.entries if e.section == SECTION_SUPERSEDED]

    @property
    def conflicted_entries(self) -> list[ContextEntry]:
        return [e for e in self.entries if e.section == SECTION_CONFLICTED]

    @property
    def restored_entries(self) -> list[ContextEntry]:
        return [e for e in self.entries if e.section == SECTION_RESTORED]

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def mu_ids(self) -> list[str]:
        return [e.mu_id for e in self.entries]


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Transform HybridHit objects into a structured LLM prompt context.

    Args:
        store: SQLite source of truth used to resolve mu_ids for superseded-by
            and conflicts-with claims so the rendered text is human-readable
            rather than just showing raw IDs.
        max_entries: maximum number of evidence entries to include. Entries
            beyond this limit are silently dropped (already ranked).
        system_prompt: override the default system prompt.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        max_entries: int = 10,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.store = store
        self.max_entries = max_entries
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Primary build
    # ------------------------------------------------------------------

    def build(self, query: str, hits: list[HybridHit]) -> BuiltContext:
        """Build a :class:`BuiltContext` from a list of retrieved hits.

        Args:
            query: the original user question (embedded in the rendered text).
            hits: ranked list of :class:`HybridHit` from the hybrid retriever.

        Returns:
            :class:`BuiltContext` with rendered text, system prompt, and metadata.
        """
        limited_hits = hits[: self.max_entries]

        # Build entries
        entries: list[ContextEntry] = []
        for idx, hit in enumerate(limited_hits, start=1):
            mu = hit.mu
            section = self._assign_section(hit)
            entry = ContextEntry(
                index=idx,
                mu_id=mu.mu_id,
                claim=mu.claim,
                section=section,
                confidence=mu.confidence,
                source_speaker=mu.source_speaker or "",
                source_session=mu.session_id,
                source_timestamp=mu.timestamp,
                superseded_by_ids=list(hit.relation_meta.superseded_by),
                conflicts_with_ids=list(hit.relation_meta.conflicts_with),
                related_to_ids=list(hit.relation_meta.related_to),
                label_summary=hit.label_summary,
            )
            entries.append(entry)

        # Render the evidence block
        rendered = self._render(query, entries)

        # Build evidence token set for grounding checks
        evidence_tokens = self._collect_evidence_tokens(entries)

        return BuiltContext(
            query=query,
            entries=entries,
            rendered_text=rendered,
            system_prompt=self.system_prompt,
            evidence_tokens=frozenset(evidence_tokens),
            has_active=any(e.section == SECTION_ACTIVE for e in entries),
            has_superseded=any(e.section == SECTION_SUPERSEDED for e in entries),
            has_conflicted=any(e.section == SECTION_CONFLICTED for e in entries),
            has_restored=any(e.section == SECTION_RESTORED for e in entries),
        )

    # ------------------------------------------------------------------
    # Section assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_section(hit: HybridHit) -> str:
        """Assign one section to a hit (priority: restored > superseded > conflicted > active)."""
        if hit.is_from_label:
            return SECTION_RESTORED
        if hit.relation_meta.superseded_by:
            return SECTION_SUPERSEDED
        if hit.relation_meta.conflicts_with:
            return SECTION_CONFLICTED
        return SECTION_ACTIVE

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, query: str, entries: list[ContextEntry]) -> str:
        """Render the structured evidence block."""
        if not entries:
            return "(No memory evidence retrieved for this query.)"

        lines: list[str] = []

        # Grouped sections in display order
        ordered_sections = [
            (SECTION_ACTIVE, "ACTIVE MEMORIES (use these first)"),
            (SECTION_RESTORED, "RESTORED FROM COMPRESSED (label match — full data retrieved)"),
            (SECTION_SUPERSEDED, "HISTORICAL CONTEXT (superseded, kept for reference)"),
            (SECTION_CONFLICTED, "CONFLICTING (treat with caution)"),
        ]

        for section_key, section_header in ordered_sections:
            section_entries = [e for e in entries if e.section == section_key]
            if not section_entries:
                continue
            lines.append(section_header + ":")
            for entry in section_entries:
                lines.append(self._render_entry(entry))
            lines.append("")  # blank line between sections

        # Trim trailing blank line
        while lines and lines[-1] == "":
            lines.pop()

        return "\n".join(lines)

    def _render_entry(self, entry: ContextEntry) -> str:
        """Render one indexed entry."""
        lines: list[str] = []
        lines.append(f"[{entry.index}] {entry.claim}")
        lines.append(f"    {entry.source_info}")

        if entry.section == SECTION_SUPERSEDED and entry.superseded_by_ids:
            labels = self._resolve_claims(entry.superseded_by_ids)
            lines.append(f"    SUPERSEDED BY: {'; '.join(labels)}")

        if entry.section == SECTION_CONFLICTED and entry.conflicts_with_ids:
            labels = self._resolve_claims(entry.conflicts_with_ids)
            lines.append(f"    CONFLICTS WITH: {'; '.join(labels)}")

        if entry.section == SECTION_RESTORED and entry.label_summary:
            lines.append(f"    Label matched: \"{entry.label_summary}\"")

        return "\n".join(lines)

    def _resolve_claims(self, mu_ids: list[str]) -> list[str]:
        """Look up claims for a list of mu_ids, falling back to the raw ID."""
        resolved: list[str] = []
        for mu_id in mu_ids:
            mu = self.store.get_memory_unit(mu_id)
            if mu is not None:
                resolved.append(mu.claim)
            else:
                resolved.append(mu_id)
        return resolved

    # ------------------------------------------------------------------
    # Evidence token extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_evidence_tokens(entries: list[ContextEntry]) -> set[str]:
        """Build a normalized token set across all evidence claims."""
        tokens: set[str] = set()
        for entry in entries:
            tokens.update(_tokenize(entry.claim))
            if entry.label_summary:
                tokens.update(_tokenize(entry.label_summary))
        return tokens

    # ------------------------------------------------------------------
    # Convenience: full prompt assembly
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, hits: list[HybridHit]) -> tuple[str, str, BuiltContext]:
        """Build context and return ``(system_prompt, user_message, built_context)``.

        ``user_message`` contains the evidence block followed by the question,
        ready to pass directly to an LLM client.
        """
        ctx = self.build(query, hits)
        user_message = f"{ctx.rendered_text}\n\nQuestion:\n{query}\n\nAnswer:"
        return ctx.system_prompt, user_message, ctx


# ---------------------------------------------------------------------------
# Tokenization helper (shared with ResponseGuard)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "through", "about", "and",
    "or", "but", "not", "what", "when", "where", "who", "which", "how",
    "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "my", "your", "his", "her", "our", "their",
})


def _tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "SECTION_ACTIVE",
    "SECTION_CONFLICTED",
    "SECTION_RESTORED",
    "SECTION_SUPERSEDED",
    "SYSTEM_PROMPT",
    "BuiltContext",
    "ContextBuilder",
    "ContextEntry",
]
