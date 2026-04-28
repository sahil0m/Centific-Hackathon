"""Prompt templates for answer generation."""

from __future__ import annotations

from locomo_memory.data.schemas import RetrievedChunk


ANSWER_PROMPT_TEMPLATE = """\
You are answering a question about a long multi-session conversation.

Rules:
1. Use only the retrieved conversation evidence.
2. If the evidence does not contain the answer, reply exactly: "No information available."
3. Give a short direct answer.
4. Do not explain your reasoning.
5. Do not mention evidence IDs in the final answer unless asked.

Retrieved evidence:
{retrieved_context}

Question:
{question}

Answer:"""


DEBUG_PROMPT_TEMPLATE = """\
You are answering a question about a long multi-session conversation.
Return JSON only, with no markdown fences.

Rules:
1. Use only the retrieved conversation evidence.
2. If the evidence does not contain the answer, set answer to "No information available."
3. Give a short direct answer.

Retrieved evidence:
{retrieved_context}

Question:
{question}

Return exactly this JSON structure:
{{"answer": "...", "used_evidence_ids": ["..."], "confidence": "low|medium|high"}}"""


def build_answer_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context = _format_context(retrieved_chunks)
    return ANSWER_PROMPT_TEMPLATE.format(retrieved_context=context, question=question)


def build_debug_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context = _format_context(retrieved_chunks)
    return DEBUG_PROMPT_TEMPLATE.format(retrieved_context=context, question=question)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "[No evidence retrieved]"
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Evidence {i}]\n{chunk.text}")
    return "\n\n".join(parts)


def count_prompt_tokens(prompt: str) -> int:
    """Approximate token count without calling tiktoken if not installed."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompt))
    except Exception:
        return len(prompt.split())
