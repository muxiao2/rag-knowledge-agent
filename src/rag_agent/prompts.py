"""Prompt templates for the RAG agent."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful knowledge-base assistant. Answer the user's question using "
    "ONLY the provided context. If the context does not contain the answer, say you "
    "don't know based on the knowledge base — do not make things up. Cite the source "
    "filenames you used. Answer in the same language as the question."
)

RAG_PROMPT_TEMPLATE = """{system}

# Context
{context}

# Question
{question}

# Answer
"""


def format_context(snippets: list[str]) -> str:
    """Join retrieved snippets into a numbered context block."""
    if not snippets:
        return "(no relevant context found)"
    return "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets))
