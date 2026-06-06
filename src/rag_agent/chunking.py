"""Lightweight, dependency-free text chunking.

Splitting text into overlapping chunks is the core preprocessing step for RAG.
This module intentionally avoids heavy dependencies so it is fast to import and
trivial to unit-test.

The algorithm splits text into paragraph units (windowing any paragraph that is
itself larger than ``chunk_size``), then greedily merges consecutive units into
chunks no larger than ``chunk_size``. When a chunk is emitted, trailing units
that fit within ``chunk_overlap`` characters are carried over to the start of the
next chunk so context is preserved across boundaries.
"""

from __future__ import annotations

import re

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SEPARATOR = "\n"


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def _window(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]


def _units(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_window(paragraph, chunk_size, chunk_overlap))
    return units


def _join(parts: list[str]) -> str:
    return _SEPARATOR.join(parts)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` characters.

    Consecutive chunks share up to ``chunk_overlap`` characters of context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    units = _units(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0  # length of the joined current parts

    for unit in units:
        added = len(unit) + (len(_SEPARATOR) if current else 0)
        if current and current_len + added > chunk_size:
            chunks.append(_join(current))
            # Carry over trailing units that fit within the overlap budget.
            carry: list[str] = []
            carry_len = 0
            for part in reversed(current):
                extra = len(part) + (len(_SEPARATOR) if carry else 0)
                if carry_len + extra > chunk_overlap:
                    break
                carry.insert(0, part)
                carry_len += extra
            current = carry
            current_len = carry_len

        sep = len(_SEPARATOR) if current else 0
        current.append(unit)
        current_len += len(unit) + sep

    if current:
        chunks.append(_join(current))
    return chunks
