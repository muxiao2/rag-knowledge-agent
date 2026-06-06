"""Ingestion pipeline: load -> chunk -> embed -> persist into Chroma."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document as LCDocument

from .chunking import chunk_text
from .config import Settings, get_settings
from .loaders import load_directory
from .vectorstore import get_vectorstore


def build_chunks(settings: Settings) -> list[LCDocument]:
    """Load every document under ``settings.docs_dir`` and split it into chunks."""
    source_docs = load_directory(settings.docs_dir)
    chunks: list[LCDocument] = []
    for doc in source_docs:
        pieces = chunk_text(
            doc.content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for idx, piece in enumerate(pieces):
            metadata = {**doc.metadata, "chunk": idx}
            chunks.append(LCDocument(page_content=piece, metadata=metadata))
    return chunks


def ingest(settings: Settings | None = None, reset: bool = False) -> int:
    """Ingest all documents into the vector store.

    Returns the number of chunks added. When ``reset`` is true the existing
    collection is cleared first so re-ingestion does not create duplicates.
    """
    settings = settings or get_settings()
    store = get_vectorstore(settings)

    if reset:
        try:
            store.delete_collection()
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        store = get_vectorstore(settings)

    chunks = build_chunks(settings)
    if not chunks:
        return 0

    store.add_documents(chunks)
    return len(chunks)


def ingest_path(path: Path, settings: Settings | None = None) -> int:
    """Ingest a single file or directory path (overrides ``docs_dir``)."""
    settings = settings or get_settings()
    path = Path(path)
    if path.is_dir():
        settings = settings.model_copy(update={"docs_dir": path})
        return ingest(settings)

    from .loaders import load_file

    doc = load_file(path)
    pieces = chunk_text(
        doc.content,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    docs = [
        LCDocument(page_content=p, metadata={**doc.metadata, "chunk": i})
        for i, p in enumerate(pieces)
    ]
    if not docs:
        return 0
    get_vectorstore(settings).add_documents(docs)
    return len(docs)
