"""Vector store wiring: Ollama embeddings persisted in a local Chroma database."""

from __future__ import annotations

from .config import Settings, get_settings


def get_embeddings(settings: Settings | None = None):
    """Return an Ollama embeddings client."""
    settings = settings or get_settings()
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_vectorstore(settings: Settings | None = None):
    """Return a persistent Chroma vector store backed by Ollama embeddings."""
    settings = settings or get_settings()
    from langchain_chroma import Chroma

    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.persist_dir),
    )
