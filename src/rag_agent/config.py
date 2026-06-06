"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration.

    Values are read from environment variables (optionally via a `.env` file).
    All settings use the ``RAG_`` prefix, e.g. ``RAG_OLLAMA_BASE_URL``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"
    embedding_model: str = "nomic-embed-text"

    # Generation
    temperature: float = 0.1
    num_ctx: int = 4096

    # Vector store
    persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "knowledge_base"

    # Ingestion / retrieval
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached `Settings` instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
