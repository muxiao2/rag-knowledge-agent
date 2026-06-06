from rag_agent.config import Settings


def test_defaults():
    s = Settings()
    assert s.llm_model == "llama3.1"
    assert s.embedding_model == "nomic-embed-text"
    assert s.top_k == 4
    assert s.ollama_base_url.startswith("http")


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen2.5")
    monkeypatch.setenv("RAG_TOP_K", "7")
    s = Settings()
    assert s.llm_model == "qwen2.5"
    assert s.top_k == 7
