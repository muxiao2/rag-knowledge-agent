from fastapi.testclient import TestClient

from rag_agent.api import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm_model" in body
    assert "embedding_model" in body
