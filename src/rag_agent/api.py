"""FastAPI application exposing the RAG agent over HTTP."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .ingest import ingest
from .rag import Answer, RagAgent

app = FastAPI(
    title="RAG Knowledge Agent",
    version=__version__,
    description="A local RAG knowledge-base Q&A agent powered by Ollama and ChromaDB.",
)

_agent: RagAgent | None = None


def get_agent() -> RagAgent:
    global _agent
    if _agent is None:
        _agent = RagAgent()
    return _agent


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question.")
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceModel(BaseModel):
    content: str
    source: str
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]


class IngestResponse(BaseModel):
    chunks_added: int


def _to_response(answer: Answer) -> AskResponse:
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        sources=[
            SourceModel(content=s.content, source=s.source, score=s.score)
            for s in answer.sources
        ],
    )


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(reset: bool = False) -> IngestResponse:
    count = ingest(reset=reset)
    return IngestResponse(chunks_added=count)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    answer = get_agent().ask(request.question, top_k=request.top_k)
    return _to_response(answer)
