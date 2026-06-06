"""The RAG question-answering pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings
from .prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT, format_context
from .vectorstore import get_vectorstore


@dataclass
class Source:
    """A retrieved context chunk used to ground the answer."""

    content: str
    source: str
    score: float | None = None


@dataclass
class Answer:
    """The agent's response together with the sources it relied on."""

    question: str
    answer: str
    sources: list[Source]


def get_llm(settings: Settings | None = None):
    """Return an Ollama chat model."""
    settings = settings or get_settings()
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
    )


class RagAgent:
    """Retrieve relevant chunks and answer questions grounded in them."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._store = None
        self._llm = None

    @property
    def store(self):
        if self._store is None:
            self._store = get_vectorstore(self.settings)
        return self._store

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(self.settings)
        return self._llm

    def retrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        k = top_k or self.settings.top_k
        results = self.store.similarity_search_with_relevance_scores(question, k=k)
        sources: list[Source] = []
        for doc, score in results:
            sources.append(
                Source(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "unknown")),
                    score=float(score) if score is not None else None,
                )
            )
        return sources

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        sources = self.retrieve(question, top_k=top_k)
        context = format_context([s.content for s in sources])
        prompt = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT, context=context, question=question
        )
        response = self.llm.invoke(prompt)
        text = getattr(response, "content", str(response))
        return Answer(question=question, answer=text, sources=sources)
