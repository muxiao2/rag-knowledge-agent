"""Command-line interface for the RAG knowledge agent."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import get_settings

app = typer.Typer(
    add_completion=False,
    help="Local RAG knowledge-base Q&A agent (Ollama + ChromaDB).",
)
console = Console()


@app.command()
def ingest(
    path: Path | None = typer.Option(
        None, "--path", "-p", help="File or directory to ingest (defaults to data/docs)."
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Clear the existing collection before ingesting."
    ),
) -> None:
    """Load documents, chunk them, embed them, and store them in Chroma."""
    from .ingest import ingest as run_ingest
    from .ingest import ingest_path

    if path is not None:
        count = ingest_path(path)
    else:
        count = run_ingest(reset=reset)
    console.print(f"[green]Ingested {count} chunk(s).[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the knowledge base."),
    top_k: int | None = typer.Option(
        None, "--top-k", "-k", help="Number of chunks to retrieve."
    ),
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Show retrieved sources."
    ),
) -> None:
    """Ask a question and get an answer grounded in your knowledge base."""
    from .rag import RagAgent

    agent = RagAgent()
    result = agent.ask(question, top_k=top_k)
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))
    if show_sources and result.sources:
        console.print("\n[bold]Sources[/bold]")
        for i, src in enumerate(result.sources, start=1):
            score = f" (score={src.score:.3f})" if src.score is not None else ""
            console.print(f"  [{i}] {src.source}{score}")


@app.command()
def chat(
    top_k: int | None = typer.Option(
        None, "--top-k", "-k", help="Number of chunks to retrieve."
    ),
) -> None:
    """Start an interactive chat session against the knowledge base."""
    from .rag import RagAgent

    agent = RagAgent()
    console.print("[bold]RAG chat[/bold] — type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            question = console.input("[cyan]you[/cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = agent.ask(question, top_k=top_k)
        console.print(Panel(Markdown(result.answer), title="agent", border_style="green"))


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Host to bind (defaults to config)."),
    port: int | None = typer.Option(None, help="Port to bind (defaults to config)."),
    reload: bool = typer.Option(False, help="Enable autoreload (development)."),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag_agent.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@app.command()
def info() -> None:
    """Print the active configuration."""
    settings = get_settings()
    console.print(Panel.fit(
        f"LLM model:        {settings.llm_model}\n"
        f"Embedding model:  {settings.embedding_model}\n"
        f"Ollama base URL:  {settings.ollama_base_url}\n"
        f"Docs dir:         {settings.docs_dir}\n"
        f"Persist dir:      {settings.persist_dir}\n"
        f"Collection:       {settings.collection_name}\n"
        f"chunk_size/overlap: {settings.chunk_size}/{settings.chunk_overlap}\n"
        f"top_k:            {settings.top_k}",
        title="rag-agent config",
    ))


if __name__ == "__main__":
    app()
