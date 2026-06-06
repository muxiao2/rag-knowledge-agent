# RAG Knowledge Agent

A local, offline-first **RAG (Retrieval-Augmented Generation) knowledge-base Q&A agent**.
Drop your documents into a folder, ingest them into a local vector database, and
ask questions — answers are grounded in your own knowledge base and cite their
sources. Everything runs on your machine via [Ollama](https://ollama.com); no data
leaves your computer.

## Features

- 🧠 **Local LLM + embeddings** via Ollama (`llama3.1` + `nomic-embed-text` by default)
- 📚 **Document ingestion** for `.md`, `.txt`, and `.pdf` files
- 🔎 **Local vector store** with [ChromaDB](https://www.trychroma.com/) (persisted to disk)
- 💬 **CLI** (`ask`, `chat`, `ingest`, `serve`, `info`) built with Typer + Rich
- 🌐 **HTTP API** with FastAPI (`/ask`, `/ingest`, `/health`)
- ✅ Tests + linting + GitHub Actions CI

## Architecture

```
            ┌──────────────┐      ┌───────────────┐      ┌──────────────┐
documents → │   loaders    │ ───► │   chunking    │ ───► │  embeddings  │
(.md/.txt/  │ (read files) │      │ (split text)  │      │   (Ollama)   │
 .pdf)      └──────────────┘      └───────────────┘      └──────┬───────┘
                                                                │
                                                          ┌─────▼──────┐
                                                          │  ChromaDB  │  (persisted)
                                                          └─────┬──────┘
   question ─────────────────────────────────────────────────►│ retrieve top-k
                                                          ┌─────▼──────┐
                                                          │  ChatOllama│ → grounded answer
                                                          │ (llama3.1) │   + sources
                                                          └────────────┘
```

## Prerequisites

1. **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`.
2. **Ollama** installed and running: https://ollama.com/download
3. Pull the default models:

   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

## Setup

```bash
# clone, then:
uv sync --extra dev          # install dependencies (incl. dev tools)
cp .env.example .env         # optional: customize models / settings
```

> Using pip instead of uv? `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

## Usage

### 1. Add your documents

Put your `.md`, `.txt`, or `.pdf` files into `data/docs/` (a sample file is included).

### 2. Ingest them into the vector store

```bash
uv run rag-agent ingest --reset
```

### 3. Ask questions

```bash
uv run rag-agent ask "What is RAG and how does this project work?"
```

### 4. Interactive chat

```bash
uv run rag-agent chat
```

### 5. Run the HTTP API

```bash
uv run rag-agent serve          # http://localhost:8000  (docs at /docs)
```

```bash
# ingest, then ask via HTTP
curl -X POST localhost:8000/ingest
curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What models does this project use by default?"}'
```

### Inspect configuration

```bash
uv run rag-agent info
```

## Configuration

All settings are environment variables with the `RAG_` prefix (see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_LLM_MODEL` | `llama3.1` | Chat model |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_CHUNK_SIZE` | `800` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `RAG_TOP_K` | `4` | Chunks retrieved per query |
| `RAG_COLLECTION_NAME` | `knowledge_base` | Chroma collection name |
| `RAG_API_HOST` / `RAG_API_PORT` | `0.0.0.0` / `8000` | API bind address |

## Project layout

```
src/rag_agent/
  config.py       # settings (pydantic-settings)
  loaders.py      # read .md/.txt/.pdf into Documents
  chunking.py     # dependency-free text splitter
  vectorstore.py  # Ollama embeddings + Chroma
  ingest.py       # load -> chunk -> embed -> persist
  rag.py          # retrieve + generate (the agent)
  prompts.py      # prompt templates
  api.py          # FastAPI app
  cli.py          # Typer CLI
tests/            # unit tests (no Ollama required)
data/docs/        # your knowledge base lives here
```

## Development

```bash
make dev     # install with dev extras
make lint    # ruff check
make test    # pytest
make fmt     # ruff format
```

## License

MIT
