# RAG Knowledge Agent — Sample Knowledge Base

This is a sample document so you can try the agent immediately. Replace the
files in `data/docs/` with your own `.md`, `.txt`, or `.pdf` documents and run
`rag-agent ingest --reset`.

## What is RAG?

Retrieval-Augmented Generation (RAG) combines a retrieval system with a large
language model. Instead of relying only on what the model memorized during
training, RAG first retrieves relevant passages from a knowledge base and then
asks the model to answer using those passages as grounding context. This makes
answers more accurate, up to date, and traceable to sources.

## How this project works

1. **Ingest** — documents in `data/docs/` are loaded and split into overlapping
   chunks of about 800 characters.
2. **Embed** — each chunk is converted into a vector using the Ollama embedding
   model `nomic-embed-text`.
3. **Store** — vectors are persisted locally in a ChromaDB collection.
4. **Retrieve** — at query time the most similar chunks are fetched.
5. **Generate** — the chunks are passed to the Ollama chat model `llama3.1`,
   which produces an answer grounded in the retrieved context.

## Default models

The agent uses Ollama by default. The chat model is `llama3.1` and the
embedding model is `nomic-embed-text`. Both run fully offline on your machine,
so no data leaves your computer.

## Frequently asked questions

**Q: Does this require an internet connection?**
No. Once the Ollama models are pulled, everything runs locally and offline.

**Q: What file types are supported?**
Plain text (`.txt`), Markdown (`.md`/`.markdown`), and PDF (`.pdf`).

**Q: How do I change the model?**
Set `RAG_LLM_MODEL` and `RAG_EMBEDDING_MODEL` in your `.env` file.
