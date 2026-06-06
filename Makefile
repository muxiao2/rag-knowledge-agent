.PHONY: install dev lint test fmt ingest serve chat clean

install:
	uv sync

dev:
	uv sync --extra dev

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

test:
	uv run pytest

ingest:
	uv run rag-agent ingest --reset

serve:
	uv run rag-agent serve

chat:
	uv run rag-agent chat

clean:
	rm -rf data/chroma .pytest_cache .ruff_cache
