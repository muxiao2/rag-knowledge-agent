import pytest

from rag_agent.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_single_chunk():
    chunks = chunk_text("hello world", chunk_size=800, chunk_overlap=120)
    assert chunks == ["hello world"]


def test_respects_chunk_size():
    text = "\n\n".join(f"paragraph number {i} " * 5 for i in range(20))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    # Allow some slack for overlap prefix but chunks should be bounded.
    assert all(len(c) <= 200 + 40 for c in chunks)


def test_long_paragraph_is_windowed():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_preserves_context():
    text = "\n\n".join(f"sentence-{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (100, 100), (100, 200), (10, -1)],
)
def test_invalid_arguments_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=size, chunk_overlap=overlap)
