import pytest

from rag_agent.loaders import load_directory, load_file


def test_load_markdown_file(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\nhello", encoding="utf-8")
    doc = load_file(p)
    assert "hello" in doc.content
    assert doc.metadata["filename"] == "note.md"


def test_unsupported_file_raises(tmp_path):
    p = tmp_path / "data.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(p)


def test_load_directory_skips_unsupported_and_empty(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")

    docs = load_directory(tmp_path)
    contents = sorted(d.content for d in docs)
    assert contents == ["alpha", "beta"]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "does-not-exist")
