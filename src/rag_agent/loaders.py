"""Document loading utilities.

Reads supported files from a directory into plain ``Document`` records so the
rest of the pipeline does not depend on a particular file format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Document:
    """A loaded source document."""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_file(path: Path) -> Document:
    """Load a single supported file into a :class:`Document`."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path})")

    if suffix == ".pdf":
        content = _read_pdf(path)
    else:
        content = path.read_text(encoding="utf-8")

    return Document(content=content, metadata={"source": str(path), "filename": path.name})


def load_directory(directory: Path) -> list[Document]:
    """Recursively load all supported files under ``directory``."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {directory}")

    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            doc = load_file(path)
            if doc.content.strip():
                docs.append(doc)
    return docs
