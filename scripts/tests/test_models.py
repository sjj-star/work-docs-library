"""test_models 模块."""

import tempfile
from pathlib import Path

from core.models import Chapter, Chunk, Document


def test_chapter_to_dict():
    """Test chapter to dict."""
    ch = Chapter(title="Intro", start_page=1, end_page=5, level=1)
    d = ch.to_dict()
    assert d == {"title": "Intro", "start_page": 1, "end_page": 5, "level": 1}


def test_chunk_defaults():
    """Test chunk defaults."""
    ck = Chunk(doc_id="d1", chunk_id="c1", content="hello", chunk_type="text")
    assert ck.keywords == []
    assert ck.metadata == {}


def test_document_defaults():
    """Test document defaults."""
    doc = Document(
        doc_id="d1",
        title="t",
        source_path=str(Path(tempfile.gettempdir()) / "a.pdf"),
        file_type="pdf",
    )
    assert doc.status == "pending"
    assert doc.chapters == []
