import tempfile
from pathlib import Path

from core.models import Chapter, Chunk, Document


def test_chapter_to_dict():
    ch = Chapter(title="Intro", start_page=1, end_page=5, level=1)
    d = ch.to_dict()
    assert d == {"title": "Intro", "start_page": 1, "end_page": 5, "level": 1}


def test_chunk_defaults():
    ck = Chunk(doc_id="d1", chunk_id="c1", content="hello", chunk_type="text")
    assert ck.page_start == 0
    assert ck.keywords == []
    assert ck.metadata == {}


def test_document_defaults():
    doc = Document(doc_id="d1", title="t", source_path=str(Path(tempfile.gettempdir()) / "a.pdf"), file_type="pdf")
    assert doc.status == "pending"
    assert doc.chunks == []
    assert doc.chapters == []
