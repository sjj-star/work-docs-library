import builtins
from io import StringIO

import pytest

from core.chapter_editor import ChapterEditor
from core.db import KnowledgeDB
from core.models import Chapter, Document


def _mock_input(returns):
    gen = iter(returns)
    def fake_input(prompt=""):
        return next(gen)
    return fake_input


def test_interactive_edit_save(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    db = KnowledgeDB(db_path=db_path)
    doc = Document(
        doc_id="d1", title="Doc", source_path=str(tmp_path / "a.pdf"), file_type="pdf",
        total_pages=5, chapters=[Chapter(title="Original", start_page=1, end_page=5, level=1)]
    )
    db.upsert_document(doc)

    editor = ChapterEditor(db=db)
    monkeypatch.setattr(builtins, "input", _mock_input([
        "add", "New Ch", "2", "4", "1",
        "save"
    ]))
    editor.interactive_edit("d1")

    chapters = db.get_chapters("d1")
    assert len(chapters) == 2
    assert chapters[1].title == "New Ch"


def test_interactive_edit_invalid_input(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "test.db"
    db = KnowledgeDB(db_path=db_path)
    doc = Document(
        doc_id="d1", title="Doc", source_path=str(tmp_path / "a.pdf"), file_type="pdf",
        total_pages=5, chapters=[Chapter(title="Original", start_page=1, end_page=5, level=1)]
    )
    db.upsert_document(doc)

    editor = ChapterEditor(db=db)
    monkeypatch.setattr(builtins, "input", _mock_input([
        "add", "Bad Ch", "abc", "2", "4", "1",
        "save"
    ]))
    editor.interactive_edit("d1")
    assert "must be integers" in caplog.text


def test_interactive_edit_del_and_reorder(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    db = KnowledgeDB(db_path=db_path)
    doc = Document(
        doc_id="d1", title="Doc", source_path=str(tmp_path / "a.pdf"), file_type="pdf",
        total_pages=5, chapters=[
            Chapter(title="A", start_page=1, end_page=2, level=1),
            Chapter(title="B", start_page=3, end_page=4, level=1),
        ]
    )
    db.upsert_document(doc)

    editor = ChapterEditor(db=db)
    monkeypatch.setattr(builtins, "input", _mock_input([
        "reorder", "2,1",
        "del", "2",
        "save"
    ]))
    editor.interactive_edit("d1")

    chapters = db.get_chapters("d1")
    assert len(chapters) == 1
    assert chapters[0].title == "B"


def test_edit_nonexistent_doc(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "test.db"
    db = KnowledgeDB(db_path=db_path)
    editor = ChapterEditor(db=db)
    monkeypatch.setattr(builtins, "input", _mock_input(["quit"]))
    editor.interactive_edit("missing")
    assert "not found" in caplog.text


def test_interactive_edit_quit_without_save(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    db = KnowledgeDB(db_path=db_path)
    doc = Document(
        doc_id="d1", title="Doc", source_path=str(tmp_path / "a.pdf"), file_type="pdf",
        total_pages=5, chapters=[Chapter(title="Original", start_page=1, end_page=5, level=1)]
    )
    db.upsert_document(doc)

    editor = ChapterEditor(db=db)
    monkeypatch.setattr(builtins, "input", _mock_input([
        "add", "New Ch", "2", "4", "1",
        "quit"
    ]))
    editor.interactive_edit("d1")

    chapters = db.get_chapters("d1")
    assert len(chapters) == 1
    assert chapters[0].title == "Original"


def test_reorder_error_handling(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "test.db"
    db = KnowledgeDB(db_path=db_path)
    doc = Document(
        doc_id="d1", title="Doc", source_path=str(tmp_path / "a.pdf"), file_type="pdf",
        total_pages=5, chapters=[
            Chapter(title="A", start_page=1, end_page=2, level=1),
            Chapter(title="B", start_page=3, end_page=4, level=1),
        ]
    )
    db.upsert_document(doc)

    editor = ChapterEditor(db=db)
    monkeypatch.setattr(builtins, "input", _mock_input([
        "reorder", "bad,order",
        "save"
    ]))
    editor.interactive_edit("d1")
    assert "Error" in caplog.text
