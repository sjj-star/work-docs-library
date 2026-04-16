import json
from pathlib import Path

import fitz
import pytest

from core.config import Config
from core.db import KnowledgeDB
from core.models import Document
from core.pipeline import IngestionPipeline
from core.vector_index import VectorIndex


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def patched_config(monkeypatch, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    monkeypatch.setattr(Config, "DB_PATH", kb / "workdocs.db")
    monkeypatch.setattr(Config, "FAISS_INDEX_PATH", kb / "faiss.index")
    monkeypatch.setattr(Config, "ID_MAP_PATH", kb / "id_map.json")
    monkeypatch.setattr(Config, "BATCH_SIZE", 2)
    return tmp_path


class FakeEmbedder:
    def __init__(self):
        self.dim = 4

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self):
        pass


def test_scan_file_and_dir(patched_config):
    pdf = patched_config / "a.pdf"
    _make_pdf(pdf, ["x"])
    sub = patched_config / "sub"
    sub.mkdir()
    pdf2 = sub / "b.pdf"
    _make_pdf(pdf2, ["y"])

    pipe = IngestionPipeline()
    files = pipe.scan(str(patched_config))
    assert str(pdf.resolve()) in files
    assert str(pdf2.resolve()) in files


def test_process_one_pdf_with_embedding(patched_config, monkeypatch):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page 1", "Page 2"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())

    doc_id = pipe._process_one(str(pdf), dry_run=False, auto_chapter=False)
    assert doc_id is not None

    db = KnowledgeDB()
    doc = db.get_document(doc_id)
    assert doc.status == "done"
    chunks = db.query_by_page(doc_id, 1, 2)
    assert len(chunks) == 2
    assert chunks[0].status == "embedded"

    vi = VectorIndex()
    assert vi._index.ntotal == 2


def test_skip_unchanged_file(patched_config, monkeypatch):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page 1"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())

    first_id = pipe._process_one(str(pdf), dry_run=False, auto_chapter=False)
    second_id = pipe._process_one(str(pdf), dry_run=False, auto_chapter=False)
    assert second_id == first_id


def test_force_reprocess(patched_config, monkeypatch):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page 1"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())

    first_id = pipe._process_one(str(pdf), dry_run=False, auto_chapter=False)
    second_id = pipe._process_one(str(pdf), dry_run=False, auto_chapter=False, force=True)
    assert second_id == first_id
    db = KnowledgeDB()
    chunks = db.query_by_page(first_id, 1, 1)
    assert len(chunks) == 1


def test_dry_run(patched_config):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page 1"])
    pipe = IngestionPipeline()
    doc_id = pipe._process_one(str(pdf), dry_run=True, auto_chapter=False)
    assert doc_id is not None
    db = KnowledgeDB()
    assert db.get_document(doc_id) is None


def test_ingest_auto_chapter_mapping(patched_config, monkeypatch):
    pdf = patched_config / "doc.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.set_toc([
        (1, "Ch1", 1),
        (1, "Ch2", 2),
    ])
    doc.save(str(pdf))
    doc.close()

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    doc_id = pipe._process_one(str(pdf), dry_run=False, auto_chapter=True)
    db = KnowledgeDB()
    chunks = db.query_by_page(doc_id, 1, 3)
    titles = {ck.chapter_title for ck in chunks}
    assert "Ch1" in titles or "Ch2" in titles
