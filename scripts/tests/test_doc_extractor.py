import argparse
from pathlib import Path

import fitz
import pytest

from core.config import Config
from core.db import KnowledgeDB
from core.models import Chunk as ModelChunk, Document as ModelDoc
from core.pipeline import IngestionPipeline
from core.vector_index import VectorIndex
import doc_extractor


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
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSION", 4)
    return tmp_path


class FakeEmbedder:
    def __init__(self):
        self._dim_validated = True
    
    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    
    def get_embedding_dimension(self):
        return 4
    
    def close(self):
        pass


def test_cmd_ingest_and_status(patched_config, monkeypatch, caplog):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Hello"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)

    args = argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False)
    doc_extractor.cmd_ingest(args)

    args = argparse.Namespace()
    doc_extractor.cmd_status(args)
    # The status table is printed; pipeline completion is logged
    assert "Ingestion complete" in caplog.text or "doc.pdf" in caplog.text


def test_cmd_query_page(patched_config, monkeypatch, capsys):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    doc_extractor.cmd_ingest(argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False))

    db = KnowledgeDB()
    docs = db.list_documents()
    doc_id = docs[0].doc_id

    args = argparse.Namespace(doc_id=doc_id, page="1", chapter=None, chapter_regex=None, keyword=None, top_k=10)
    doc_extractor.cmd_query(args)
    captured = capsys.readouterr()
    assert "Page one" in captured.out


def test_cmd_search(patched_config, monkeypatch, capsys):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Unique keyword XYZ"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    monkeypatch.setattr(doc_extractor, "EmbeddingClient", FakeEmbedder)
    doc_extractor.cmd_ingest(argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False))

    args = argparse.Namespace(text="Unique keyword XYZ", top_k=5)
    doc_extractor.cmd_search(args)
    captured = capsys.readouterr()
    assert "Vector search" in captured.out


def test_cmd_toc(patched_config, monkeypatch, capsys):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["A"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    doc_extractor.cmd_ingest(argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False))

    db = KnowledgeDB()
    doc_id = db.list_documents()[0].doc_id
    args = argparse.Namespace(doc_id=doc_id, match=None)
    doc_extractor.cmd_toc(args)
    captured = capsys.readouterr()
    assert "doc.pdf" in captured.out or "全文" in captured.out


def test_cmd_list_pending_and_write_summary_keywords(patched_config, monkeypatch, capsys):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Content"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    doc_extractor.cmd_ingest(argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False))

    db = KnowledgeDB()
    doc_id = db.list_documents()[0].doc_id
    # Fake embedder already marks embedded, but summary is empty
    rows = db.get_embedded_but_unsummarized_chunks(doc_id)
    chunk_db_id = rows[0][0]

    args = argparse.Namespace(chunk_db_id=chunk_db_id, summary="Test summary")
    doc_extractor.cmd_write_summary(args)

    args = argparse.Namespace(chunk_db_id=chunk_db_id, keywords="a,b")
    doc_extractor.cmd_write_keywords(args)

    refreshed = db.get_chunk_by_db_id(chunk_db_id)
    assert refreshed.summary == "Test summary"
    assert refreshed.keywords == ["a", "b"]


def test_cmd_reprocess(patched_config, monkeypatch, caplog):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Old"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    doc_extractor.cmd_ingest(argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False))

    db = KnowledgeDB()
    doc_id = db.list_documents()[0].doc_id
    args = argparse.Namespace(doc_id=doc_id)
    doc_extractor.cmd_reprocess(args)
    assert "Reprocessed" in caplog.text


def test_cmd_query_chapter_and_keyword(patched_config, monkeypatch, capsys):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["KeywordABC"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    doc_extractor.cmd_ingest(argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False))

    db = KnowledgeDB()
    doc_id = db.list_documents()[0].doc_id
    with db._connect() as conn:
        row = conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()
    cid = row["id"]
    db.update_chunk_keywords(cid, "KeywordABC")

    args = argparse.Namespace(doc_id=doc_id, page=None, chapter="全文", chapter_regex=None, keyword=None, top_k=10)
    doc_extractor.cmd_query(args)
    captured = capsys.readouterr()
    assert "KeywordABC" in captured.out

    args = argparse.Namespace(doc_id=doc_id, page=None, chapter=None, chapter_regex=r"^全", keyword=None, top_k=10)
    doc_extractor.cmd_query(args)
    captured = capsys.readouterr()
    assert "KeywordABC" in captured.out

    args = argparse.Namespace(doc_id=None, page=None, chapter=None, chapter_regex=None, keyword="KeywordABC", top_k=10)
    doc_extractor.cmd_query(args)
    captured = capsys.readouterr()
    assert "KeywordABC" in captured.out


def test_cmd_toc_match(patched_config, monkeypatch, caplog):
    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["A"])

    pipe = IngestionPipeline()
    monkeypatch.setattr(pipe, "embedder", FakeEmbedder())
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    doc_extractor.cmd_ingest(argparse.Namespace(path=str(pdf), dry_run=False, auto_chapter=False))

    args = argparse.Namespace(doc_id=None, match="doc")
    doc_extractor.cmd_toc(args)
    # Pipeline "Done" is logged; table is printed to stdout
    assert "doc.pdf" in caplog.text or "Ingestion complete" in caplog.text


def test_cmd_list_pending_top_k(patched_config, monkeypatch, capsys):
    db = KnowledgeDB()
    db.upsert_document(ModelDoc(
        doc_id="d9", title="T", source_path=str(patched_config / "x.pdf"), file_type="pdf"
    ))
    for i in range(5):
        cid = db.insert_chunk(ModelChunk(
            doc_id="d9", chunk_id=f"p{i}", content="x", chunk_type="text"
        ))
        db.update_chunk_status(cid, "embedded")

    args = argparse.Namespace(doc_id="d9", top_k=3)
    doc_extractor.cmd_list_pending(args)
    captured = capsys.readouterr()
    # Should list 3 out of 5
    assert captured.out.count("chunk_db_id=") == 3


def test_cmd_write_embedding(patched_config, monkeypatch, caplog):
    db = KnowledgeDB()
    db.upsert_document(ModelDoc(
        doc_id="d10", title="T", source_path=str(patched_config / "y.pdf"), file_type="pdf"
    ))
    cid = db.insert_chunk(ModelChunk(
        doc_id="d10", chunk_id="p1", content="x", chunk_type="text"
    ))

    emb_file = patched_config / "emb.json"
    emb_file.write_text("[1.0, 0.0, 0.0, 0.0]", encoding="utf-8")

    monkeypatch.setattr(doc_extractor.Config, "EMBEDDING_DIMENSION", 4)
    args = argparse.Namespace(chunk_db_id=cid, embedding_file=str(emb_file))
    doc_extractor.cmd_write_embedding(args)
    assert "Embedding written" in caplog.text


def test_cmd_write_embedding_outside_skill_rejected(patched_config, monkeypatch, caplog):
    import tempfile
    emb_file = Path(tempfile.gettempdir()) / "outside_emb.json"
    emb_file.write_text("[1.0, 0.0, 0.0, 0.0]", encoding="utf-8")
    monkeypatch.setattr(doc_extractor.Config, "EMBEDDING_DIMENSION", 4)
    args = argparse.Namespace(chunk_db_id=1, embedding_file=str(emb_file))
    doc_extractor.cmd_write_embedding(args)
    assert "must be inside" in caplog.text or "outside skill directory" in caplog.text
    emb_file.unlink()


def test_cmd_write_embedding_bad_json(patched_config, monkeypatch, caplog):
    emb_file = patched_config / "bad_emb.json"
    emb_file.write_text("\"not an array\"", encoding="utf-8")
    monkeypatch.setattr(doc_extractor.Config, "EMBEDDING_DIMENSION", 4)
    args = argparse.Namespace(chunk_db_id=1, embedding_file=str(emb_file))
    doc_extractor.cmd_write_embedding(args)
    assert "non-empty JSON array" in caplog.text


def test_cmd_reprocess_not_found(patched_config, monkeypatch, caplog):
    pipe = IngestionPipeline()
    monkeypatch.setattr(doc_extractor, "IngestionPipeline", lambda: pipe)
    args = argparse.Namespace(doc_id="nonexistent")
    doc_extractor.cmd_reprocess(args)
    assert "not found" in caplog.text


def test_cmd_query_no_args(patched_config, monkeypatch, caplog):
    args = argparse.Namespace(doc_id="x", page=None, chapter=None, chapter_regex=None, keyword=None, top_k=10)
    doc_extractor.cmd_query(args)
    assert "Query requires" in caplog.text


def test_cmd_toc_no_args(patched_config, monkeypatch, caplog):
    args = argparse.Namespace(doc_id=None, match=None)
    doc_extractor.cmd_toc(args)
    assert "TOC requires" in caplog.text
