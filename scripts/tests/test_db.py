"""test_db 模块."""

import pytest
from core.db import KnowledgeDB
from core.models import Chapter, Chunk, Document


@pytest.fixture
def db(tmp_path):
    """Db 函数."""
    db_path = tmp_path / "test.db"
    return KnowledgeDB(db_path=db_path)


@pytest.fixture
def sample_doc(tmp_path):
    """sample_doc 函数."""
    return Document(
        doc_id="doc1",
        title="Test Doc",
        source_path=str(tmp_path / "test.pdf"),
        file_type="pdf",
        total_pages=10,
        chapters=[Chapter(title="Ch1", start_page=1, end_page=5, level=1)],
        file_hash="abc123",
        status="pending",
    )


def test_upsert_and_get_document(db, sample_doc):
    """Test upsert and get document."""
    db.upsert_document(sample_doc)
    retrieved = db.get_document("doc1")
    assert retrieved is not None
    assert retrieved.title == "Test Doc"
    assert retrieved.chapters[0].title == "Ch1"


def test_get_document_missing(db):
    """Test get document missing."""
    assert db.get_document("nonexistent") is None


def test_list_documents(db, sample_doc):
    """Test list documents."""
    db.upsert_document(sample_doc)
    docs = db.list_documents()
    assert len(docs) == 1
    assert docs[0].doc_id == "doc1"


def test_get_document_by_path(db, sample_doc):
    """Test get document by path."""
    db.upsert_document(sample_doc)
    found = db.get_document_by_path(sample_doc.source_path)
    assert found is not None
    assert found.doc_id == "doc1"


def test_search_documents_by_title(db, sample_doc):
    """Test search documents by title."""
    db.upsert_document(sample_doc)
    results = db.search_documents_by_title("Test")
    assert len(results) == 1
    results = db.search_documents_by_title("Nothing")
    assert len(results) == 0


def test_update_document_status(db, sample_doc):
    """Test update document status."""
    db.upsert_document(sample_doc)
    db.update_document_status("doc1", "done")
    doc = db.get_document("doc1")
    assert doc.status == "done"


def test_insert_chunk_and_query(db, sample_doc):
    """Test insert chunk and query."""
    db.upsert_document(sample_doc)
    chunk = Chunk(
        doc_id="doc1",
        chunk_id="p1",
        content="hello world",
        chunk_type="text",
        chapter_title="Ch1",
        keywords=["kw1", "kw2"],
        summary="sum",
    )
    db_id = db.insert_chunk(chunk)
    assert isinstance(db_id, int)

    by_doc = db.query_by_doc("doc1")
    assert len(by_doc) == 1
    assert by_doc[0].content == "hello world"

    by_chapter = db.query_by_chapter("doc1", "Ch1")
    assert len(by_chapter) == 1

    by_regex = db.query_by_chapter_regex("doc1", r"^Ch")
    assert len(by_regex) == 1

    by_kw = db.query_by_keyword("kw1")
    assert len(by_kw) == 1

    fetched = db.get_chunk_by_db_id(db_id)
    assert fetched is not None
    assert fetched.keywords == ["kw1", "kw2"]


def test_chunk_summary(db, sample_doc):
    """Test chunk summary."""
    db.upsert_document(sample_doc)
    chunk = Chunk(doc_id="doc1", chunk_id="p1", content="c", chunk_type="text")
    db_id = db.insert_chunk(chunk)
    db.update_chunk_summary(db_id, "new summary")
    ck = db.get_chunk_by_db_id(db_id)
    assert ck.summary == "new summary"


def test_chunk_embedding_batch(db, sample_doc):
    """Test chunk embedding batch update."""
    db.upsert_document(sample_doc)
    chunk = Chunk(doc_id="doc1", chunk_id="p1", content="c", chunk_type="text")
    db_id = db.insert_chunk(chunk)
    db.update_chunks_embedded_batch([(db_id, [0.1, 0.2, 0.3])])
    ck = db.get_chunk_by_db_id(db_id)
    assert ck.metadata["embedding"] == [0.1, 0.2, 0.3]
    assert ck.status == "embedded"


def test_delete_chunks_by_doc(db, sample_doc):
    """Test delete chunks by doc."""
    db.upsert_document(sample_doc)
    db.insert_chunk(Chunk(doc_id="doc1", chunk_id="p1", content="c", chunk_type="text"))
    db.delete_chunks_by_doc("doc1")
    assert db.query_by_doc("doc1") == []


def test_get_pending_chunks(db, sample_doc):
    """Test get pending chunks."""
    db.upsert_document(sample_doc)
    c1 = Chunk(doc_id="doc1", chunk_id="p1", content="pending", chunk_type="text")
    db.insert_chunk(c1)

    pending = db.get_pending_chunks("doc1")
    assert len(pending) == 1
    assert pending[0][3] == "pending"
