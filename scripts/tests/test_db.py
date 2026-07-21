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


def test_update_document_status(db, sample_doc):
    """Test update document status."""
    db.upsert_document(sample_doc)
    db.update_document_status("doc1", "done")
    doc = db.get_document("doc1")
    assert doc.status == "done"


def test_insert_block_and_query(db, sample_doc):
    """Test insert block and query."""
    db.upsert_document(sample_doc)
    db_id = db.insert_block(
        doc_id="doc1",
        block_id="b1",
        content="hello world",
        seq_index=0,
        metadata={"section_title": "Ch1"},
    )
    assert isinstance(db_id, int)

    by_doc = db.query_blocks_by_doc("doc1")
    assert len(by_doc) == 1
    assert by_doc[0]["content"] == "hello world"

    fetched = db.get_block_by_db_id(db_id)
    assert fetched is not None
    assert fetched["metadata"]["section_title"] == "Ch1"


def test_block_embedding_batch(db, sample_doc):
    """Test block embedding batch update."""
    db.upsert_document(sample_doc)
    db_id = db.insert_block(doc_id="doc1", block_id="b1", content="c", seq_index=0)
    db.update_blocks_embedded_batch([(db_id, [0.1, 0.2, 0.3])])
    block = db.get_block_by_db_id(db_id)
    assert block["metadata"]["embedding"] == [0.1, 0.2, 0.3]
    assert block["status"] == "embedded"


def test_delete_blocks_by_doc(db, sample_doc):
    """Test delete blocks by doc."""
    db.upsert_document(sample_doc)
    db.insert_block(doc_id="doc1", block_id="b1", content="c", seq_index=0)
    db.delete_blocks_by_doc("doc1")
    assert db.query_blocks_by_doc("doc1") == []


def test_conflict_logs(db):
    """冲突日志插入和查询."""
    logs = [
        {
            "entity_type": "Register",
            "name": "CTRL",
            "property_key": "width",
            "old_value": "32",
            "new_value": "64",
            "timestamp": "2026-01-01T00:00:00",
            "doc_id": "doc1",
        }
    ]
    db.insert_conflict_logs(logs)
    results = db.query_conflict_logs()
    assert len(results) == 1
    assert results[0]["property_key"] == "width"
    assert results[0]["old_value"] == "32"

    # 按实体过滤
    results = db.query_conflict_logs(entity_type="Register", name="CTRL")
    assert len(results) == 1
    results = db.query_conflict_logs(entity_type="Signal")
    assert len(results) == 0


def test_feedback(db):
    """反馈插入和查询."""
    fid = db.insert_feedback(
        rating=1,
        entity_type="Module",
        entity_name="TOP",
        comment="Correct",
    )
    assert fid > 0

    results = db.query_feedback(entity_type="Module", entity_name="TOP")
    assert len(results) == 1
    assert results[0]["rating"] == 1
    assert results[0]["comment"] == "Correct"

    score = db.get_entity_feedback_score("Module", "TOP")
    assert score == 1

    # 添加负反馈
    db.insert_feedback(rating=-1, entity_type="Module", entity_name="TOP")
    score = db.get_entity_feedback_score("Module", "TOP")
    assert score == 0

    # 关系反馈评分
    db.insert_feedback(
        rating=1,
        relation_type="HAS_REGISTER",
        relation_from_type="Module",
        relation_from_name="TOP",
        relation_to_type="Register",
        relation_to_name="CTRL",
    )
    rel_score = db.get_relation_feedback_score("HAS_REGISTER", "Module", "TOP", "Register", "CTRL")
    assert rel_score == 1

    db.insert_feedback(
        rating=-1,
        relation_type="HAS_REGISTER",
        relation_from_type="Module",
        relation_from_name="TOP",
        relation_to_type="Register",
        relation_to_name="CTRL",
    )
    rel_score = db.get_relation_feedback_score("HAS_REGISTER", "Module", "TOP", "Register", "CTRL")
    assert rel_score == 0


# -- 从 test_models.py 合并的模型默认值测试 --


def test_chapter_to_dict():
    """Chapter.to_dict() 应正确序列化."""
    ch = Chapter(title="Intro", start_page=1, end_page=5, level=1)
    d = ch.to_dict()
    assert d == {"title": "Intro", "start_page": 1, "end_page": 5, "level": 1}


def test_chunk_defaults():
    """Chunk 默认值：metadata 应为空字典."""
    ck = Chunk(doc_id="d1", chunk_id="c1", content="hello", chunk_type="text")
    assert ck.metadata == {}


def test_document_defaults():
    """Document 默认值：status 为 pending，chapters 为空列表."""
    import tempfile
    from pathlib import Path

    doc = Document(
        doc_id="d1",
        title="t",
        source_path=str(Path(tempfile.gettempdir()) / "a.pdf"),
        file_type="pdf",
    )
    assert doc.status == "pending"
    assert doc.chapters == []
