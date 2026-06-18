import pytest
from core.db import KnowledgeDB
from core.models import Document
from core.sparse_index import BM25SparseIndex


@pytest.fixture
def sample_db(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "test.db")
    db.upsert_document(
        Document(
            doc_id="doc-a",
            title="Test",
            source_path="/tmp/test.pdf",
            file_type="pdf",
            total_pages=1,
            chapters=[],
            extracted_at="2026-01-01",
            file_hash="abc",
            status="done",
        )
    )
    db.insert_block("doc-a", "b1", "The SPI module reset sequence begins with CS low.", 0, {})
    db.insert_block("doc-a", "b2", "GPIO configuration is unrelated to SPI timing.", 1, {})
    db.insert_block("doc-a", "b3", "Another paragraph about timers.", 2, {})
    return db


def test_bm25_sparse_index_search(sample_db):
    idx = BM25SparseIndex(sample_db)
    hits = idx.search("SPI reset sequence", top_k=5)
    assert len(hits) > 0
    # The first block should rank highest for SPI reset sequence
    top_id, top_score = hits[0]
    assert top_score > 0
    block = sample_db.get_block_by_db_id(top_id)
    assert block is not None
    assert "SPI" in block["content"]


def test_bm25_sparse_index_empty_db(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "empty.db")
    idx = BM25SparseIndex(db)
    assert idx.search("anything") == []
    assert idx.index_info()["num_blocks"] == 0
    assert idx.index_info()["built"] is False


def test_bm25_tokenize():
    assert BM25SparseIndex._tokenize("SPI_reset") == ["spi_reset"]
    assert BM25SparseIndex._tokenize("Hello 世界") == ["hello", "世", "界"]
