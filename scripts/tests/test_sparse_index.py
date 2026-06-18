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
    assert "spi_reset" in BM25SparseIndex._tokenize("SPI_reset")
    assert "hello" in BM25SparseIndex._tokenize("Hello 世界")
    assert "世界" in BM25SparseIndex._tokenize("Hello 世界")


def test_bm25_empty_query(sample_db):
    idx = BM25SparseIndex(sample_db)
    assert idx.search("", top_k=5) == []


def test_bm25_multi_document_search(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "multi.db")
    for doc_id in ("doc-a", "doc-b", "doc-c"):
        db.upsert_document(
            Document(
                doc_id=doc_id,
                title=f"Test {doc_id}",
                source_path=f"/tmp/{doc_id}.pdf",
                file_type="pdf",
                total_pages=1,
                chapters=[],
                extracted_at="2026-01-01",
                file_hash=doc_id,
                status="done",
            )
        )
    db.insert_block("doc-a", "ba1", "SPI protocol details", 0, {})
    db.insert_block("doc-b", "bb1", "GPIO pin assignments", 0, {})
    db.insert_block("doc-c", "bc1", "Timer interrupt handling", 0, {})
    idx = BM25SparseIndex(db)

    hits_spi = idx.search("SPI", top_k=5)
    hits_gpio = idx.search("GPIO", top_k=5)

    def _doc_id_of(block_id: int) -> str | None:
        block = db.get_block_by_db_id(block_id)
        return block["doc_id"] if block else None

    assert any(_doc_id_of(bid) == "doc-a" for bid, _ in hits_spi)
    assert any(_doc_id_of(bid) == "doc-b" for bid, _ in hits_gpio)


def test_bm25_no_match_score_zero(sample_db):
    idx = BM25SparseIndex(sample_db)
    hits = idx.search("xyznonexistentterm", top_k=5)
    assert hits == []


def test_bm25_from_blocks():
    blocks = [
        {"id": 1, "content": "SPI reset sequence"},
        {"id": 2, "content": "GPIO configuration unrelated"},
        {"id": 3, "content": "Timer interrupt handling"},
    ]
    idx = BM25SparseIndex.from_blocks(blocks)
    assert idx.index_info()["num_blocks"] == 3
    assert idx.index_info()["built"] is True
    hits = idx.search("SPI reset", top_k=5)
    assert len(hits) > 0
    assert hits[0][0] == 1
