"""UsageLogger 与 usage_logs / block_activation 表测试."""

import pytest
from core.db import KnowledgeDB
from core.models import Chunk
from core.usage_logger import UsageLogger


@pytest.fixture
def usage_db(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "test.db")
    yield db


@pytest.fixture
def usage_logger(usage_db):
    return UsageLogger(usage_db)


def _make_chunk(chunk_id: int, doc_id: str = "doc1") -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        chunk_id=f"b{chunk_id}",
        content=f"content {chunk_id}",
        chapter_title="ch1",
    )


def test_log_query_records_usage(usage_logger, usage_db):
    chunk = _make_chunk(10000001)
    result = {
        "chunks": [{"score": 0.9, "chunk": chunk}],
        "entities": [
            {"entity_type": "Module", "name": "EPWM_TZ"},
        ],
        "relations": [
            {
                "rel_type": "TRIGGERS",
                "from_type": "Module",
                "from_name": "EPWM_TZ",
                "to_type": "Task",
                "to_name": "CLA1TASK1",
            }
        ],
    }

    log_id = usage_logger.log_query(
        tool_name="search",
        mode="hybrid",
        params={"text": "test query", "top_k": 5},
        result=result,
        session_id="sess_1",
    )

    assert log_id > 0
    rows = usage_db.get_usage_trace(session_id="sess_1")
    assert len(rows) == 1
    row = rows[0]
    assert row["tool_name"] == "search"
    assert row["mode"] == "hybrid"
    assert row["session_id"] == "sess_1"
    assert "EPWM_TZ" in row["entity_hits"]
    assert "CLA1TASK1" in row["relation_hits"]
    assert "10000001" in row["vector_hits"]


def test_block_activation_incremented(usage_logger, usage_db):
    chunk = _make_chunk(10000002)
    result = {"chunks": [{"score": 0.8, "chunk": chunk}], "entities": [], "relations": []}

    usage_logger.log_query(
        tool_name="search",
        mode="semantic",
        params={"text": "x"},
        result=result,
    )
    usage_logger.log_query(
        tool_name="search",
        mode="semantic",
        params={"text": "x"},
        result=result,
    )

    report = usage_db.get_usage_report(top_n=5)
    hot = report["hot_blocks"]
    assert any(b["block_db_id"] == 10000002 and b["hit_count"] == 2 for b in hot)


def test_cleanup_usage_logs(usage_db):
    db = usage_db
    db.log_usage(
        tool_name="search",
        mode="hybrid",
        params={"text": "old"},
        vector_hits=[],
        entity_hits=[],
        relation_hits=[],
        elapsed_ms=1,
        session_id="old",
    )

    deleted = db.cleanup_usage_logs(max_days=0, max_rows=10000)
    assert deleted["deleted_by_time"] == 1


def test_add_usage_flag(usage_logger, usage_db):
    chunk = _make_chunk(10000003)
    result = {"chunks": [{"score": 0.7, "chunk": chunk}], "entities": [], "relations": []}
    log_id = usage_logger.log_query(
        tool_name="search",
        mode="hybrid",
        params={"text": "flag test"},
        result=result,
    )

    ok = usage_db.add_usage_flag(
        log_id=log_id,
        kind="entity",
        identifier={"type": "Module", "name": "X"},
        reason="wrong",
    )
    assert ok

    rows = usage_db.get_usage_trace()
    assert rows[0]["flagged_items"] is not None
    assert "wrong" in rows[0]["flagged_items"]
