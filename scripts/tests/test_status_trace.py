"""StatusCollector trace scope 测试."""

import pytest
from core.db import KnowledgeDB
from core.graph_store import NetworkXGraphStore
from core.knowledge_base_service import KnowledgeBaseService
from core.models import Chunk
from core.status_collector import StatusCollector
from core.usage_logger import UsageLogger
from core.vector_index import VectorIndex


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    db = KnowledgeDB(db_path=tmp_path / "test.db")
    vec = VectorIndex(dim=8, index_path=tmp_path / "faiss.index")
    graph = NetworkXGraphStore()
    service = KnowledgeBaseService(db=db, vec=vec, graph_store=graph)
    yield service


def test_status_trace_by_session(svc):
    chunk = Chunk(
        id=10000001,
        doc_id="doc1",
        chunk_id="c1",
        content="test",
        chapter_title="ch1",
    )
    svc.db.insert_block(doc_id="doc1", block_id="b1", content="test", seq_index=1)
    svc.vec.add(chunk.id, [1.0] * 8)

    logger = UsageLogger(svc.db)
    logger.log_query(
        tool_name="search",
        mode="hybrid",
        params={"text": "x"},
        result={"chunks": [{"score": 0.9, "chunk": chunk}], "entities": [], "relations": []},
        session_id="sess_a",
    )
    logger.log_query(
        tool_name="read",
        mode=None,
        params={"chunk_db_id": 10000001},
        result={"chunks": [chunk], "entities": [], "relations": []},
        session_id="sess_a",
    )

    collector = StatusCollector(svc)
    trace = collector.collect_trace_status(session_id="sess_a", top_n=10)

    assert trace["success"]
    assert trace["scope"] == "trace"
    assert trace["count"] == 2
    assert trace["traces"][0]["tool_name"] == "read"
    assert trace["traces"][1]["tool_name"] == "search"
