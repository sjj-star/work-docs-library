"""StatusCollector usage scope 测试."""

import pytest
from core.db import KnowledgeDB
from core.graph_store import GraphEntity, NetworkXGraphStore
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


def test_status_usage_basic(svc):
    chunk = Chunk(
        id=10000001,
        doc_id="doc1",
        chunk_id="c1",
        content="test",
        chapter_title="ch1",
    )
    svc.db.insert_block(doc_id="doc1", block_id="b1", content="test", seq_index=1)
    svc.vec.add(chunk.id, [1.0] * 8)

    entity = GraphEntity(
        entity_type="Module",
        name="EPWM_TZ",
        properties={},
        source_doc_ids={"doc1"},
        confidence=0.9,
        verified=False,
    )
    svc.add_entity(entity)

    logger = UsageLogger(svc.db)
    logger.log_query(
        tool_name="search",
        mode="hybrid",
        params={"text": "epwm"},
        result={
            "chunks": [{"score": 0.9, "chunk": chunk}],
            "entities": [entity],
            "relations": [],
        },
    )

    collector = StatusCollector(svc)
    usage = collector.collect_usage_status(top_n=5)

    assert usage["success"]
    assert usage["scope"] == "usage"
    assert usage["summary"]["total_logs"] == 1
    assert usage["summary"]["unused_entities"] == 0
    assert usage["summary"]["unverified_entities"] == 1


def test_status_usage_unused_entity(svc):
    entity = GraphEntity(
        entity_type="Module",
        name="UNUSED",
        properties={},
        source_doc_ids={"doc1"},
        verified=False,
    )
    svc.add_entity(entity)

    collector = StatusCollector(svc)
    usage = collector.collect_usage_status(top_n=5)

    assert usage["summary"]["unused_entities"] == 1
    assert any(e["name"] == "UNUSED" for e in usage["unused_entities_sample"])
