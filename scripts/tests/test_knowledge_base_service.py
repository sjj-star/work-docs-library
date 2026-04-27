"""KnowledgeBaseService 单元测试."""

import pytest
from core.db import KnowledgeDB
from core.graph_store import GraphEntity, GraphRelation, NetworkXGraphStore
from core.knowledge_base_service import KnowledgeBaseService
from core.models import Chunk, Document
from core.vector_index import VectorIndex


def test_kbservice_uses_provided_components(tmp_path):
    """KBS 使用传入的组件而非新建."""
    db = KnowledgeDB(db_path=tmp_path / "test.db")
    vec = VectorIndex(
        dim=4, index_path=tmp_path / "faiss.index", id_map_path=tmp_path / "id_map.json"
    )
    graph = NetworkXGraphStore()
    svc = KnowledgeBaseService(db=db, vec=vec, graph_store=graph)
    assert svc.db is db
    assert svc.vec is vec
    assert svc.graph is graph


def test_kbservice_get_document_progress(tmp_path, monkeypatch):
    """get_document_progress 返回正确的统计."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    # Insert doc and chunks with different statuses
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))
    statuses = ["done", "done", "embedded", "pending", "failed"]
    for i, status in enumerate(statuses):
        ck = Chunk(doc_id="d1", chunk_id=f"c{i}", content="x", chunk_type="text", status="pending")
        db_id = svc.db.insert_chunk(ck)
        # insert_chunk hard-codes status to pending, update afterward
        svc.db.update_chunk_status(db_id, status)

    prog = svc.get_document_progress("d1")
    assert prog["total"] == 5
    assert prog["done"] == 2
    assert prog["embedded"] == 1
    assert prog["pending"] == 1
    assert prog["failed"] == 1
    assert prog["skipped"] == 0


def test_kbservice_query_chunks_requires_doc_id_for_chapter():
    """query_chunks chapter 查询需要 doc_id."""
    svc = KnowledgeBaseService()
    with pytest.raises(ValueError, match="doc_id"):
        svc.query_chunks(chapter="Intro")


def test_kbservice_query_chunks_requires_query_type():
    """query_chunks 至少需要一个查询条件."""
    svc = KnowledgeBaseService()
    with pytest.raises(ValueError, match="Provide"):
        svc.query_chunks()


def test_kbservice_get_chunk_content_by_chunk_id(tmp_path, monkeypatch):
    """get_chunk_content 通过 chunk_db_id 获取内容."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))
    ck = Chunk(doc_id="d1", chunk_id="c0", content="hello world", chunk_type="text")
    db_id = svc.db.insert_chunk(ck)

    result = svc.get_chunk_content(chunk_db_id=db_id)
    assert result["query_type"] == "chunk"
    assert result["content"] == "hello world"
    assert len(result["chunks"]) == 1


def test_kbservice_get_chunk_content_missing_chunk():
    """get_chunk_content 找不到 chunk 时抛出 ValueError."""
    svc = KnowledgeBaseService()
    with pytest.raises(ValueError, match="not found"):
        svc.get_chunk_content(chunk_db_id=99999)


def test_kbservice_get_chunk_content_missing_params():
    """get_chunk_content 缺少参数时抛出 ValueError."""
    svc = KnowledgeBaseService()
    with pytest.raises(ValueError, match="Provide"):
        svc.get_chunk_content()
    with pytest.raises(ValueError, match="doc_id"):
        svc.get_chunk_content(doc_id="d1")


def test_kbservice_graph_operations():
    """图谱查询操作正确委托给 GraphStore."""
    graph = NetworkXGraphStore()
    graph.add_entity(GraphEntity(entity_type="Module", name="M1"))
    graph.add_entity(GraphEntity(entity_type="Signal", name="S1"))
    graph.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="M1",
            to_name="S1",
            from_type="Module",
            to_type="Signal",
        )
    )
    svc = KnowledgeBaseService(graph_store=graph)

    # get_entity
    ent = svc.get_entity("Module", "M1")
    assert ent is not None
    assert ent.name == "M1"

    # find_entities
    ents = svc.find_entities(entity_type="Module")
    assert len(ents) == 1

    ents = svc.find_entities(name_pattern="S")
    assert len(ents) == 1
    assert ents[0].name == "S1"

    # get_neighbors
    neighbors = svc.get_neighbors("Module", "M1")
    assert len(neighbors) == 1
    assert neighbors[0][0].name == "S1"

    # get_subgraph
    sg = svc.get_subgraph("Module", "M1", depth=1)
    assert sg.node_count == 2

    # graph_stats
    stats = svc.graph_stats()
    assert stats == {"nodes": 2, "edges": 1}


def test_kbservice_find_path():
    """find_path 委托给 GraphStore."""
    graph = NetworkXGraphStore()
    for name in ("A", "B", "C"):
        graph.add_entity(GraphEntity(entity_type="Module", name=name))
    graph.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="A",
            to_name="B",
            from_type="Module",
            to_type="Module",
        )
    )
    graph.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="B",
            to_name="C",
            from_type="Module",
            to_type="Module",
        )
    )
    svc = KnowledgeBaseService(graph_store=graph)

    paths = svc.find_path("Module", "A", "Module", "C", max_depth=3)
    assert len(paths) == 1
    assert paths[0] == ["Module::A", "Module::B", "Module::C"]


def test_kbservice_reprocess_missing_doc():
    """reprocess_document 文档不存在时抛出 ValueError."""
    svc = KnowledgeBaseService()
    with pytest.raises(ValueError, match="not found"):
        svc.reprocess_document("nonexistent")


def test_kbservice_list_documents_empty(tmp_path, monkeypatch):
    """list_documents 空库返回空列表."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    assert svc.list_documents() == []


def test_kbservice_query_by_doc(tmp_path, monkeypatch):
    """query_by_doc 返回文档的所有 chunks."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))
    ck = Chunk(doc_id="d1", chunk_id="c0", content="hello", chunk_type="text")
    svc.db.insert_chunk(ck)

    chunks = svc.db.query_by_doc("d1")
    assert len(chunks) == 1
    assert chunks[0].content == "hello"
