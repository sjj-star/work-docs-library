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


# ---------------------------------------------------------------------------
# EntityChunkBridge 跨粒度关联测试
# ---------------------------------------------------------------------------


def _make_chunk_with_entities(doc_id, chunk_id, content, entities):
    """辅助函数：创建带 extracted_entities metadata 的 Chunk."""
    meta = {"extracted_entities": [{"type": et, "name": en} for et, en in entities]}
    return Chunk(
        doc_id=doc_id, chunk_id=chunk_id, content=content, chunk_type="text", metadata=meta
    )


def test_bridge_rebuild_from_db(tmp_path, monkeypatch):
    """Bridge 从 SQLite 全量重建，正确解析 metadata 中的实体."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))

    ck1 = _make_chunk_with_entities("d1", "c0", "reg A desc", [("Register", "A"), ("Module", "M1")])
    ck2 = _make_chunk_with_entities("d1", "c1", "reg B desc", [("Register", "B"), ("Module", "M1")])
    id1 = svc.db.insert_chunk(ck1)
    id2 = svc.db.insert_chunk(ck2)

    # 重建后验证
    svc._bridge.rebuild(svc.db)

    # 正向查询
    assert svc._bridge.get_entities(id1) == {("Register", "A"), ("Module", "M1")}
    assert svc._bridge.get_entities(id2) == {("Register", "B"), ("Module", "M1")}

    # 反向查询
    assert svc._bridge.get_chunks(("Register", "A")) == {id1}
    assert svc._bridge.get_chunks(("Register", "B")) == {id2}
    assert svc._bridge.get_chunks(("Module", "M1")) == {id1, id2}
    assert svc._bridge.get_chunks(("Signal", "X")) == set()


def test_bridge_attach_detach_idempotent():
    """Attach 幂等：同一 chunk 重复 attach 不会累积；detach 后双向清理."""
    from core.knowledge_base_service import _EntityChunkBridge, _EntityRef

    bridge = _EntityChunkBridge()
    ref_a = _EntityRef("Register", "A")
    ref_b = _EntityRef("Module", "M1")

    bridge.attach(1, {ref_a, ref_b})
    assert bridge.get_entities(1) == {ref_a, ref_b}
    assert bridge.get_chunks(ref_a) == {1}
    assert bridge.get_chunks(ref_b) == {1}

    # 重复 attach（替换为不同实体）
    ref_c = _EntityRef("Signal", "S1")
    bridge.attach(1, {ref_c})
    assert bridge.get_entities(1) == {ref_c}
    assert bridge.get_chunks(ref_a) == set()   # 旧引用已清理
    assert bridge.get_chunks(ref_b) == set()
    assert bridge.get_chunks(ref_c) == {1}

    # detach
    bridge.detach(1)
    assert bridge.get_entities(1) == set()
    assert bridge.get_chunks(ref_c) == set()
    assert len(bridge._forward) == 0
    assert len(bridge._reverse) == 0


def test_bridge_forward_reverse_consistency():
    """_forward 和 _reverse 始终保持双向一致."""
    from core.knowledge_base_service import _EntityChunkBridge, _EntityRef

    bridge = _EntityChunkBridge()
    refs = {_EntityRef("Register", "A"), _EntityRef("Register", "B")}

    bridge.attach(10, refs)
    bridge.attach(20, {_EntityRef("Register", "A")})

    # 验证 forward
    assert bridge._forward[10] == refs
    assert bridge._forward[20] == {_EntityRef("Register", "A")}

    # 验证 reverse
    assert bridge._reverse[_EntityRef("Register", "A")] == {10, 20}
    assert bridge._reverse[_EntityRef("Register", "B")] == {10}

    # detach 一个 chunk 后 reverse 自动清理空集合
    bridge.detach(20)
    ref_a = _EntityRef("Register", "A")
    assert ref_a not in bridge._reverse or bridge._reverse[ref_a] == {10}
    assert bridge.get_chunks(_EntityRef("Register", "B")) == {10}


def test_sync_bridge_for_doc(tmp_path, monkeypatch):
    """_sync_bridge_for_doc 正确同步指定文档的索引."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))
    svc.db.upsert_document(Document(doc_id="d2", title="T2", source_path="/y.pdf", file_type="pdf"))

    id1 = svc.db.insert_chunk(_make_chunk_with_entities("d1", "c0", "x", [("Register", "A")]))
    id2 = svc.db.insert_chunk(_make_chunk_with_entities("d2", "c0", "y", [("Register", "B")]))

    svc._sync_bridge_for_doc("d1")

    # d1 的 chunk 已索引
    assert svc._entity_to_chunks("Register", "A") == {id1}
    # d2 未被同步（旧数据若存在应被清理）
    assert svc._entity_to_chunks("Register", "B") == set()

    # 同步 d2
    svc._sync_bridge_for_doc("d2")
    assert svc._entity_to_chunks("Register", "B") == {id2}
    # d1 不受影响
    assert svc._entity_to_chunks("Register", "A") == {id1}


def test_find_chunks_by_entity(tmp_path, monkeypatch):
    """find_chunks_by_entity 通过桥接索引 O(1) 反向查找 chunk."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))

    id1 = svc.db.insert_chunk(_make_chunk_with_entities("d1", "c0", "reg A", [("Register", "A")]))
    id2 = svc.db.insert_chunk(
        _make_chunk_with_entities("d1", "c1", "reg A and B", [("Register", "A"), ("Register", "B")])
    )
    svc._sync_bridge_for_doc("d1")

    # 不限制 doc_id
    chunks = svc.find_chunks_by_entity("Register", "A")
    assert len(chunks) == 2
    assert {c.id for c in chunks} == {id1, id2}

    # 限制 doc_id（虽只有 d1，测试过滤逻辑）
    chunks = svc.find_chunks_by_entity("Register", "A", doc_id="d1")
    assert len(chunks) == 2

    # 不存在的实体
    assert svc.find_chunks_by_entity("Signal", "X") == []

    # 返回深拷贝验证
    chunks[0].content = "modified"
    fresh = svc.find_chunks_by_entity("Register", "A")
    assert fresh[0].content != "modified"


def test_chunk_to_entities_and_entity_to_chunks(tmp_path, monkeypatch):
    """原子操作 _chunk_to_entities 和 _entity_to_chunks 正确工作."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))

    id1 = svc.db.insert_chunk(_make_chunk_with_entities("d1", "c0", "x", [("Register", "A")]))
    svc._sync_bridge_for_doc("d1")

    from core.knowledge_base_service import _EntityRef

    assert svc._chunk_to_entities(id1) == {_EntityRef("Register", "A")}
    assert svc._entity_to_chunks("Register", "A") == {id1}
    assert svc._entity_to_chunks("Register", "B") == set()


def test_get_chunk_deep_copy(tmp_path, monkeypatch):
    """_get_chunk 返回深拷贝，修改不影响原数据."""
    monkeypatch.setattr("core.config.Config.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("core.config.Config.FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr("core.config.Config.ID_MAP_PATH", tmp_path / "id_map.json")

    svc = KnowledgeBaseService()
    svc.db.upsert_document(Document(doc_id="d1", title="T", source_path="/x.pdf", file_type="pdf"))
    id1 = svc.db.insert_chunk(
        Chunk(doc_id="d1", chunk_id="c0", content="original", chunk_type="text")
    )

    ck = svc._get_chunk(id1)
    assert ck is not None
    assert ck.content == "original"

    ck.content = "modified"
    ck2 = svc._get_chunk(id1)
    assert ck2.content == "original"
