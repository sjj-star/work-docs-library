"""KnowledgeBaseService 联合查询测试."""

import pytest
from core.config import Config
from core.graph_store import GraphEntity, GraphRelation
from core.knowledge_base_service import KnowledgeBaseService, _EntityRef


def _make_service(tmp_path, monkeypatch):
    """创建测试用的 KnowledgeBaseService 实例."""
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(Config, "FAISS_INDEX_PATH", tmp_path / "faiss.index")
    return KnowledgeBaseService()


def test_search_with_graph_basic(tmp_path, monkeypatch):
    """search_with_graph 返回 chunks + 关联实体."""
    svc = _make_service(tmp_path, monkeypatch)

    # 插入 block
    db_id = svc.db.insert_block(
        doc_id="d1", block_id="b1", content="Test content about PIEIER register", seq_index=0
    )

    # 手动建立桥接索引
    svc._bridge.attach(db_id, {_EntityRef("Register", "PIEIER")})

    # 添加实体到图谱
    svc.graph.add_entity(
        GraphEntity(
            entity_type="Register",
            name="PIEIER",
            properties={"address": "0x0000"},
            source_doc_ids={"d1"},
        )
    )

    # mock 语义搜索返回该 chunk
    monkeypatch.setattr(svc, "_semantic_hits", lambda text, top_k: [(db_id, 0.95)])

    result = svc.search_with_graph("test", top_k=5, graph_depth=0)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["chunk"].content == "Test content about PIEIER register"
    assert len(result["related_entities"]) == 1
    assert result["related_entities"][0]["type"] == "Register"
    assert result["related_entities"][0]["name"] == "PIEIER"


def test_get_content_with_entities(tmp_path, monkeypatch):
    """get_content_with_entities 返回 chunk + 实体 + 关系."""
    svc = _make_service(tmp_path, monkeypatch)

    # 插入 block
    db_id = svc.db.insert_block(
        doc_id="d1", block_id="b1", content="Module A contains Signal B", seq_index=0
    )

    # 手动建立桥接索引
    svc._bridge.attach(db_id, {_EntityRef("Module", "A"), _EntityRef("Signal", "B")})

    # 添加实体和关系到图谱
    svc.graph.add_entity(
        GraphEntity(entity_type="Module", name="A", properties={}, source_doc_ids={"d1"})
    )
    svc.graph.add_entity(
        GraphEntity(entity_type="Signal", name="B", properties={}, source_doc_ids={"d1"})
    )
    svc.graph.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_type="Module",
            from_name="A",
            to_type="Signal",
            to_name="B",
            properties={},
            source_doc_ids={"d1"},
        )
    )

    result = svc.get_content_with_entities(db_id)
    assert result["chunk"].content == "Module A contains Signal B"
    assert len(result["entities"]) == 2
    names = {e.name for e in result["entities"]}
    assert names == {"A", "B"}
    assert len(result["relations"]) == 1
    assert result["relations"][0].rel_type == "CONTAINS"


def test_get_content_with_entities_not_found(tmp_path, monkeypatch):
    """get_content_with_entities 对不存在的 chunk 应抛出 ValueError."""
    svc = _make_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Block 99999 not found"):
        svc.get_content_with_entities(99999)
