"""test_status_tool 模块."""

import plugin_router
import pytest
from core.config import Config
from core.graph_store import GraphEntity, GraphRelation
from core.knowledge_base_service import KnowledgeBaseService
from core.models import Chapter, Document


@pytest.fixture
def status_svc(monkeypatch):
    """创建已填充示例数据的服务，并让 tool_status 使用该服务."""
    service = KnowledgeBaseService()
    # 清理历史数据（使用同一个测试 DB）
    service.graph.clear()
    for doc in service.list_documents():
        service.db.delete_blocks_by_doc(doc.doc_id)
        service.db.delete_heading_maps_by_doc(doc.doc_id)
        # 删除文档本身
        with service.db._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc.doc_id,))

    # 文档 A：完整处理
    doc_a = Document(
        doc_id="doc-a",
        title="Doc A",
        source_path="/tmp/doc-a.pdf",
        file_type="pdf",
        total_pages=10,
        chapters=[Chapter(title="Ch1", start_page=1, end_page=5, level=2)],
        extracted_at="2026-06-06T10:00:00",
        file_hash="hash-a",
        status="done",
    )
    service.db.upsert_document(doc_a)
    for i in range(3):
        service.db.insert_block("doc-a", f"b{i}", f"content {i}", i)
    # 模拟 embedding
    with service.db._connect() as conn:
        conn.execute("UPDATE content_blocks SET status = 'embedded' WHERE doc_id = 'doc-a'")

    # 添加实体/关系到图中
    service.graph.add_entity(
        GraphEntity(
            entity_type="Register",
            name="REG_A",
            properties={"desc": "test"},
            source_doc_ids={"doc-a"},
            confidence=0.9,
        )
    )
    service.graph.add_relation(
        GraphRelation(
            rel_type="belongs_to",
            from_type="Register",
            from_name="REG_A",
            to_type="Module",
            to_name="MOD_A",
            source_doc_ids={"doc-a"},
            confidence=0.8,
        )
    )

    # 文档 B：失败
    doc_b = Document(
        doc_id="doc-b",
        title="Doc B",
        source_path="/tmp/doc-b.pdf",
        file_type="pdf",
        total_pages=5,
        chapters=[],
        extracted_at="2026-06-06T11:00:00",
        file_hash="hash-b",
        status="failed",
    )
    service.db.upsert_document(doc_b)

    # 添加 heading_map
    service.db.insert_heading_map("doc-a", "Ch1", 2, None, [1, 2, 3])

    # 让 plugin_router 的工具函数使用本 fixture 中的服务
    monkeypatch.setattr(plugin_router, "_get_service", lambda: service)

    yield service

    service.graph.clear()


def test_status_overview_default(status_svc):
    """默认 overview 返回文档列表."""
    result = plugin_router.tool_status({})
    assert result["success"] is True
    assert "documents" in result
    doc_ids = {d["doc_id"] for d in result["documents"]}
    assert doc_ids == {"doc-a", "doc-b"}


def test_status_overview_doc_id(status_svc):
    """Overview 带 doc_id 返回进度."""
    result = plugin_router.tool_status({"doc_id": "doc-a"})
    assert result["success"] is True
    assert result["doc_id"] == "doc-a"
    assert "total" in result
    assert result["total"] == 3


def test_status_documents_scope(status_svc):
    """Documents scope 返回按文档聚合的统计."""
    result = plugin_router.tool_status({"scope": "documents"})
    assert result["success"] is True
    assert result["scope"] == "documents"
    assert result["summary"]["total_documents"] == 2
    doc = next(d for d in result["documents"] if d["doc_id"] == "doc-a")
    assert doc["blocks"]["total"] == 3
    assert doc["graph_entities"] == 1
    assert doc["graph_relations"] == 1
    assert len(result["failed_documents"]) == 1


def test_status_vectors_scope(status_svc):
    """Vectors scope 返回索引信息."""
    result = plugin_router.tool_status({"scope": "vectors"})
    assert result["success"] is True
    assert result["scope"] == "vectors"
    assert "index" in result
    assert "database" in result
    assert "consistency" in result
    assert result["index"]["dimension"] == Config.EMBEDDING_DIMENSION


def test_status_vectors_scope_counts_done_as_embedded(status_svc):
    """状态为 done 的 block 也应计入 embedded_blocks."""
    # 将 doc-a 的部分 block 改为 done，模拟 stage6 最终状态
    with status_svc.db._connect() as conn:
        conn.execute("UPDATE content_blocks SET status = 'done' WHERE doc_id = 'doc-a' LIMIT 2")
    result = plugin_router.tool_status({"scope": "vectors"})
    assert result["database"]["embedded_blocks"] == 3


def test_status_graph_scope(status_svc):
    """Graph scope 返回图谱统计和分布."""
    result = plugin_router.tool_status({"scope": "graph"})
    assert result["success"] is True
    assert result["scope"] == "graph"
    assert result["summary"]["node_count"] >= 2
    assert result["summary"]["edge_count"] >= 1
    assert "Register" in result["entity_type_distribution"]
    assert "belongs_to" in result["relation_type_distribution"]


def test_status_blocks_scope(status_svc):
    """Blocks scope 返回 block 统计."""
    result = plugin_router.tool_status({"scope": "blocks"})
    assert result["success"] is True
    assert result["summary"]["total"] == 3


def test_status_headings_scope(status_svc):
    """Headings scope 返回 heading 统计."""
    result = plugin_router.tool_status({"scope": "headings"})
    assert result["success"] is True
    assert result["summary"]["total"] == 1


def test_status_config_scope_masked(status_svc):
    """Config scope 返回脱敏配置."""
    result = plugin_router.tool_status({"scope": "config"})
    assert result["success"] is True
    assert "config_groups" in result
    # 敏感 key 应被脱敏
    for group in result["config_groups"].values():
        for key, val in group.items():
            if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
                assert val in ("***", "") or val is None


def test_status_quality_scope(status_svc):
    """Quality scope 返回质量检查."""
    result = plugin_router.tool_status({"scope": "quality"})
    assert result["success"] is True
    assert result["scope"] == "quality"
    assert "score" in result
    assert "metrics" in result
    assert "issues" in result
    assert any(i["category"] == "document" for i in result["issues"])


def test_status_ingest_pipeline_scope(status_svc):
    """Ingest_pipeline scope 返回阶段推断."""
    result = plugin_router.tool_status({"scope": "ingest_pipeline"})
    assert result["success"] is True
    doc_a = next(p for p in result["pipelines"] if p["doc_id"] == "doc-a")
    assert doc_a["stages"]["parsed"] is True


def test_status_all_scope(status_svc):
    """All scope 返回所有子 scope."""
    result = plugin_router.tool_status({"scope": "all"})
    assert result["success"] is True
    expected_keys = [
        "documents",
        "vectors",
        "graph",
        "blocks",
        "headings",
        "quality",
        "ingest_pipeline",
    ]
    for key in expected_keys:
        assert key in result


def test_status_unknown_scope(status_svc):
    """未知 scope 返回错误."""
    result = plugin_router.tool_status({"scope": "not_exist"})
    assert result["success"] is False
    assert "Unknown scope" in result["error"]


def test_status_top_n(status_svc):
    """Top_n 参数生效."""
    result = plugin_router.tool_status({"scope": "documents", "top_n": 1})
    assert len(result["documents"]) == 1
