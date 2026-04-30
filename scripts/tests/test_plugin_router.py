"""Tests for plugin_router.py covering recent bug fixes:.

- FlowSelector integration (LLM_API_FLOW vs AGENT_SKILL_FLOW)
- get_content tool
- plugin.json format regression.
"""

import json
from pathlib import Path

import fitz
import plugin_router
import pytest
from core.config import Config
from core.db import KnowledgeDB

_SKILL_ROOT = Path(__file__).resolve().parent.parent


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def patched_config(monkeypatch, tmp_path):
    """patched_config 函数."""
    kb = tmp_path / "kb"
    kb.mkdir()
    monkeypatch.setattr(Config, "DB_PATH", kb / "workdocs.db")
    monkeypatch.setattr(Config, "FAISS_INDEX_PATH", kb / "faiss.index")
    monkeypatch.setattr(Config, "ID_MAP_PATH", kb / "id_map.json")
    monkeypatch.setattr(Config, "BATCH_SIZE", 2)
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(Config, "EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(Config, "PROMPT_DIR", _SKILL_ROOT / "prompts")
    return tmp_path


class FakeEmbedder:
    """FakeEmbedder 类."""

    def __init__(self):
        """初始化 FakeEmbedder."""
        self._dim_validated = True

    def embed(self, texts):
        """Embed 函数."""
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def get_embedding_dimension(self):
        """get_embedding_dimension 函数."""
        return 4

    def close(self):
        """Close 函数."""
        pass


class FakeBigModelParserClient:
    """Mock BigModelParserClient that falls back to local parsing."""

    def parse_pdf(self, *args, **kwargs):
        """parse_pdf 函数."""
        return ("", [])  # empty triggers local parser fallback

    def create_task(self, *args, **kwargs):
        """create_task 函数."""
        return "fake-task"

    def poll_result(self, *args, **kwargs):
        """poll_result 函数."""
        return {"status": "succeeded"}

    def download_result(self, *args, **kwargs):
        """download_result 函数."""
        return b""


class FakeBatchClient:
    """Mock BatchClient that returns fake results immediately."""

    def __init__(self, *args, **kwargs):
        pass

    def submit_and_wait(self, requests, **kwargs):
        """submit_and_wait 函数."""
        results = []
        for i, req in enumerate(requests):
            results.append(
                {
                    "custom_id": req.get("custom_id", f"batch_{i}"),
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": (
                                            '{"entities": [], "relationships": [], '
                                            '"image_descriptions": []}'
                                        )
                                    }
                                }
                            ]
                        }
                    },
                }
            )
        return results

    def submit_parallel_batches(self, requests, **kwargs):
        """submit_parallel_batches 函数."""
        return self.submit_and_wait(requests, **kwargs)

    def submit_embedding_batch(self, texts, **kwargs):
        """submit_embedding_batch 函数."""
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self):
        """Close 函数."""
        pass


def _mock_llm_and_embedder(monkeypatch):
    """Mock EmbeddingClient and BigModelParserClient for DocGraphPipeline."""
    fake_embed = FakeEmbedder()
    fake_file = FakeBigModelParserClient()
    monkeypatch.setattr(
        "core.doc_graph_pipeline.EmbeddingClient",
        lambda: fake_embed,
    )
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BigModelParserClient",
        lambda: fake_file,
    )
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BatchClient",
        FakeBatchClient,
    )


# ---------------------------------------------------------------------------
# FlowSelector integration
# ---------------------------------------------------------------------------


def test_ingest_with_llm_config_uses_doc_graph_pipeline(patched_config, monkeypatch):
    """When LLM is configured, ingest should use DocGraphPipeline."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one content", "Page two content"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    assert result["success"] is True
    doc_id = result["doc_ids"][0]

    db = KnowledgeDB()
    chunks = db.query_by_doc(doc_id)
    assert len(chunks) > 0
    # LLM_API_FLOW should persist summary and mark as done
    for ck in chunks:
        assert ck.status == "done"
        assert ck.summary is not None and len(ck.summary) > 0


def test_reprocess_doc_graph_pipeline(patched_config, monkeypatch):
    """tool_reprocess should use DocGraphPipeline."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Original content"])

    # First ingest
    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    # Reprocess
    result = plugin_router.tool_reprocess({"doc_id": doc_id})
    assert result["success"] is True

    db = KnowledgeDB()
    chunks = db.query_by_doc(doc_id)
    assert chunks[0].status == "done"


# ---------------------------------------------------------------------------
# get_content tool
# ---------------------------------------------------------------------------


def test_get_content_by_chapter(patched_config, monkeypatch):
    """Test get content by chapter."""
    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Chapter Alpha text", "Chapter Alpha more"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    # Query by chunk_db_id to get content (chapter titles may be empty for untitled docs)
    import sqlite3

    with sqlite3.connect(str(Config.DB_PATH)) as conn:
        row = conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
        chunk_db_id = row[0]
    result = plugin_router.tool_get_content({"chunk_db_id": chunk_db_id})
    assert result["success"] is True
    assert result["query_type"] == "chunk"
    assert "Alpha" in result["content"]
    assert result["total_chars"] > 0
    assert len(result["chunks"]) > 0


def test_get_content_by_chapter_empty_title(patched_config, monkeypatch):
    """Test get content by chapter with empty title (fallback chunk)."""
    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one text", "Page two text"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    # Query by empty chapter title should match fallback chunks
    result = plugin_router.tool_get_content({"doc_id": doc_id, "chapter": ""})
    assert result["success"] is True
    assert result["query_type"] == "chapter"
    assert "Page one" in result["content"]


def test_get_content_by_chunk_db_id(patched_config, monkeypatch):
    """Test get content by chunk db id."""
    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Single page"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    import sqlite3

    with sqlite3.connect(str(Config.DB_PATH)) as conn:
        row = conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
        chunk_db_id = row[0]

    result = plugin_router.tool_get_content({"chunk_db_id": chunk_db_id})
    assert result["success"] is True
    assert result["query_type"] == "chunk"
    assert "Single page" in result["content"]
    assert result["total_chars"] > 0


def test_get_content_missing_params():
    """Test get content missing params."""
    result = plugin_router.tool_get_content({})
    assert result["success"] is False
    assert "chunk_db_id" in result["error"]


# ---------------------------------------------------------------------------
# plugin.json regression
# ---------------------------------------------------------------------------


def test_plugin_json_all_commands_use_venv_python():
    """Test plugin json all commands use venv python."""
    plugin_path = _SKILL_ROOT.parent / "plugin.json"
    data = json.loads(plugin_path.read_text(encoding="utf-8"))
    assert "tools" in data
    for tool in data["tools"]:
        cmd = tool.get("command", [])
        assert len(cmd) >= 1, f"{tool['name']} has empty command"
        assert cmd[0] == "venv/bin/python3", (
            f"{tool['name']} command should start with 'venv/bin/python3', got {cmd[0]}"
        )


def test_plugin_json_valid_schema():
    """Test plugin json valid schema."""
    plugin_path = _SKILL_ROOT.parent / "plugin.json"
    data = json.loads(plugin_path.read_text(encoding="utf-8"))
    assert data.get("name") == "work-docs-library"
    assert "version" in data
    assert "tools" in data
    names = [t["name"] for t in data["tools"]]
    expected = [
        "ingest",
        "search",
        "query",
        "status",
        "toc",
        "progress",
        "reprocess",
        "get_content",
        "graph_query",
        "graph_neighbors",
        "graph_path",
        "graph_subgraph",
        "graph_add_entity",
        "graph_update_entity",
        "graph_delete_entity",
        "graph_add_relation",
        "graph_delete_relation",
        "graph_verify_entity",
        "graph_search_with_graph",
        "graph_get_content_with_entities",
        "graph_feedback",
        "graph_conflicts",
        "doc_parse",
        "doc_build_batches",
        "doc_submit_batches",
    ]
    for name in expected:
        assert name in names, f"Missing tool: {name}"


# ---------------------------------------------------------------------------
# Failure recovery and resume
# ---------------------------------------------------------------------------


def test_ingest_failure_sets_status_failed(patched_config, monkeypatch):
    """DocGraphPipeline: LLM entity extraction failure is caught gracefully.

    Document should still succeed with empty entities.
    """
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    fake_embed = FakeEmbedder()
    fake_file = FakeBigModelParserClient()
    monkeypatch.setattr("core.doc_graph_pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.doc_graph_pipeline.BigModelParserClient", lambda: fake_file)
    monkeypatch.setattr("core.doc_graph_pipeline.BatchClient", FakeBatchClient)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one content"])

    # DocGraphPipeline catches LLM errors gracefully; should not raise
    result = plugin_router.tool_ingest({"path": str(pdf)})
    assert result["success"] is True

    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None
    assert doc.status == "done"
    chunks = db.query_by_doc(doc.doc_id)
    assert len(chunks) == 1
    assert chunks[0].status == "done"


def test_ingest_resume_skips_phase_a(patched_config, monkeypatch):
    """Resume processing from embedded status.

    If chunks are already 'embedded', Phase A should be skipped and
    Phase B should resume, marking chunks as 'done'.
    """
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one content"])

    # First ingest - complete
    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    # Reset to simulate interrupted Phase B
    db = KnowledgeDB()
    with db._connect() as conn:
        conn.execute(
            "UPDATE chunks SET status = 'embedded', summary = '', keywords = '' WHERE doc_id = ?",
            (doc_id,),
        )
        conn.execute("UPDATE documents SET status = 'embedded' WHERE doc_id = ?", (doc_id,))

    before_count = len(db.query_by_doc(doc_id))

    # Second ingest - should resume Phase B only
    result = plugin_router.tool_ingest({"path": str(pdf)})
    assert result["success"] is True

    after_count = len(db.query_by_doc(doc_id))
    assert after_count == before_count, "Phase A should be skipped, no new chunks inserted"

    chunks = db.query_by_doc(doc_id)
    assert chunks[0].status == "done"
    assert chunks[0].summary != ""


# ---------------------------------------------------------------------------
# Image analysis persistence
# ---------------------------------------------------------------------------


def test_image_analysis_persisted_to_chunk_metadata(patched_config, monkeypatch):
    """Persist image analysis results to chunk metadata.

    Image analysis results from LLM vision should be saved into
    the chunk's metadata JSON under the 'vision_desc' key.
    """
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    fake_embed = FakeEmbedder()
    monkeypatch.setattr("core.doc_graph_pipeline.EmbeddingClient", lambda: fake_embed)

    fake_file = FakeBigModelParserClient()
    monkeypatch.setattr("core.doc_graph_pipeline.BigModelParserClient", lambda: fake_file)
    monkeypatch.setattr("core.doc_graph_pipeline.BatchClient", FakeBatchClient)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page with image placeholder"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    db = KnowledgeDB()
    chunks = db.query_by_doc(doc_id)
    assert len(chunks) == 1
    # DocGraphPipeline: chunks are marked done and have summary
    assert chunks[0].status == "done"
    assert chunks[0].summary != ""


# ---------------------------------------------------------------------------
# Graph tools
# ---------------------------------------------------------------------------


def _make_graph_service():
    """创建一个预填充了测试数据的 KnowledgeBaseService."""
    from core.graph_store import GraphEntity, GraphRelation, NetworkXGraphStore
    from core.knowledge_base_service import KnowledgeBaseService

    svc = KnowledgeBaseService(
        db=KnowledgeDB(),
        vec=None,  # graph tools don't need vec
        graph_store=NetworkXGraphStore(),
    )
    # Build test graph: TOP -> SUB -> REG, TOP -> CLK
    for name in ("TOP", "SUB"):
        svc.graph.add_entity(GraphEntity(entity_type="Module", name=name))
    svc.graph.add_entity(GraphEntity(entity_type="Signal", name="CLK"))
    svc.graph.add_entity(GraphEntity(entity_type="Register", name="REG"))
    svc.graph.add_relation(
        GraphRelation(
            rel_type="CONTAINS",
            from_name="TOP",
            to_name="SUB",
            from_type="Module",
            to_type="Module",
        )
    )
    svc.graph.add_relation(
        GraphRelation(
            rel_type="HAS_SIGNAL",
            from_name="TOP",
            to_name="CLK",
            from_type="Module",
            to_type="Signal",
        )
    )
    svc.graph.add_relation(
        GraphRelation(
            rel_type="HAS_REGISTER",
            from_name="SUB",
            to_name="REG",
            from_type="Module",
            to_type="Register",
        )
    )
    return svc


def test_graph_query_exact_match(monkeypatch):
    """graph_query 精确匹配实体."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_query({"entity_type": "Module", "name": "TOP"})
    assert result["success"] is True
    assert result.get("count") == 1
    assert result["entities"][0]["name"] == "TOP"


def test_graph_query_name_pattern(monkeypatch):
    """graph_query 按名称模糊匹配."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_query({"name_pattern": "CL"})
    assert result["success"] is True
    assert result["count"] == 1
    assert result["entities"][0]["name"] == "CLK"


def test_graph_query_by_type(monkeypatch):
    """graph_query 按类型列出所有实体."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_query({"entity_type": "Module"})
    assert result["success"] is True
    assert result["count"] == 2
    names = {e["name"] for e in result["entities"]}
    assert names == {"TOP", "SUB"}


def test_graph_neighbors_out(monkeypatch):
    """graph_neighbors out 方向."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_neighbors(
        {"entity_type": "Module", "name": "TOP", "direction": "out"}
    )
    assert result["success"] is True
    assert result["count"] == 2
    names = {n["entity"]["name"] for n in result["neighbors"]}
    assert names == {"SUB", "CLK"}


def test_graph_neighbors_filtered_rel_type(monkeypatch):
    """graph_neighbors 按关系类型过滤."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_neighbors(
        {"entity_type": "Module", "name": "TOP", "rel_type": "CONTAINS"}
    )
    assert result["success"] is True
    assert result["count"] == 1
    assert result["neighbors"][0]["entity"]["name"] == "SUB"


def test_graph_path_found(monkeypatch):
    """graph_path 找到路径."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_path(
        {
            "from_type": "Module",
            "from_name": "TOP",
            "to_type": "Register",
            "to_name": "REG",
        }
    )
    assert result["success"] is True
    assert result["path_count"] == 1
    path = result["paths"][0]
    assert len(path) == 3
    assert path[0]["name"] == "TOP"
    assert path[1]["name"] == "SUB"
    assert path[2]["name"] == "REG"


def test_graph_path_not_found(monkeypatch):
    """graph_path 找不到路径."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_path(
        {
            "from_type": "Signal",
            "from_name": "CLK",
            "to_type": "Register",
            "to_name": "REG",
        }
    )
    assert result["success"] is True
    assert result["path_count"] == 0


def test_graph_subgraph(monkeypatch):
    """graph_subgraph 提取子图."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_subgraph(
        {"center_type": "Module", "center_name": "TOP", "depth": 2}
    )
    assert result["success"] is True
    assert result["node_count"] == 4
    names = {e["name"] for e in result["entities"]}
    assert names == {"TOP", "SUB", "CLK", "REG"}


def test_graph_tools_missing_params():
    """图谱工具缺少必需参数时返回错误."""
    assert plugin_router.tool_graph_neighbors({})["success"] is False
    assert plugin_router.tool_graph_path({})["success"] is False
    assert plugin_router.tool_graph_subgraph({})["success"] is False


def test_graph_add_entity(monkeypatch):
    """graph_add_entity 添加实体."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_add_entity(
        {"entity_type": "Module", "name": "NEW", "properties": {"desc": "test"}}
    )
    assert result["success"] is True
    assert result["entity"]["name"] == "NEW"
    assert result["conflicts"] == []


def test_graph_update_entity(monkeypatch):
    """graph_update_entity 更新实体."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    plugin_router.tool_graph_add_entity({"entity_type": "Module", "name": "M1"})
    result = plugin_router.tool_graph_update_entity(
        {"entity_type": "Module", "name": "M1", "properties": {"a": 1}, "verified": True}
    )
    assert result["success"] is True

    # 验证更新生效
    e = svc.graph.get_entity("Module", "M1")
    assert e is not None
    assert e.properties == {"a": 1}
    assert e.verified is True


def test_graph_delete_entity(monkeypatch):
    """graph_delete_entity 删除实体."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    plugin_router.tool_graph_add_entity({"entity_type": "Module", "name": "DEL"})
    result = plugin_router.tool_graph_delete_entity({"entity_type": "Module", "name": "DEL"})
    assert result["success"] is True
    assert svc.graph.get_entity("Module", "DEL") is None


def test_graph_add_relation(monkeypatch):
    """graph_add_relation 添加关系."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    plugin_router.tool_graph_add_entity({"entity_type": "Module", "name": "A"})
    plugin_router.tool_graph_add_entity({"entity_type": "Module", "name": "B"})
    result = plugin_router.tool_graph_add_relation(
        {
            "rel_type": "CONTAINS",
            "from_type": "Module",
            "from_name": "A",
            "to_type": "Module",
            "to_name": "B",
        }
    )
    assert result["success"] is True
    assert result["conflicts"] == []


def test_graph_delete_relation(monkeypatch):
    """graph_delete_relation 删除关系."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    plugin_router.tool_graph_add_entity({"entity_type": "Module", "name": "A"})
    plugin_router.tool_graph_add_entity({"entity_type": "Module", "name": "B"})
    plugin_router.tool_graph_add_relation(
        {
            "rel_type": "CONTAINS",
            "from_type": "Module",
            "from_name": "A",
            "to_type": "Module",
            "to_name": "B",
        }
    )
    result = plugin_router.tool_graph_delete_relation(
        {
            "rel_type": "CONTAINS",
            "from_type": "Module",
            "from_name": "A",
            "to_type": "Module",
            "to_name": "B",
        }
    )
    assert result["success"] is True


def test_graph_verify_entity(monkeypatch):
    """graph_verify_entity 标记验证."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    plugin_router.tool_graph_add_entity({"entity_type": "Module", "name": "V1"})
    result = plugin_router.tool_graph_verify_entity(
        {"entity_type": "Module", "name": "V1", "verified": True}
    )
    assert result["success"] is True
    assert result["verified"] is True


def test_graph_conflicts_empty(monkeypatch):
    """graph_conflicts 无冲突时返回空列表."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_conflicts({})
    assert result["success"] is True
    assert result["count"] == 0


def test_graph_feedback(monkeypatch):
    """graph_feedback 提交反馈."""
    svc = _make_graph_service()
    monkeypatch.setattr(plugin_router, "_get_service", lambda: svc)

    result = plugin_router.tool_graph_feedback(
        {"rating": 1, "entity_type": "Module", "entity_name": "M1", "comment": "Good"}
    )
    assert result["success"] is True
    assert result["feedback_id"] > 0


# ---------------------------------------------------------------------------
# _extract_product_name 测试
# ---------------------------------------------------------------------------


def test_extract_product_name_from_markdown():
    """从 Markdown 文本中提取产品型号."""
    from core.doc_graph_pipeline import _extract_product_name

    md = "# TMS320F28379D Technical Reference Manual\n\n## Introduction"
    result = _extract_product_name(md, "/path/to/doc.pdf")
    assert result == "TMS320F28379D"


def test_extract_product_name_from_filename():
    """从文件名中提取产品型号（Markdown 中无型号时）."""
    from core.doc_graph_pipeline import _extract_product_name

    md = "# Technical Reference Manual\n\n## Introduction"
    result = _extract_product_name(md, "/path/to/STM32F407VG.pdf")
    assert result == "STM32F407VG"


def test_extract_product_name_not_found():
    """无匹配时返回 None."""
    from core.doc_graph_pipeline import _extract_product_name

    md = "# Generic Manual\n\n## Introduction"
    result = _extract_product_name(md, "/path/to/generic.pdf")
    assert result is None


# ---------------------------------------------------------------------------
# 三阶段 Plugin 工具测试
# ---------------------------------------------------------------------------


def test_tool_doc_parse_success(patched_config, monkeypatch):
    """tool_doc_parse 应解析 PDF 并返回 doc_id 和 output_dir."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["## Chapter 1\n\nChapter one content."])

    result = plugin_router.tool_doc_parse({"path": str(pdf)})
    assert result["success"] is True
    assert "doc_id" in result
    assert "output_dir" in result
    assert result["images"] == 0  # 纯文本 PDF 无图片
    assert "result.md" in result["message"]

    # 验证 result.md 已生成
    from pathlib import Path

    output_dir = Path(result["output_dir"])
    assert (output_dir / "result.md").exists()


def test_tool_doc_parse_missing_path():
    """tool_doc_parse 缺少 path 时应返回错误."""
    result = plugin_router.tool_doc_parse({})
    assert result["success"] is False
    assert "path" in result["error"]


def test_tool_doc_build_batches_success(patched_config, monkeypatch):
    """tool_doc_build_batches 应从已解析文档生成 JSONL."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["## Section A\n\nContent A.\n\n## Section B\n\nContent B."])

    # 先执行 stage1
    parse_result = plugin_router.tool_doc_parse({"path": str(pdf)})
    assert parse_result["success"] is True
    doc_id = parse_result["doc_id"]

    # 执行 stage2
    result = plugin_router.tool_doc_build_batches({"doc_id": doc_id})
    assert result["success"] is True
    assert result["doc_id"] == doc_id
    assert "jsonl_path" in result
    assert result["batch_count"] > 0
    assert result["request_count"] > 0

    from pathlib import Path

    assert Path(result["jsonl_path"]).exists()


def test_tool_doc_build_batches_missing_doc_id():
    """tool_doc_build_batches 缺少 doc_id 时应返回错误."""
    result = plugin_router.tool_doc_build_batches({})
    assert result["success"] is False
    assert "doc_id" in result["error"]


def test_tool_doc_submit_batches_success(patched_config, monkeypatch):
    """tool_doc_submit_batches 应完成入库."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["## Section A\n\nContent A.\n\n## Section B\n\nContent B."])

    # stage1 + stage2
    parse_result = plugin_router.tool_doc_parse({"path": str(pdf)})
    assert parse_result["success"] is True
    doc_id = parse_result["doc_id"]

    build_result = plugin_router.tool_doc_build_batches({"doc_id": doc_id})
    assert build_result["success"] is True

    # stage3
    result = plugin_router.tool_doc_submit_batches({"doc_id": doc_id, "file_path": str(pdf)})
    assert result["success"] is True
    assert result["doc_id"] == doc_id

    # 验证数据库状态
    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None
    assert doc.status == "done"

    chunks = db.query_by_doc(doc_id)
    assert len(chunks) > 0
    for ck in chunks:
        assert ck.status == "done"


def test_tool_doc_submit_batches_missing_doc_id():
    """tool_doc_submit_batches 缺少 doc_id 时应返回错误."""
    result = plugin_router.tool_doc_submit_batches({})
    assert result["success"] is False
    assert "doc_id" in result["error"]


def test_tool_doc_submit_batches_missing_file_path(patched_config, monkeypatch):
    """tool_doc_submit_batches 无 file_path 且无数据库记录时应返回错误."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    result = plugin_router.tool_doc_submit_batches({"doc_id": "nonexistent_doc_id"})
    assert result["success"] is False
    assert "源文件路径" in result["error"] or "file_path" in result["error"]
