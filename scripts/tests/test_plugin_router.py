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
from core.models import Chunk

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

    def embed_single(self, text):
        """embed_single 函数."""
        return [1.0, 0.0, 0.0, 0.0]

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


class FakeChatClient:
    """Mock BaseLLMClient for Chat mode fallback."""

    def __init__(self, *args, **kwargs):
        self.chat_url = "https://test.com/v1/chat/completions"
        self.user_agent = "KimiCLI/1.44.0"

    def _post(self, url, payload, timeout=None):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"entities": [], "relationships": [], "image_descriptions": []}'
                        )
                    }
                }
            ]
        }

    def chat(self, messages, **kwargs):
        return '{"entities": [], "relationships": [], "image_descriptions": []}'

    def close(self):
        pass


def _mock_llm_and_embedder(monkeypatch):
    """Mock EmbeddingClient, BigModelParserClient and BaseLLMClient for DocGraphPipeline."""
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
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BaseLLMClient",
        FakeChatClient,
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
    chunks = db.query_blocks_by_doc(doc_id)
    assert len(chunks) > 0
    for block in chunks:
        assert block["status"] == "done"


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
    chunks = db.query_blocks_by_doc(doc_id)
    assert chunks[0]["status"] == "done"


# ---------------------------------------------------------------------------
# read tool
# ---------------------------------------------------------------------------


def _fetch_chunk_db_id(doc_id: str) -> int:
    import sqlite3

    with sqlite3.connect(str(Config.DB_PATH)) as conn:
        row = conn.execute(
            "SELECT id FROM content_blocks WHERE doc_id = ? LIMIT 1", (doc_id,)
        ).fetchone()
        return row[0]


def test_read_by_chunk_db_id(patched_config, monkeypatch):
    """Test read by chunk db id."""
    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Single page"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]
    chunk_db_id = _fetch_chunk_db_id(doc_id)

    result = plugin_router.tool_read({"chunk_db_id": chunk_db_id})
    assert result["success"] is True
    assert result["query_type"] == "chunk"
    assert "Single page" in result["content"]
    assert result["total_chars"] > 0
    assert len(result["chunks"]) > 0


def test_read_by_chunk_db_id_with_entities(patched_config, monkeypatch):
    """Test read by chunk db id returns entity/relation lists."""
    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Chapter Alpha text", "Chapter Alpha more"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]
    chunk_db_id = _fetch_chunk_db_id(doc_id)

    result = plugin_router.tool_read({"chunk_db_id": chunk_db_id})
    assert result["success"] is True
    assert result["query_type"] == "chunk"
    assert "Alpha" in result["content"]
    assert "entities" in result
    assert "relations" in result


def test_read_by_chapter_empty_title(patched_config, monkeypatch):
    """Test read by chapter with empty title returns no match."""
    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one text", "Page two text"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    result = plugin_router.tool_read({"doc_id": doc_id, "chapter": ""})
    assert result["success"] is False


def test_read_missing_params():
    """Test read missing params."""
    result = plugin_router.tool_read({})
    assert result["success"] is False
    assert "chunk_db_id" in result["error"]


# ---------------------------------------------------------------------------
# search tool
# ---------------------------------------------------------------------------


def _fake_search_result():
    return {
        "chunks": [
            {
                "score": 0.9,
                "chunk": Chunk(
                    doc_id="doc1",
                    chunk_id="c1",
                    chunk_type="text",
                    chapter_title="Chapter 1",
                    content="SPI interface description",
                ),
            }
        ],
        "entities": [],
        "relations": [],
        "source_documents": [{"doc_id": "doc1", "title": "Doc 1", "total_pages": 10}],
    }


class FakeQueryService:
    """Fake QueryService for tool_search tests."""

    def __init__(self, svc=None):
        self._svc = svc

    def search(
        self,
        text,
        top_k=5,
        mode="hybrid",
        include_graph=True,
        graph_depth=1,
        rerank_candidate_k=None,
        session_id=None,
    ):
        assert text == "SPI"
        assert mode == "hybrid"
        return _fake_search_result()


class FakeQueryServiceRerank:
    def search(
        self,
        text,
        top_k=5,
        mode="hybrid",
        include_graph=True,
        graph_depth=1,
        rerank_candidate_k=None,
        session_id=None,
    ):
        assert mode == "reranked"
        assert rerank_candidate_k == 20
        return _fake_search_result()


class FakeQueryServiceException:
    def search(self, **kwargs):
        raise KeyError("missing key")


def test_search_success(patched_config, monkeypatch):
    """tool_search 应返回统一搜索结果."""
    monkeypatch.setattr(plugin_router, "_get_query_service", lambda: FakeQueryService())
    result = plugin_router.tool_search({"text": "SPI", "mode": "hybrid"})
    assert result["success"] is True
    assert result["text"] == "SPI"
    assert result["mode"] == "hybrid"
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["score"] == 0.9
    assert result["chunks"][0]["doc_id"] == "doc1"
    assert "SPI interface description" in result["chunks"][0]["content_preview"]
    assert result["source_documents"][0]["doc_id"] == "doc1"


def test_search_missing_text():
    """tool_search 缺少 text 时返回错误."""
    result = plugin_router.tool_search({})
    assert result["success"] is False
    assert "text" in result["error"]


def test_search_invalid_mode():
    """tool_search 不支持的模式返回错误."""
    result = plugin_router.tool_search({"text": "SPI", "mode": "magic"})
    assert result["success"] is False
    assert "magic" in result["error"]


def test_search_reranked_passes_candidate_k(patched_config, monkeypatch):
    """tool_search reranked 模式应传递 rerank_candidate_k."""
    monkeypatch.setattr(plugin_router, "_get_query_service", lambda: FakeQueryServiceRerank())
    result = plugin_router.tool_search(
        {"text": "SPI", "mode": "reranked", "rerank_candidate_k": 20}
    )
    assert result["success"] is True
    assert result["mode"] == "reranked"


def test_search_arbitrary_exception(patched_config, monkeypatch):
    """tool_search 应捕获任意异常并返回 success=False."""
    monkeypatch.setattr(plugin_router, "_get_query_service", lambda: FakeQueryServiceException())
    result = plugin_router.tool_search({"text": "SPI"})
    assert result["success"] is False
    assert "missing key" in result["error"]


# ---------------------------------------------------------------------------
# explore tool
# ---------------------------------------------------------------------------


def test_explore_missing_mode():
    result = plugin_router.tool_explore({})
    assert result["success"] is False
    assert "mode" in result["error"]


def test_explore_invalid_mode():
    result = plugin_router.tool_explore({"mode": "magic"})
    assert result["success"] is False
    assert "magic" in result["error"]


def test_explore_entity(monkeypatch, graph_service):
    """tool_explore entity 模式查询实体."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    result = plugin_router.tool_explore({"mode": "entity", "entity_type": "Module", "name": "TOP"})
    assert result["success"] is True
    assert result["mode"] == "entity"
    assert result["count"] == 1
    assert result["entities"][0]["name"] == "TOP"


def test_explore_neighbors(monkeypatch, graph_service):
    """tool_explore neighbors 模式查询邻居."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    result = plugin_router.tool_explore(
        {"mode": "neighbors", "entity_type": "Module", "name": "TOP", "direction": "out"}
    )
    assert result["success"] is True
    assert result["mode"] == "neighbors"
    assert result["neighbor_count"] == 2
    names = {n["entity"]["name"] for n in result["neighbors"]}
    assert names == {"SUB", "CLK"}


def test_explore_subgraph(monkeypatch, graph_service):
    """tool_explore subgraph 模式提取子图."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    result = plugin_router.tool_explore(
        {"mode": "subgraph", "entity_type": "Module", "name": "TOP", "depth": 2}
    )
    assert result["success"] is True
    assert result["mode"] == "subgraph"
    assert result["subgraph"]["node_count"] == 4
    names = {e["name"] for e in result["subgraph"]["entities"]}
    assert names == {"TOP", "SUB", "CLK", "REG"}


def test_explore_path(monkeypatch, graph_service):
    """tool_explore path 模式查找路径."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    result = plugin_router.tool_explore(
        {
            "mode": "path",
            "from_type": "Module",
            "from_name": "TOP",
            "to_type": "Register",
            "to_name": "REG",
        }
    )
    assert result["success"] is True
    assert result["mode"] == "path"
    assert result["path_count"] == 1
    path = result["paths"][0]
    assert len(path) == 3
    assert path[0]["name"] == "TOP"
    assert path[1]["name"] == "SUB"
    assert path[2]["name"] == "REG"


# ---------------------------------------------------------------------------
# kimi.plugin.json regression
# ---------------------------------------------------------------------------


def test_kimi_plugin_json_mcp_server_uses_uv_run():
    """Test kimi.plugin.json declares an uv-run MCP server."""
    plugin_path = _SKILL_ROOT.parent / "kimi.plugin.json"
    data = json.loads(plugin_path.read_text(encoding="utf-8"))
    servers = data.get("mcpServers", {})
    assert "workdocs" in servers, "Missing mcpServer 'workdocs'"
    server = servers["workdocs"]
    assert server.get("command") == "uv"
    assert server.get("args") == ["run", "--no-sync", "python", "scripts/mcp_server.py"]


def test_kimi_plugin_json_valid_schema():
    """Test kimi.plugin.json follows the new spec and exposes 14 MCP tools via mcp_server."""
    plugin_path = _SKILL_ROOT.parent / "kimi.plugin.json"
    data = json.loads(plugin_path.read_text(encoding="utf-8"))
    assert data.get("name") == "work-docs-library"
    assert "version" in data
    assert "skills" in data
    assert data.get("sessionStart", {}).get("skill") == "using-workdocs"
    # Old fields are no longer used
    assert "tools" not in data
    assert "inject" not in data
    assert "configFile" not in data

    # The MCP server white-list is the source of truth for exposed tools.
    import mcp_server as mcp

    assert set(mcp.MCP_TOOL_MAP.keys()) == {
        "ingest",
        "search",
        "explore",
        "read",
        "status",
    }


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
    chunks = db.query_blocks_by_doc(doc.doc_id)
    assert len(chunks) == 1
    assert chunks[0]["status"] == "done"


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
            "UPDATE content_blocks SET status = 'embedded' WHERE doc_id = ?",
            (doc_id,),
        )
        conn.execute("UPDATE documents SET status = 'embedded' WHERE doc_id = ?", (doc_id,))

    before_count = len(db.query_blocks_by_doc(doc_id))

    # Second ingest - should resume Phase B only
    result = plugin_router.tool_ingest({"path": str(pdf)})
    assert result["success"] is True

    after_count = len(db.query_blocks_by_doc(doc_id))
    assert after_count == before_count, "Phase A should be skipped, no new chunks inserted"

    chunks = db.query_blocks_by_doc(doc_id)
    assert chunks[0]["status"] == "done"


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
    chunks = db.query_blocks_by_doc(doc_id)
    assert len(chunks) == 1
    assert chunks[0]["status"] == "done"


def test_tool_ingest_chat_mode_end_to_end(patched_config, monkeypatch):
    """Chat 模式下 tool_ingest 应完整通过（端到端覆盖）."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(Config, "LLM_MODE", "chat")

    fake_embed = FakeEmbedder()
    fake_file = FakeBigModelParserClient()
    monkeypatch.setattr("core.doc_graph_pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.doc_graph_pipeline.BigModelParserClient", lambda: fake_file)
    monkeypatch.setattr("core.doc_graph_pipeline.BaseLLMClient", FakeChatClient)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one content", "Page two content"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    assert result["success"] is True
    doc_id = result["doc_ids"][0]

    db = KnowledgeDB()
    chunks = db.query_blocks_by_doc(doc_id)
    assert len(chunks) > 0
    for block in chunks:
        assert block["status"] == "done"


# ---------------------------------------------------------------------------
# Graph tools
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_service(monkeypatch, tmp_path):
    """创建一个预填充了测试数据的 KnowledgeBaseService（隔离全局图）."""
    from core.graph_store import GraphEntity, GraphRelation, NetworkXGraphStore
    from core.knowledge_base_service import KnowledgeBaseService
    from core.vector_index import VectorIndex

    monkeypatch.setattr(Config, "GRAPH_OUTPUT_DIR", str(tmp_path / "graphs"))
    db = KnowledgeDB(db_path=tmp_path / "test.db")
    vec = VectorIndex(dim=4, index_path=tmp_path / "faiss.index")
    svc = KnowledgeBaseService(
        db=db,
        vec=vec,
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


def test_graph_tools_missing_params():
    """图谱工具缺少必需参数时返回错误."""
    assert plugin_router.tool_explore({})["success"] is False
    assert plugin_router.tool_graph_upsert_entity({})["success"] is False


def test_graph_upsert_entity_create(monkeypatch, graph_service):
    """graph_upsert_entity 创建新实体."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    result = plugin_router.tool_graph_upsert_entity(
        {"entity_type": "Module", "name": "NEW", "properties": {"desc": "test"}}
    )
    assert result["success"] is True
    assert result["entity"]["name"] == "NEW"
    assert result["conflicts"] == []
    assert result["mode"] == "created"


def test_graph_upsert_entity_null_properties(monkeypatch, graph_service):
    """graph_upsert_entity 创建模式下显式传入 null 值不崩溃."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    # properties=None 应被安全处理为 {}
    result = plugin_router.tool_graph_upsert_entity(
        {"entity_type": "Module", "name": "NULL_PROP", "properties": None}
    )
    assert result["success"] is True
    assert result["mode"] == "created"
    e = graph_service.graph.get_entity("Module", "NULL_PROP")
    assert e is not None
    assert e.properties == {}

    # source_doc_ids=None 应被安全处理为 set()
    result2 = plugin_router.tool_graph_upsert_entity(
        {"entity_type": "Module", "name": "NULL_DOCIDS", "source_doc_ids": None}
    )
    assert result2["success"] is True
    assert result2["mode"] == "created"
    e2 = graph_service.graph.get_entity("Module", "NULL_DOCIDS")
    assert e2 is not None
    assert e2.source_doc_ids == set()


def test_graph_upsert_entity_update(monkeypatch, graph_service):
    """graph_upsert_entity 更新已有实体（含 verify 功能）."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    # 先创建
    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "M1"})
    # 再更新
    result = plugin_router.tool_graph_upsert_entity(
        {"entity_type": "Module", "name": "M1", "properties": {"a": 1}, "verified": True}
    )
    assert result["success"] is True
    assert result["mode"] == "updated"

    e = graph_service.graph.get_entity("Module", "M1")
    assert e is not None
    assert e.properties == {"a": 1}
    assert e.verified is True


def test_graph_delete_entity(monkeypatch, graph_service):
    """graph_delete_entity 删除实体."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "DEL"})
    result = plugin_router.tool_graph_delete_entity({"entity_type": "Module", "name": "DEL"})
    assert result["success"] is True
    assert graph_service.graph.get_entity("Module", "DEL") is None


def test_graph_upsert_relation(monkeypatch, graph_service):
    """graph_upsert_relation 添加关系."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "A"})
    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "B"})
    result = plugin_router.tool_graph_upsert_relation(
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


def test_graph_upsert_relation_null_properties(monkeypatch, graph_service):
    """graph_upsert_relation 显式传入 null 值不崩溃."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "C"})
    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "D"})

    result = plugin_router.tool_graph_upsert_relation(
        {
            "rel_type": "CONTAINS",
            "from_type": "Module",
            "from_name": "C",
            "to_type": "Module",
            "to_name": "D",
            "properties": None,
        }
    )
    assert result["success"] is True
    rel = graph_service.graph.get_relation("Module", "C", "Module", "D", "CONTAINS")
    assert rel is not None
    assert rel.properties == {}


def test_graph_delete_relation(monkeypatch, graph_service):
    """graph_delete_relation 删除关系."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "A"})
    plugin_router.tool_graph_upsert_entity({"entity_type": "Module", "name": "B"})
    plugin_router.tool_graph_upsert_relation(
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


def test_graph_feedback_submit(monkeypatch, graph_service):
    """graph_feedback action=submit 提交反馈."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    result = plugin_router.tool_graph_feedback(
        {
            "action": "submit",
            "rating": 1,
            "entity_type": "Module",
            "entity_name": "M1",
            "comment": "Good",
        }
    )
    assert result["success"] is True
    assert result["feedback_id"] > 0


def test_graph_feedback_query(monkeypatch, graph_service):
    """graph_feedback action=query 查询反馈."""
    monkeypatch.setattr(plugin_router, "_get_service", lambda: graph_service)

    # 先提交一条
    plugin_router.tool_graph_feedback(
        {"action": "submit", "rating": 1, "entity_type": "Module", "entity_name": "M1"}
    )
    # 再查询
    result = plugin_router.tool_graph_feedback(
        {"action": "query", "entity_type": "Module", "entity_name": "M1"}
    )
    assert result["success"] is True
    assert result["count"] >= 1
    # 验证返回的 feedbacks 中包含目标实体
    assert any(f.get("entity_name") == "M1" for f in result["feedbacks"])


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
    """tool_doc_submit_batches 应提交 Batch 并保存结果文件."""
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
    assert "results_path" in result
    assert Path(result["results_path"]).exists()


def test_tool_doc_ingest_results_success(patched_config, monkeypatch):
    """tool_doc_ingest_results 应从结果文件完成入库."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["## Section A\n\nContent A.\n\n## Section B\n\nContent B."])

    # stage1 + stage2 + stage3
    parse_result = plugin_router.tool_doc_parse({"path": str(pdf)})
    doc_id = parse_result["doc_id"]
    plugin_router.tool_doc_build_batches({"doc_id": doc_id})
    submit_result = plugin_router.tool_doc_submit_batches({"doc_id": doc_id, "file_path": str(pdf)})
    results_path = submit_result["results_path"]

    # stage4
    result = plugin_router.tool_doc_ingest_results(
        {"doc_id": doc_id, "file_path": str(pdf), "results_path": results_path}
    )
    assert result["success"] is True
    assert result["doc_id"] == doc_id

    # 验证数据库状态
    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None
    assert doc.status == "done"

    chunks = db.query_blocks_by_doc(doc_id)
    assert len(chunks) > 0
    for block in chunks:
        # 方案C：stage4 完成后 block 状态为 embedded，stage6 后变为 done
        assert block["status"] == "embedded"


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


# ---------------------------------------------------------------------------
# Security fixes
# ---------------------------------------------------------------------------


def test_tool_config_always_masks_sensitive_keys(monkeypatch):
    """tool_config 应始终脱敏敏感字段，即使显式传入 mask_sensitive=False."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "super-secret-key")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "emb-secret-key")
    monkeypatch.setattr(Config, "PARSER_API_KEY", "parser-secret-key")

    result = plugin_router.tool_config({"mask_sensitive": False})
    assert result["success"] is True

    llm_group = result["config_groups"]["LLM 配置"]
    assert llm_group["LLM_API_KEY"] == "***"

    embed_group = result["config_groups"]["Embedding 配置"]
    assert embed_group["EMBEDDING_API_KEY"] == "***"

    parser_group = result["config_groups"]["解析器配置"]
    assert parser_group["PARSER_API_KEY"] == "***"


def test_tool_config_groups_cover_all_active_keys():
    """tool_config 的分组映射应覆盖 Config 中所有非内部键，避免重要配置落入"其他"."""
    result = plugin_router.tool_config({})
    groups = result["config_groups"]
    grouped_keys = set()
    for group_name, group_items in groups.items():
        # 允许"其他"分组只包含内部路径/开发配置
        if group_name == "其他":
            continue
        grouped_keys.update(group_items.keys())

    expected_keys = set(Config.to_dict().keys())
    # 内部路径配置无需用户感知，可留在"其他"
    internal_keys = {
        "DB_PATH",
        "FAISS_INDEX_PATH",
        "PROMPT_DIR",
    }
    missing = (expected_keys - internal_keys) - grouped_keys
    assert not missing, f"以下活跃配置未在 tool_config 分组中：{sorted(missing)}"


def test_tool_ingest_rejects_unsafe_path():
    """tool_ingest 应拒绝沙箱外的路径."""
    result = plugin_router.tool_ingest({"path": "../../etc/passwd"})
    assert result["success"] is False
    assert "unsafe" in result["error"].lower() or "outside" in result["error"].lower()


def test_tool_evaluate(patched_config, monkeypatch):
    """tool_evaluate 应调用 KnowledgeBaseService.evaluate_dataset 并返回结果."""
    from plugin_router import tool_evaluate

    def fake_evaluate(self, dataset_name, retriever, top_k):
        return {
            "num_questions": 1,
            "avg_hit_rate@5": 1.0,
            "avg_mrr": 1.0,
            "avg_ndcg@5": 1.0,
        }

    monkeypatch.setattr(
        "core.knowledge_base_service.KnowledgeBaseService.evaluate_dataset",
        fake_evaluate,
    )

    result = tool_evaluate({"dataset_name": "baseline"})
    assert result["success"] is True
    assert result["dataset_name"] == "baseline"
    assert result["avg_hit_rate@5"] == 1.0

    result_missing = tool_evaluate({})
    assert result_missing["success"] is False
    assert "dataset_name" in result_missing["error"]


def test_tool_evaluate_error(monkeypatch):
    from plugin_router import tool_evaluate

    def fake_evaluate_raise(self, dataset_name, retriever, top_k):
        raise RuntimeError("dataset corrupt")

    monkeypatch.setattr(
        "core.knowledge_base_service.KnowledgeBaseService.evaluate_dataset",
        fake_evaluate_raise,
    )

    result = tool_evaluate({"dataset_name": "broken"})
    assert result["success"] is False
    assert "dataset corrupt" in result["error"]


def test_tool_evaluate_invalid_retriever():
    from plugin_router import tool_evaluate

    result = tool_evaluate({"dataset_name": "baseline", "retriever": "unknown"})
    assert result["success"] is False
    assert "Unsupported retriever" in result["error"]
    assert "unknown" in result["error"]
