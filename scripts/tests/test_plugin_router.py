"""
Tests for plugin_router.py covering recent bug fixes:
- FlowSelector integration (LLM_API_FLOW vs AGENT_SKILL_FLOW)
- get_content tool
- plugin.json format regression
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest

from core.config import Config
from core.db import KnowledgeDB
from core.models import Chunk, Document

_SKILL_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_SKILL_ROOT))

import plugin_router


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def patched_config(monkeypatch, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    monkeypatch.setattr(Config, "DB_PATH", kb / "workdocs.db")
    monkeypatch.setattr(Config, "FAISS_INDEX_PATH", kb / "faiss.index")
    monkeypatch.setattr(Config, "ID_MAP_PATH", kb / "id_map.json")
    monkeypatch.setattr(Config, "BATCH_SIZE", 2)
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(Config, "PROMPT_DIR", _SKILL_ROOT / "prompts")
    return tmp_path


class FakeEmbedder:
    def __init__(self):
        self._dim_validated = True

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def get_embedding_dimension(self):
        return 4

    def close(self):
        pass


class FakeLLMClient:
    def chat(self, messages, **kwargs):
        return "Summary: test summary\nKeywords: keyword1, keyword2"

    def summarize(self, text):
        return {"summary": "test summary", "keywords": ["keyword1", "keyword2"]}

    def vision_describe(self, path, prompt):
        return "image analysis"

    def close(self):
        pass


def _mock_llm_and_embedder(monkeypatch):
    """Mock LLMChatClient and EmbeddingClient for LLMAPIIngestionPipeline."""
    fake_llm = FakeLLMClient()
    fake_embed = FakeEmbedder()
    monkeypatch.setattr(
        "core.llm_api_pipeline.LLMChatClient",
        lambda: fake_llm,
    )
    monkeypatch.setattr(
        "core.llm_api_pipeline.EmbeddingClient",
        lambda: fake_embed,
    )
    monkeypatch.setattr(
        "core.pipeline.EmbeddingClient",
        lambda: fake_embed,
    )
    monkeypatch.setattr(
        "core.compatibility_pipeline.EmbeddingClient",
        lambda: fake_embed,
    )


# ---------------------------------------------------------------------------
# FlowSelector integration
# ---------------------------------------------------------------------------

def test_ingest_with_llm_config_uses_llm_pipeline(patched_config, monkeypatch):
    """When LLM is configured, ingest should use LLMAPIIngestionPipeline
    and mark chunks as done with summary/keywords persisted."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    _mock_llm_and_embedder(monkeypatch)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one content", "Page two content"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    assert result["success"] is True
    doc_id = result["doc_ids"][0]

    db = KnowledgeDB()
    chunks = db.query_by_page(doc_id, 1, 2)
    assert len(chunks) > 0
    # LLM_API_FLOW should persist summary and mark as done
    for ck in chunks:
        assert ck.status == "done"
        assert ck.summary is not None and len(ck.summary) > 0


def test_ingest_agent_mode_forces_compat_pipeline(patched_config, monkeypatch):
    """agent_mode=true should force CompatibilityIngestionPipeline,
    leaving chunks in 'embedded' status."""
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    fake_embed = FakeEmbedder()
    monkeypatch.setattr("core.pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.compatibility_pipeline.EmbeddingClient", lambda: fake_embed)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Hello world"])

    result = plugin_router.tool_ingest({"path": str(pdf), "agent_mode": True})
    assert result["success"] is True
    doc_id = result["doc_ids"][0]

    db = KnowledgeDB()
    chunks = db.query_by_page(doc_id, 1, 1)
    assert len(chunks) == 1
    assert chunks[0].status == "embedded"
    assert (chunks[0].summary is None or chunks[0].summary == "")


def test_reprocess_uses_flow_selector(patched_config, monkeypatch):
    """tool_reprocess should use FlowSelector, not hardcode IngestionPipeline."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
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
    chunks = db.query_by_page(doc_id, 1, 1)
    assert chunks[0].status == "done"


# ---------------------------------------------------------------------------
# get_content tool
# ---------------------------------------------------------------------------

def test_get_content_by_chapter(patched_config, monkeypatch):
    fake_embed = FakeEmbedder()
    monkeypatch.setattr("core.pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.compatibility_pipeline.EmbeddingClient", lambda: fake_embed)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Chapter Alpha text", "Chapter Alpha more"])

    result = plugin_router.tool_ingest({"path": str(pdf), "agent_mode": True})
    doc_id = result["doc_ids"][0]

    # PDF without TOC gets a single chapter titled "全文"
    result = plugin_router.tool_get_content({"doc_id": doc_id, "chapter": "全文"})
    assert result["success"] is True
    assert result["query_type"] == "chapter"
    assert "Alpha" in result["content"]
    assert result["total_chars"] > 0
    assert len(result["chunks"]) > 0


def test_get_content_by_page(patched_config, monkeypatch):
    fake_embed = FakeEmbedder()
    monkeypatch.setattr("core.pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.compatibility_pipeline.EmbeddingClient", lambda: fake_embed)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one text", "Page two text"])

    result = plugin_router.tool_ingest({"path": str(pdf), "agent_mode": True})
    doc_id = result["doc_ids"][0]

    result = plugin_router.tool_get_content({"doc_id": doc_id, "page": "1"})
    assert result["success"] is True
    assert result["query_type"] == "page"
    assert result["page_start"] == 1
    assert "Page one" in result["content"]


def test_get_content_by_chunk_db_id(patched_config, monkeypatch):
    fake_embed = FakeEmbedder()
    monkeypatch.setattr("core.pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.compatibility_pipeline.EmbeddingClient", lambda: fake_embed)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Single page"])

    result = plugin_router.tool_ingest({"path": str(pdf), "agent_mode": True})
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
    result = plugin_router.tool_get_content({})
    assert result["success"] is False
    assert "Provide page, chapter, or chunk_db_id" in result["error"]


# ---------------------------------------------------------------------------
# plugin.json regression
# ---------------------------------------------------------------------------

def test_plugin_json_all_commands_use_venv_python():
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
    plugin_path = _SKILL_ROOT.parent / "plugin.json"
    data = json.loads(plugin_path.read_text(encoding="utf-8"))
    assert data.get("name") == "work-docs-library"
    assert "version" in data
    assert "tools" in data
    names = [t["name"] for t in data["tools"]]
    expected = [
        "ingest", "search", "query", "status", "toc",
        "auto_summarize", "synthesize_chapters", "progress",
        "reprocess", "get_content",
    ]
    for name in expected:
        assert name in names, f"Missing tool: {name}"


# ---------------------------------------------------------------------------
# Failure recovery and resume
# ---------------------------------------------------------------------------

class FailingLLMClient:
    """LLM client that always raises RuntimeError."""
    def chat(self, messages, **kwargs):
        raise RuntimeError("LLM chat failed")
    def summarize(self, text):
        raise RuntimeError("LLM summarize failed")
    def vision_describe(self, path, prompt):
        raise RuntimeError("LLM vision failed")
    def close(self):
        pass


def test_ingest_failure_sets_status_failed(patched_config, monkeypatch):
    """If LLM enhancement fails, document status should become 'failed'
    and chunks should remain 'embedded' (Phase A completed)."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    fake_embed = FakeEmbedder()
    monkeypatch.setattr("core.llm_api_pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.compatibility_pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.llm_api_pipeline.LLMChatClient", lambda: FailingLLMClient())

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page one content"])

    with pytest.raises(RuntimeError):
        plugin_router.tool_ingest({"path": str(pdf)})

    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None
    assert doc.status == "failed"
    chunks = db.query_by_page(doc.doc_id, 1, 1)
    assert len(chunks) == 1
    assert chunks[0].status == "embedded"


def test_ingest_resume_skips_phase_a(patched_config, monkeypatch):
    """If chunks are already 'embedded', Phase A should be skipped and
    Phase B should resume, marking chunks as 'done'."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
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
        conn.execute("DELETE FROM chapter_summaries WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM concept_index WHERE doc_id = ?", (doc_id,))

    before_count = len(db.query_by_page(doc_id, 1, 1))

    # Second ingest - should resume Phase B only
    result = plugin_router.tool_ingest({"path": str(pdf)})
    assert result["success"] is True

    after_count = len(db.query_by_page(doc_id, 1, 1))
    assert after_count == before_count, "Phase A should be skipped, no new chunks inserted"

    chunks = db.query_by_page(doc_id, 1, 1)
    assert chunks[0].status == "done"
    assert chunks[0].summary != ""


# ---------------------------------------------------------------------------
# Image analysis persistence
# ---------------------------------------------------------------------------

def test_image_analysis_persisted_to_chunk_metadata(patched_config, monkeypatch):
    """Image analysis results from LLM vision should be saved into
    the chunk's metadata JSON under the 'vision_desc' key."""
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-llm-key")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(Config, "EMBEDDING_API_KEY", "fake-emb-key")
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

    fake_embed = FakeEmbedder()
    monkeypatch.setattr("core.llm_api_pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.pipeline.EmbeddingClient", lambda: fake_embed)
    monkeypatch.setattr("core.compatibility_pipeline.EmbeddingClient", lambda: fake_embed)

    fake_llm = FakeLLMClient()
    monkeypatch.setattr("core.llm_api_pipeline.LLMChatClient", lambda: fake_llm)

    # Mock _analyze_document_images to inject a fake analysis result
    from core.llm_api_pipeline import LLMAPIIngestionPipeline
    original_analyze = LLMAPIIngestionPipeline._analyze_document_images

    def mock_analyze(self, doc):
        return [{
            "image_id": "img_1",
            "path": "/fake/path.png",
            "analysis": "This is a test image analysis",
            "chunk_id": doc.chunks[0].chunk_id if doc.chunks else "c1",
            "page": 1,
        }]

    monkeypatch.setattr(LLMAPIIngestionPipeline, "_analyze_document_images", mock_analyze)

    pdf = patched_config / "doc.pdf"
    _make_pdf(pdf, ["Page with image placeholder"])

    result = plugin_router.tool_ingest({"path": str(pdf)})
    doc_id = result["doc_ids"][0]

    db = KnowledgeDB()
    chunks = db.query_by_page(doc_id, 1, 1)
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert "images" in meta
    assert any(img.get("vision_desc") == "This is a test image analysis" for img in meta["images"])
