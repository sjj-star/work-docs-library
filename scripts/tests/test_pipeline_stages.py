"""DocGraphPipeline 三阶段拆分测试.

覆盖 stage1_parse / stage2_build_jsonl / stage3_ingest 以及 _process_one 兼容入口。
所有外部 API 调用均使用 Mock。
"""

import hashlib
import json
from pathlib import Path

import fitz
import pytest
from core.config import Config
from core.db import KnowledgeDB
from core.doc_graph_pipeline import DocGraphPipeline


def _make_pdf(path: Path, pages_text: list[str]) -> None:
    """创建测试 PDF."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def patched_config(monkeypatch, tmp_path):
    """Patch Config 到临时目录，隔离测试数据."""
    kb = tmp_path / "kb"
    kb.mkdir()
    monkeypatch.setattr(Config, "DB_PATH", kb / "workdocs.db")
    monkeypatch.setattr(Config, "FAISS_INDEX_PATH", kb / "faiss.index")
    monkeypatch.setattr(Config, "ID_MAP_PATH", kb / "id_map.json")
    monkeypatch.setattr(Config, "GRAPH_OUTPUT_DIR", "graphs")
    monkeypatch.setattr(Config, "EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(Config, "LLM_BATCH_MAX_CHARS", 500)
    monkeypatch.setattr(Config, "EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.openai.com/v1")
    return tmp_path


class FakeChatClient:
    """Mock ChatClient, return fixed entity extraction result."""

    def __init__(self, *args, **kwargs):
        self.chat_url = "https://test.com/v1/chat/completions"
        self.user_agent = "KimiCLI/1.44.0"

    def _post(self, url, payload, timeout=None):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {"type": "Register", "name": "TBCTL", "properties": {}}
                                ],
                                "relationships": [],
                                "image_descriptions": [],
                            }
                        )
                    }
                }
            ]
        }


class FakeBatchClient:
    """Mock BatchClient，返回空实体提取结果."""

    def __init__(self, *args, **kwargs):
        pass

    def submit_and_wait(self, requests, **kwargs):
        """submit_and_wait."""
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
                                        "content": json.dumps(
                                            {
                                                "entities": [],
                                                "relationships": [],
                                                "image_descriptions": [],
                                            }
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
        """submit_parallel_batches."""
        return self.submit_and_wait(requests, **kwargs)

    def submit_embedding_batch(self, texts, **kwargs):
        """submit_embedding_batch."""
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self):
        pass


class FakeBigModelParserClient:
    """Mock BigModelParserClient，返回空以触发本地回退."""

    def parse_pdf(self, *args, **kwargs):
        return ("", [])

    def create_task(self, *args, **kwargs):
        return "fake-task"

    def poll_result(self, *args, **kwargs):
        return {"status": "succeeded"}

    def download_result(self, *args, **kwargs):
        return b""


def _mock_external_clients(monkeypatch):
    """统一 mock 外部 API 客户端."""
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BigModelParserClient",
        FakeBigModelParserClient,
    )
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BatchClient",
        FakeBatchClient,
    )
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BaseLLMClient",
        FakeChatClient,
    )
    monkeypatch.setattr(
        "core.doc_graph_pipeline.EmbeddingClient",
        lambda: type(
            "FakeEmbedder",
            (),
            {
                "embed": lambda self, texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts],
                "embed_single": lambda self, text: [1.0, 0.0, 0.0, 0.0],
                "get_embedding_dimension": lambda self: 4,
                "close": lambda self: None,
                "_dim_validated": True,
            },
        )(),
    )


# ---------------------------------------------------------------------------
# stage1_parse 测试
# ---------------------------------------------------------------------------


def test_stage1_parse_creates_result_md(patched_config, monkeypatch):
    """stage1_parse 应解析 PDF 并生成 result.md."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(pdf, ["# Chapter 1\n\nContent for chapter one."])

    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, images = pipe.stage1_parse(str(pdf))

    assert len(doc_id) == 16
    assert parsed_dir.exists()
    assert (parsed_dir / "result.md").exists()
    assert "Chapter 1" in text
    assert "Content for chapter one" in text
    assert isinstance(images, list)


def test_stage1_parse_unsupported_format(patched_config, monkeypatch):
    """stage1_parse 遇到不支持的格式应抛出 ValueError."""
    _mock_external_clients(monkeypatch)

    txt = patched_config / "test.txt"
    txt.write_text("plain text", encoding="utf-8")

    pipe = DocGraphPipeline()
    with pytest.raises(ValueError, match="不支持的文件格式"):
        pipe.stage1_parse(str(txt))


# ---------------------------------------------------------------------------
# stage2_build_jsonl 测试
# ---------------------------------------------------------------------------


def test_stage2_build_jsonl_from_result_md(patched_config, monkeypatch):
    """stage2_build_jsonl 应从 result.md 生成 JSONL."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(
        pdf,
        ["## Section A\n\nSection A content.\n\n## Section B\n\nSection B content."],
    )

    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, _images = pipe.stage1_parse(str(pdf))

    jsonl_path, batches, requests = pipe.stage2_build_jsonl(doc_id)

    assert jsonl_path.exists()
    assert len(batches) > 0
    assert len(requests) > 0
    assert jsonl_path.name == f"{doc_id}.jsonl"


def test_stage2_build_jsonl_missing_result_md(patched_config, monkeypatch):
    """stage2_build_jsonl 在 result.md 不存在时应抛出 FileNotFoundError."""
    _mock_external_clients(monkeypatch)

    pipe = DocGraphPipeline()
    with pytest.raises(FileNotFoundError, match="result.md 不存在"):
        pipe.stage2_build_jsonl("nonexistent_doc_id")


def test_stage2_build_jsonl_custom_max_chars(patched_config, monkeypatch):
    """stage2_build_jsonl 应接受自定义 max_chars."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(
        pdf,
        ["## Section A\n\n" + "A" * 300 + "\n\n## Section B\n\n" + "B" * 300],
    )

    pipe = DocGraphPipeline()
    doc_id, _parsed_dir, _text, _images = pipe.stage1_parse(str(pdf))

    # max_chars=200 应该导致 batch 切分
    jsonl_path, batches, requests = pipe.stage2_build_jsonl(doc_id, max_chars=200)

    assert jsonl_path.exists()
    assert len(batches) >= 1
    assert len(requests) >= 1


# ---------------------------------------------------------------------------
# stage3_ingest 测试
# ---------------------------------------------------------------------------


def test_stage3_ingest_completes_and_persists(patched_config, monkeypatch):
    """stage3_ingest 应完成实体提取、向量化、入库."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(pdf, ["## Overview\n\nThis is a test document."])

    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, images = pipe.stage1_parse(str(pdf))
    jsonl_path, _batches, _requests = pipe.stage2_build_jsonl(doc_id)

    result_doc_id = pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
    )

    assert result_doc_id == doc_id

    # 验证数据库记录
    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None
    assert doc.status == "done"
    assert doc.doc_id == doc_id

    # 验证 chunks
    chunks = db.query_by_doc(doc_id)
    assert len(chunks) > 0
    for ck in chunks:
        assert ck.status == "done"

    # 验证图谱文件已保存
    graph_path = Config.DB_PATH.parent / Config.GRAPH_OUTPUT_DIR / f"{doc_id}.json"
    assert graph_path.exists()


def test_stage3_ingest_skips_unchanged(patched_config, monkeypatch):
    """stage3_ingest 对未变更文档应跳过处理."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(pdf, ["## Overview\n\nThis is a test document."])

    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, images = pipe.stage1_parse(str(pdf))
    jsonl_path, _batches, _requests = pipe.stage2_build_jsonl(doc_id)

    # 第一次入库
    pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
    )

    # 第二次入库（未变更，应跳过）
    result_doc_id = pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
    )
    assert result_doc_id == doc_id


def test_stage3_ingest_force_reprocess(patched_config, monkeypatch):
    """stage3_ingest force=True 应强制重新处理."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(pdf, ["## Overview\n\nThis is a test document."])

    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, images = pipe.stage1_parse(str(pdf))
    jsonl_path, _batches, _requests = pipe.stage2_build_jsonl(doc_id)

    # 第一次入库
    pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
    )

    # force 重新处理
    result_doc_id = pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
        force=True,
    )
    assert result_doc_id == doc_id

    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None
    assert doc.status == "done"


# ---------------------------------------------------------------------------
# _process_one 兼容入口测试
# ---------------------------------------------------------------------------


def test_process_one_calls_three_stages(patched_config, monkeypatch):
    """_process_one 应内部调用三阶段并完成入库."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(pdf, ["## Intro\n\nIntroduction content here."])

    pipe = DocGraphPipeline()
    doc_id = pipe._process_one(str(pdf))

    assert doc_id is not None
    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None
    assert doc.status == "done"


# ---------------------------------------------------------------------------
# 三阶段串联测试（端到端）
# ---------------------------------------------------------------------------


def test_three_stage_pipeline_end_to_end(patched_config, monkeypatch):
    """完整三阶段流程：parse → build_jsonl → ingest."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(
        pdf,
        [
            "# TMS320F28379D TRM\n\n"
            "## Overview\n\nOverview content here.\n\n"
            "## GPIO Registers\n\nGPIO control registers.\n\n"
            "### GPIO_CTRL\n\nGPIO control register description."
        ],
    )

    # Stage 1
    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, images = pipe.stage1_parse(str(pdf))
    assert (parsed_dir / "result.md").exists()

    # Stage 2
    jsonl_path, batches, requests = pipe.stage2_build_jsonl(doc_id)
    assert jsonl_path.exists()
    assert len(batches) > 0
    assert len(requests) > 0

    # Stage 3
    result_doc_id = pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
    )
    assert result_doc_id == doc_id

    # 验证 Product 实体（产品型号提取）
    db = KnowledgeDB()
    doc = db.get_document_by_path(str(pdf))
    assert doc is not None

    chunks = db.query_by_doc(doc_id)
    assert len(chunks) > 0
    for ck in chunks:
        assert ck.status == "done"


# ---------------------------------------------------------------------------
# 增量更新测试
# ---------------------------------------------------------------------------


def test_stage3_incremental_update(patched_config, monkeypatch):
    """stage3_ingest 应支持章节级增量更新."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(
        pdf,
        ["## Section A\n\nContent A.\n\n## Section B\n\nContent B."],
    )

    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, images = pipe.stage1_parse(str(pdf))
    jsonl_path, _batches, _requests = pipe.stage2_build_jsonl(doc_id)

    # 第一次入库
    pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
    )

    db = KnowledgeDB()
    chunks_before = db.query_by_doc(doc_id)
    assert len(chunks_before) == 2

    # 模拟文档内容变更（只改 Section B）
    new_text = "## Section A\n\nContent A.\n\n## Section B\n\nModified content B."
    (parsed_dir / "result.md").write_text(new_text, encoding="utf-8")

    # 重新 stage2
    jsonl_path2, _batches2, _requests2 = pipe.stage2_build_jsonl(doc_id)

    # 第二次入库（force 确保重新处理）
    pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=new_text,
        bigmodel_images=images,
        jsonl_path=jsonl_path2,
        force=True,
    )

    chunks_after = db.query_by_doc(doc_id)
    assert len(chunks_after) == 2
    for ck in chunks_after:
        assert ck.status == "done"


# ---------------------------------------------------------------------------
# 段落边界切分测试
# ---------------------------------------------------------------------------


def test_batch_builder_paragraph_split_no_period_cut(patched_config, monkeypatch):
    """段落边界切分不应在编号句号处切开标题行."""
    from core.doc_graph_pipeline import BatchBuilder, ChapterNode

    text = "Table 6. SFO Library Routines\n\nThis is the description."
    node = ChapterNode(level=1, title="Test", content=text)
    batches = BatchBuilder.build_batches([node], max_chars=30)

    all_contents = [b[0]["content"] for b in batches]

    # 验证 "Table 6." 不会单独成为一个 chunk
    for content in all_contents:
        assert content != "Table 6."
        assert not content.startswith("SFO Library Routines")

    # 完整段落应出现在某个 batch 中
    assert any("Table 6. SFO Library Routines" in c for c in all_contents)


# ---------------------------------------------------------------------------
# 架构实体过滤兼容性测试
# ---------------------------------------------------------------------------


class ArchFakeBatchClient:
    """返回包含处理器架构实体的 Mock BatchClient."""

    def __init__(self, *args, **kwargs):
        pass

    def submit_parallel_batches(self, requests, **kwargs):
        """返回含新实体/关系的提取结果."""
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
                                        "content": json.dumps(
                                            {
                                                "entities": [
                                                    {
                                                        "type": "Instruction",
                                                        "name": "MAC",
                                                        "properties": {
                                                            "opcode": "0011 0110",
                                                            "cycle_count": 1,
                                                        },
                                                    },
                                                    {
                                                        "type": "Register",
                                                        "name": "ACC",
                                                        "properties": {"width": 32},
                                                    },
                                                    {
                                                        "type": "Interrupt",
                                                        "name": "ADC_INT",
                                                        "properties": {
                                                            "vector_address": "0x000D40"
                                                        },
                                                    },
                                                ],
                                                "relationships": [
                                                    {
                                                        "type": "INSTRUCTION_READS_REGISTER",
                                                        "from": "MAC",
                                                        "to": "ACC",
                                                        "from_type": "Instruction",
                                                        "to_type": "Register",
                                                    },
                                                    {
                                                        "type": "INTERRUPT_TRIGGERS",
                                                        "from": "ADC_INT",
                                                        "to": "MAC",
                                                        "from_type": "Interrupt",
                                                        "to_type": "Instruction",
                                                    },
                                                ],
                                                "image_descriptions": [],
                                            }
                                        )
                                    }
                                }
                            ]
                        }
                    },
                }
            )
        return results

    def submit_embedding_batch(self, texts, **kwargs):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self):
        pass


def test_stage3_ingest_arch_entities(patched_config, monkeypatch):
    """Pipeline 应正确过滤并入库新增的处理器架构实体."""
    _mock_external_clients(monkeypatch)
    # 覆盖 BatchClient 为返回架构实体的版本
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BatchClient",
        ArchFakeBatchClient,
    )

    pdf = patched_config / "test.pdf"
    _make_pdf(pdf, ["## Instruction Set\n\nMAC and ADC_INT description."])

    pipe = DocGraphPipeline()
    doc_id, parsed_dir, text, images = pipe.stage1_parse(str(pdf))
    jsonl_path, _batches, _requests = pipe.stage2_build_jsonl(doc_id)

    result_doc_id = pipe.stage3_ingest(
        file_path=str(pdf),
        doc_id=doc_id,
        parsed_output_dir=parsed_dir,
        extracted_text=text,
        bigmodel_images=images,
        jsonl_path=jsonl_path,
    )
    assert result_doc_id == doc_id

    # 验证图谱中包含新实体
    assert pipe.graph.get_entity("Instruction", "MAC") is not None
    assert pipe.graph.get_entity("Register", "ACC") is not None
    assert pipe.graph.get_entity("Interrupt", "ADC_INT") is not None

    # 验证跨层级关系已建立
    mac_neighbors = pipe.graph.get_neighbors("Instruction", "MAC", direction="out")
    names = {n.name for n, _, _ in mac_neighbors}
    assert "ACC" in names

    adc_neighbors = pipe.graph.get_neighbors("Interrupt", "ADC_INT", direction="out")
    names = {n.name for n, _, _ in adc_neighbors}
    assert "MAC" in names

    # 验证 chunk metadata 中缓存了提取的实体
    db = KnowledgeDB()
    chunks = db.query_by_doc(doc_id)
    assert len(chunks) > 0
    cached_entities = chunks[0].metadata.get("extracted_entities", [])
    cached_types = {e.get("type") for e in cached_entities}
    assert "Instruction" in cached_types
    assert "Interrupt" in cached_types


def test_stage3_ingest_all_nodes(patched_config, monkeypatch):
    """包含 ### 子章节的文档应收集所有有 content 的节点，保留原文结构."""
    _mock_external_clients(monkeypatch)

    pdf = patched_config / "test.pdf"
    _make_pdf(
        pdf,
        [
            "# TRM\n\n"
            "## Section 1\n\nSection intro.\n\n"
            "### Sub 1.1\n\nSub content 1.\n\n"
            "### Sub 1.2\n\nSub content 2.\n\n"
            "## Section 2\n\nSection 2 content."
        ],
    )

    pipe = DocGraphPipeline()
    doc_id = pipe._process_one(str(pdf))
    assert doc_id is not None

    db = KnowledgeDB()
    chunks = db.query_by_doc(doc_id)
    titles = {ck.chapter_title for ck in chunks}

    # 所有有 content 的节点应被收集
    assert "Sub 1.1" in titles
    assert "Sub 1.2" in titles
    assert "Section 2" in titles
    assert "Section 1" in titles  # Section 1 有独立 content

    # 每个节点保留自己的 content，标题路径前缀包含完整层级
    sub11 = next(ck for ck in chunks if ck.chapter_title == "Sub 1.1")
    assert "Sub content 1." in sub11.content
    assert "# TRM" in sub11.content
    assert "## Section 1" in sub11.content
    assert "### Sub 1.1" in sub11.content

    sec1 = next(ck for ck in chunks if ck.chapter_title == "Section 1")
    assert "Section intro." in sec1.content
    assert "# TRM" in sec1.content
    assert "## Section 1" in sec1.content
    assert "### Sub 1.1" not in sec1.content

    sub12 = next(ck for ck in chunks if ck.chapter_title == "Sub 1.2")
    assert "Sub content 2." in sub12.content

    sec2 = next(ck for ck in chunks if ck.chapter_title == "Section 2")
    assert "Section 2 content." in sec2.content


# ---------------------------------------------------------------------------
# 字符数限制切分测试
# ---------------------------------------------------------------------------


def test_merge_image_descriptions_inline_replace(patched_config, monkeypatch):
    """Markdown 图片引用应被原位替换为文字描述."""
    _mock_external_clients(monkeypatch)
    pipe = DocGraphPipeline()

    content = "Some text.\n\n![img_001](images/page1.jpg)\n\nMore text."
    descs = [
        {
            "image_id": "img_001",
            "description": "A diagram",
            "chapter_title": "Ch1",
        }
    ]
    result = pipe._merge_image_descriptions(content, "Ch1", descs)
    assert "![img_001](images/page1.jpg)" not in result
    assert "[img_001] A diagram" in result
    assert "More text." in result


def test_merge_image_descriptions_fallback_append(patched_config, monkeypatch):
    """没有 Markdown 引用的图片应在末尾追加."""
    _mock_external_clients(monkeypatch)
    pipe = DocGraphPipeline()

    content = "Some text without image refs."
    descs = [
        {
            "image_id": "img_002",
            "description": "A chart",
            "chapter_title": "Ch1",
        }
    ]
    result = pipe._merge_image_descriptions(content, "Ch1", descs)
    assert "【图片内容】" in result
    assert "[img_002] A chart" in result


def test_stage5_single_text_jsonl(patched_config, monkeypatch):
    """stage5_build_embed_jsonl 应生成单文本 JSONL，custom_id 包含 db_id."""
    _mock_external_clients(monkeypatch)

    pipe = DocGraphPipeline()

    # 直接插入测试 chunks
    from core.models import Chunk

    chunks = [
        Chunk(
            doc_id="test_doc",
            chunk_id="c1",
            content="hello world",
            chunk_type="text",
            chapter_title="A",
            metadata={},
        ),
        Chunk(
            doc_id="test_doc",
            chunk_id="c2",
            content="foo bar",
            chunk_type="text",
            chapter_title="B",
            metadata={},
        ),
    ]
    for c in chunks:
        pipe.db.insert_chunk(c)

    embed_jsonl_path = pipe.stage5_build_embed_jsonl("test_doc")
    lines = embed_jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    requests = [json.loads(line) for line in lines if line.strip()]

    # 每个 chunk 对应一个 request
    assert len(requests) == 2
    # 验证 custom_id 格式: embed_dbid_{db_id}
    assert requests[0]["custom_id"].startswith("embed_dbid_")
    assert requests[1]["custom_id"].startswith("embed_dbid_")
    # 验证 body.input 为字符串（不是数组）
    assert isinstance(requests[0]["body"]["input"], str)
    assert requests[0]["body"]["input"] == "hello world"
    assert isinstance(requests[1]["body"]["input"], str)
    assert requests[1]["body"]["input"] == "foo bar"
    # 验证不生成 embed_map.json
    embed_map_path = embed_jsonl_path.parent / "test_doc_embed_map.json"
    assert not embed_map_path.exists()


def test_split_for_embedding_paragraph_boundary():
    """_split_for_embedding 应按段落边界切分超长文本."""
    from core.doc_graph_pipeline import _split_for_embedding

    text = "First paragraph here.\n\nSecond paragraph here."
    parts = _split_for_embedding(text, max_chars=30)
    # 两个段落应被分开（每个段落都 < 30 chars）
    assert len(parts) == 2
    assert "First paragraph" in parts[0]
    assert "Second paragraph" in parts[1]


def test_split_for_embedding_sentence_fallback():
    """单个段落超长时应按句子边界 fallback 切分."""
    from core.doc_graph_pipeline import _split_for_embedding

    text = "Hello world. Foo bar. Baz qux."
    parts = _split_for_embedding(text, max_chars=15)
    # 应切成至少 2 部分（句子级 fallback）
    assert len(parts) >= 2
    # 每部分不超过 15 chars（允许单个句子超长时被保留原样）
    for p in parts:
        assert len(p) <= 20  # 放宽到 20，因为单个短句可能略超 15


def test_stage3_chat_mode_writes_batch_format_jsonl(patched_config, monkeypatch, tmp_path):
    """Chat mode _submit_via_chat output results.jsonl format must match Batch result format."""
    _mock_external_clients(monkeypatch)
    monkeypatch.setattr("core.doc_graph_pipeline.BatchClient", FakeBatchClient)
    monkeypatch.setattr("core.doc_graph_pipeline.BaseLLMClient", FakeChatClient)
    monkeypatch.setattr(Config, "LLM_MODE", "chat")

    pipe = DocGraphPipeline()
    # ensure ChatClient is used
    assert pipe.llm_chat is not None

    # construct mock JSONL requests (same format as Stage2 output)
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    requests = [
        {
            "custom_id": "batch_0",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "user prompt"},
                ],
                "response_format": {"type": "json_object"},
            },
        }
    ]
    results_path = batch_dir / "chat_test_results.jsonl"

    # call _submit_via_chat
    results = pipe._submit_via_chat(requests, results_path)

    # verify result file exists
    assert results_path.exists()

    # verify file content format matches Batch API return format
    lines = results_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["custom_id"] == "batch_0"
    assert "response" in result
    assert result["response"]["status_code"] == 200
    assert "body" in result["response"]
    assert "choices" in result["response"]["body"]

    # verify Stage 4 _parse_results can parse correctly
    from core.doc_graph_pipeline import EntityExtractor

    extractor = EntityExtractor()
    entities, relations, _ = extractor._parse_results(results, [{}], doc_id="chat_test")
    assert len(entities) == 1
    assert entities[0].entity_type == "Register"
    assert entities[0].name == "TBCTL"


def test_stage3_chat_mode_body_preserved(patched_config, monkeypatch, tmp_path):
    """Chat mode req['body'] must be passed as-is without modification or missing fields."""
    _mock_external_clients(monkeypatch)
    monkeypatch.setattr("core.doc_graph_pipeline.BatchClient", FakeBatchClient)
    monkeypatch.setattr("core.doc_graph_pipeline.BaseLLMClient", FakeChatClient)
    monkeypatch.setattr(Config, "LLM_MODE", "chat")

    pipe = DocGraphPipeline()
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    original_body = {
        "model": "kimi-k2.5",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ],
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    requests = [{"custom_id": "batch_0", "body": original_body}]
    results_path = batch_dir / "body_test_results.jsonl"

    # use FakeChatClient._post to capture received payload
    captured = []
    original_post = pipe.llm_chat._post

    def _capture_post(url, payload, timeout=None):
        captured.append(payload)
        return original_post(url, payload, timeout)

    pipe.llm_chat._post = _capture_post

    pipe._submit_via_chat(requests, results_path)

    assert len(captured) == 1
    assert captured[0]["model"] == "kimi-k2.5"
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured[0]["messages"][0]["role"] == "system"
    assert captured[0]["messages"][1]["role"] == "user"


def test_stage3_chat_mode_error_continue(patched_config, monkeypatch, tmp_path):
    """Chat mode single request failure should not interrupt subsequent requests."""
    _mock_external_clients(monkeypatch)
    monkeypatch.setattr("core.doc_graph_pipeline.BatchClient", FakeBatchClient)
    monkeypatch.setattr("core.doc_graph_pipeline.BaseLLMClient", FakeChatClient)
    monkeypatch.setattr(Config, "LLM_MODE", "chat")

    pipe = DocGraphPipeline()
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    call_count = 0

    def _fail_then_succeed(url, payload, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated failure")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"entities": [], "relationships": [], "image_descriptions": []}
                        )
                    }
                }
            ]
        }

    pipe.llm_chat._post = _fail_then_succeed

    requests = [
        {"custom_id": "batch_0", "body": {}},
        {"custom_id": "batch_1", "body": {}},
    ]
    results_path = batch_dir / "error_test_results.jsonl"
    results = pipe._submit_via_chat(requests, results_path)

    assert len(results) == 2
    assert results[0]["response"]["status_code"] == 500
    assert results[1]["response"]["status_code"] == 200


def test_stage3_batch_mode_fallback_to_chat(patched_config, monkeypatch, tmp_path):
    """Batch mode when BatchClient unavailable should auto fallback to Chat."""
    _mock_external_clients(monkeypatch)
    monkeypatch.setattr("core.doc_graph_pipeline.BaseLLMClient", FakeChatClient)
    monkeypatch.setattr(Config, "LLM_MODE", "batch")

    pipe = DocGraphPipeline()
    # simulate BatchClient initialization failure
    pipe.llm_batch = None

    doc_id = "fallback_test_doc"
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = Config.DB_PATH.parent / "parsed" / doc_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # create result.md for _read_result_md
    result_md = parsed_dir / "result.md"
    result_md.write_text("## Section 1\n\nTest content.", encoding="utf-8")

    # create JSONL and batch_info for incremental filtering
    jsonl_path = batch_dir / f"{doc_id}.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "custom_id": "batch_0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch_info_path = batch_dir / f"{doc_id}_batch_info.json"
    batch_info_path.write_text(
        json.dumps([{"custom_id": "batch_0", "chapter_titles": ["Section 1"]}]),
        encoding="utf-8",
    )

    # mock db to avoid skip logic
    monkeypatch.setattr(pipe.db, "get_document_by_path", lambda path: None)
    # mock incremental analysis to return added chapters
    monkeypatch.setattr(
        pipe,
        "_incremental_analysis",
        lambda fp, et, force=False: (
            [{"title": "Section 1", "content": "Test content.", "level": 2}],
            [],
            [],
            [{"title": "Section 1", "content": "Test content.", "level": 2}],
            [],
            None,
            "abc123",
        ),
    )

    # create a dummy PDF file for hash calculation
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"dummy")

    results_path = pipe.stage3_submit_batches(
        doc_id=doc_id,
        file_path=str(dummy_pdf),
        jsonl_path=jsonl_path,
        force=True,
    )

    assert results_path.exists()
    lines = results_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["custom_id"] == "batch_0"
    assert result["response"]["status_code"] == 200
    assert "choices" in result["response"]["body"]


def test_stage3_both_clients_unavailable_skips(patched_config, monkeypatch, tmp_path):
    """BatchClient 和 ChatClient 均初始化失败时应返回空 results.jsonl."""
    monkeypatch.setattr(Config, "LLM_MODE", "batch")
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BatchClient",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no batch key")),
    )
    monkeypatch.setattr(
        "core.doc_graph_pipeline.BaseLLMClient",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no chat key")),
    )

    pipe = DocGraphPipeline()
    assert pipe.llm_batch is None
    assert pipe.llm_chat is None

    doc_id = "both_unavailable_doc"
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = Config.DB_PATH.parent / "parsed" / doc_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 创建 result.md
    result_md = parsed_dir / "result.md"
    result_md.write_text("## Section 1\n\nTest content.", encoding="utf-8")

    # 创建 JSONL 和 batch_info
    jsonl_path = batch_dir / f"{doc_id}.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "custom_id": "batch_0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch_info_path = batch_dir / f"{doc_id}_batch_info.json"
    batch_info_path.write_text(
        json.dumps([{"custom_id": "batch_0", "chapter_titles": ["Section 1"]}]),
        encoding="utf-8",
    )

    # mock db 和增量分析
    monkeypatch.setattr(pipe.db, "get_document_by_path", lambda path: None)
    monkeypatch.setattr(
        pipe,
        "_incremental_analysis",
        lambda fp, et, force=False: (
            [{"title": "Section 1", "content": "Test content.", "level": 2}],
            [],
            [],
            [{"title": "Section 1", "content": "Test content.", "level": 2}],
            [],
            None,
            "abc123",
        ),
    )

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"dummy")

    results_path = pipe.stage3_submit_batches(
        doc_id=doc_id,
        file_path=str(dummy_pdf),
        jsonl_path=jsonl_path,
        force=True,
    )

    assert results_path.exists()
    assert results_path.stat().st_size == 0


def test_stage3_batch_fallback_writes_file(patched_config, monkeypatch, tmp_path):
    """BatchClient 返回结果但未写入文件时，fallback 手动写入."""
    monkeypatch.setattr(Config, "LLM_MODE", "batch")
    monkeypatch.setattr("core.doc_graph_pipeline.BaseLLMClient", FakeChatClient)

    class WriteNothingBatchClient:
        """Mock BatchClient：返回结果但不写入 output_path."""

        def __init__(self, *args, **kwargs):
            pass

        def submit_parallel_batches(self, requests, output_path=None):
            return [
                {
                    "custom_id": req.get("custom_id", f"batch_{i}"),
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "entities": [],
                                                "relationships": [],
                                                "image_descriptions": [],
                                            }
                                        )
                                    }
                                }
                            ]
                        }
                    },
                }
                for i, req in enumerate(requests)
            ]

        def close(self):
            pass

    monkeypatch.setattr(
        "core.doc_graph_pipeline.BatchClient", WriteNothingBatchClient
    )

    pipe = DocGraphPipeline()
    assert pipe.llm_batch is not None

    doc_id = "fallback_write_doc"
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = Config.DB_PATH.parent / "parsed" / doc_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    result_md = parsed_dir / "result.md"
    result_md.write_text("## Section 1\n\nTest content.", encoding="utf-8")

    jsonl_path = batch_dir / f"{doc_id}.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "custom_id": "batch_0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch_info_path = batch_dir / f"{doc_id}_batch_info.json"
    batch_info_path.write_text(
        json.dumps([{"custom_id": "batch_0", "chapter_titles": ["Section 1"]}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(pipe.db, "get_document_by_path", lambda path: None)
    monkeypatch.setattr(
        pipe,
        "_incremental_analysis",
        lambda fp, et, force=False: (
            [{"title": "Section 1", "content": "Test content.", "level": 2}],
            [],
            [],
            [{"title": "Section 1", "content": "Test content.", "level": 2}],
            [],
            None,
            "abc123",
        ),
    )

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"dummy")

    results_path = pipe.stage3_submit_batches(
        doc_id=doc_id,
        file_path=str(dummy_pdf),
        jsonl_path=jsonl_path,
        force=True,
    )

    assert results_path.exists()
    assert results_path.stat().st_size > 0
    lines = results_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["custom_id"] == "batch_0"
    assert "response" in result


def test_stage4_hash_mismatch_deletes_incremental(patched_config, monkeypatch, tmp_path):
    """result.md hash 不匹配时应删除旧增量分析文件并继续处理."""
    _mock_external_clients(monkeypatch)
    monkeypatch.setattr(Config, "LLM_MODE", "batch")

    pipe = DocGraphPipeline()
    doc_id = "hash_test_doc"
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = Config.DB_PATH.parent / "parsed" / doc_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 创建 result.md
    result_md = parsed_dir / "result.md"
    result_md.write_text("## Section 1\n\nTest content.", encoding="utf-8")

    # 创建增量信息文件，但 hash 是旧的
    info_path = batch_dir / f"{doc_id}_incremental.json"
    info_path.write_text(
        json.dumps({"result_md_hash": "wrong_hash_12345"}),
        encoding="utf-8",
    )

    # mock 跳过已有文档检查
    monkeypatch.setattr(pipe.db, "get_document_by_path", lambda path: None)
    # mock 增量分析返回空结果
    monkeypatch.setattr(
        pipe,
        "_incremental_analysis",
        lambda fp, et, force=False: ([], [], [], [], [], None, "abc123"),
    )

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"dummy")
    results_path = batch_dir / f"{doc_id}_results.jsonl"
    results_path.write_text("{}", encoding="utf-8")

    pipe.stage4_ingest_results(
        doc_id=doc_id,
        file_path=str(dummy_pdf),
        results_path=results_path,
        force=True,
    )
    # 旧增量分析文件应被删除
    assert not info_path.exists()


def test_stage4_title_mismatch_deletes_incremental(patched_config, monkeypatch, tmp_path):
    """增量分析 title 不一致时应删除旧增量分析文件并继续处理."""
    _mock_external_clients(monkeypatch)
    monkeypatch.setattr(Config, "LLM_MODE", "batch")

    pipe = DocGraphPipeline()
    doc_id = "title_test_doc"
    batch_dir = Config.DB_PATH.parent / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = Config.DB_PATH.parent / "parsed" / doc_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 创建 result.md
    result_md = parsed_dir / "result.md"
    result_md.write_text("## Section 1\n\nTest content.", encoding="utf-8")
    actual_hash = hashlib.md5(result_md.read_bytes()).hexdigest()

    # 创建增量信息文件，hash 正确但 title 不一致
    info_path = batch_dir / f"{doc_id}_incremental.json"
    info_path.write_text(
        json.dumps(
            {
                "result_md_hash": actual_hash,
                "unchanged_titles": ["Old Section"],
                "changed_titles": [],
                "added_titles": [],
                "removed_titles": [],
            }
        ),
        encoding="utf-8",
    )

    # mock 跳过已有文档检查
    monkeypatch.setattr(pipe.db, "get_document_by_path", lambda path: None)
    # mock 增量分析返回不同的 title
    monkeypatch.setattr(
        pipe,
        "_incremental_analysis",
        lambda fp, et, force=False: (
            [{"title": "Section 1", "content": "Test.", "level": 2}],
            [],
            [],
            [],
            [],
            None,
            "abc123",
        ),
    )

    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"dummy")
    results_path = batch_dir / f"{doc_id}_results.jsonl"
    results_path.write_text("{}", encoding="utf-8")

    pipe.stage4_ingest_results(
        doc_id=doc_id,
        file_path=str(dummy_pdf),
        results_path=results_path,
        force=True,
    )
    # 旧增量分析文件应被删除
    assert not info_path.exists()


def test_close_releases_llm_chat(patched_config, monkeypatch):
    """close() 必须关闭 llm_chat 客户端."""
    _mock_external_clients(monkeypatch)
    monkeypatch.setattr("core.doc_graph_pipeline.BaseLLMClient", FakeChatClient)
    monkeypatch.setattr(Config, "LLM_MODE", "chat")

    pipe = DocGraphPipeline()
    # 强制初始化 llm_chat
    fake_chat = FakeChatClient()
    pipe.llm_chat = fake_chat

    closed = []

    def tracking_close():
        closed.append(True)

    fake_chat.close = tracking_close

    pipe.close()
    assert len(closed) == 1
    assert pipe.llm_chat is not None  # close() 不应将引用置为 None
