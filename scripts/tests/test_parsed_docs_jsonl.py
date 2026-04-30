"""使用已解析文档生成 JSONL 的端到端测试."""

from pathlib import Path

import pytest
from core.batch_clients import _build_jsonl, _parse_jsonl
from core.config import Config
from core.doc_graph_pipeline import BatchBuilder, ChapterNode, ChapterParser, EntityExtractor


def _get_parsed_doc_dirs() -> list[Path]:
    """获取 knowledge_base/parsed/ 下的所有文档目录."""
    parsed_dir = Path(Config.DB_PATH).parent / "parsed"
    if not parsed_dir.exists():
        return []
    return [d for d in parsed_dir.iterdir() if d.is_dir()]


def _build_doc_context(root_nodes: list) -> str:
    """从章节树构建 doc_context."""
    # 取第一个非空 root 的标题作为文档标题
    doc_title = ""
    for root in root_nodes:
        if root.title:
            doc_title = root.title
            break
    return f"以下章节来自 文档《{doc_title}》。\n\n---\n"


class TestParsedDocsToJsonl:
    """测试已解析文档 → 章节树 → Batch → JSONL 的完整流程."""

    @pytest.mark.parametrize("doc_dir", _get_parsed_doc_dirs())
    def test_parsed_doc_generates_valid_jsonl(self, doc_dir: Path, tmp_path: Path):
        """每份已解析文档应能生成有效的 JSONL 文件，且 batch 内容非空."""
        result_md = doc_dir / "result.md"
        image_dir = doc_dir / "images"

        assert result_md.exists(), f"result.md 不存在 | path={result_md}"

        text = result_md.read_text(encoding="utf-8")
        assert len(text) > 0, "result.md 不应为空"

        # 1. 章节树解析
        tree = ChapterParser.parse_tree(text)
        assert len(tree) > 0, "parse_tree 应返回至少一个 root"

        # 2. 扁平化树并构建 Batch
        flat_nodes = [
            ChapterNode(level=1, title=ch["title"], content=ch["content"])
            for root in tree
            for ch in ChapterParser.collect_all_nodes(root)
        ]
        batches = BatchBuilder.build_batches(flat_nodes, max_chars=Config.LLM_BATCH_MAX_CHARS)
        assert len(batches) > 0, "应生成至少一个 batch"

        # 3. 验证每个 batch 内容非空（排除误识别的目录条目/代码噪声）
        for i, batch in enumerate(batches):
            for chunk in batch:
                assert len(chunk["content"]) >= 10, (
                    f"Batch {i} 的 chunk '{chunk['title'][:40]}' "
                    f"content 长度 {len(chunk['content'])} < 10，可能被误识别为空内容"
                )

        # 4. EntityExtractor 构建 requests
        extractor = EntityExtractor(batch_client=None)
        doc_context = _build_doc_context(tree)
        requests = extractor._build_batch_requests(
            batches, image_base_dir=image_dir, doc_context=doc_context
        )
        assert len(requests) == len(batches), "requests 数量应与 batches 一致"

        # 5. 验证每个 request 的结构
        for req in requests:
            assert req["method"] == "POST"
            assert "custom_id" in req
            body = req["body"]
            assert body["model"] == Config.LLM_MODEL
            messages = body["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            # user content 中应包含章节文本
            user_content = messages[1]["content"]
            text_parts = [c["text"] for c in user_content if c["type"] == "text"]
            full_text = "".join(text_parts)
            assert len(full_text) >= 100, "user content 文本长度应 >= 100"

        # 6. 生成 JSONL 文件
        jsonl_path = tmp_path / f"{doc_dir.name}.jsonl"
        _build_jsonl(requests, jsonl_path)

        assert jsonl_path.exists(), f"JSONL 文件应已生成 | path={jsonl_path}"
        assert jsonl_path.stat().st_size > 0, "JSONL 文件不应为空"

        # 7. 验证 JSONL 可正确解析回 requests
        parsed = _parse_jsonl(jsonl_path.read_text(encoding="utf-8"))
        assert len(parsed) == len(requests), "解析后的 JSONL 行数应与 requests 一致"

        # 打印信息供人工查看
        print(
            f"\n[DOC {doc_dir.name}] "
            f"roots={len(tree)}, batches={len(batches)}, "
            f"jsonl={jsonl_path}, size={jsonl_path.stat().st_size} bytes"
        )
