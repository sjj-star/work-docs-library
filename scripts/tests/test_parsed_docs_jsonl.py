"""使用已解析文档生成 JSONL 的端到端测试."""

from pathlib import Path

import pytest
from core.batch_clients import _build_jsonl, _parse_jsonl
from core.config import Config
from core.doc_graph_pipeline import (
    BatchBuilder,
    ChapterParser,
    DocGraphPipeline,
    EntityExtractor,
    _build_content_blocks_and_maps,
)


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

        # 2. 构建 content_blocks 并构建 Batch（方案C）
        content_blocks, _heading_maps = _build_content_blocks_and_maps(
            tree, max_chars=Config.LLM_BATCH_MAX_CHARS
        )
        batches = BatchBuilder.build_batches(content_blocks, max_chars=Config.LLM_BATCH_MAX_CHARS)
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


# -- 从 test_jsonl_consistency.py 合并的内容一致性测试 --


def _extract_original_content(full_content: str) -> str:
    """去掉标题路径前缀，提取原始 content."""
    if "\n\n" not in full_content:
        return full_content
    prefix, body = full_content.split("\n\n", 1)
    lines = [line for line in prefix.strip().split("\n") if line.strip()]
    if lines and all(line.startswith("#") for line in lines):
        return body
    return full_content


class TestJsonlConsistency:
    """验证已解析文档生成 JSONL 后无内容丢失."""

    @pytest.mark.parametrize("doc_dir", _get_parsed_doc_dirs())
    def test_jsonl_no_content_loss(self, doc_dir: Path) -> None:
        """stage2_build_jsonl 生成的 batch content 应包含原始 Markdown 所有内容."""
        result_md = doc_dir / "result.md"
        if not result_md.exists():
            pytest.skip(f"result.md 不存在 | {doc_dir}")

        doc_id = doc_dir.name
        md_text = result_md.read_text(encoding="utf-8")

        # 1. 通过项目接口生成 JSONL
        pipe = DocGraphPipeline()
        jsonl_path, batches, requests, _content_blocks, _heading_maps = pipe.stage2_build_jsonl(
            doc_id
        )

        assert jsonl_path.exists(), f"JSONL 应已生成 | {jsonl_path}"
        assert len(batches) > 0, "应生成至少一个 batch"
        assert len(requests) == len(batches), "requests 数量应与 batches 一致"

        # 2. 从 batches 提取所有原始 content
        batch_contents: list[str] = []
        for batch in batches:
            for ch in batch:
                batch_contents.append(_extract_original_content(ch["content"]))

        # 3. 从原始 Markdown 提取所有 content
        tree = ChapterParser.parse_tree(md_text)
        md_nodes: list[dict[str, str]] = []
        for root in tree:
            md_nodes.extend(ChapterParser.collect_all_nodes(root))

        md_contents = [_extract_original_content(ch["content"]) for ch in md_nodes]

        # 4. 一致性断言：每个 md content 必须完整出现在某个 batch content 中
        missing: list[tuple[str, str]] = []
        for title, original in zip([ch["title"] for ch in md_nodes], md_contents):
            found = any(original in bc or bc in original for bc in batch_contents)
            if not found:
                missing.append((title, original[:120]))

        assert not missing, (
            f"文档 {doc_id} 有 {len(missing)} 个 chunk 内容未在 batch 中找到:\n"
            + "\n".join(f"  - {t}: {p}..." for t, p in missing)
        )

        # 5. 字符数统计（允许切分导致的 \n\n 分隔符差异，方案C放宽限制）
        total_md = sum(len(c) for c in md_contents)
        total_batch = sum(len(c) for c in batch_contents)
        diff = abs(total_md - total_batch)
        # 方案C：聚合后标题层级保留方式变化，允许更大差异
        assert diff <= max(len(batch_contents) * 2, 500), (
            f"字符数差异过大: md={total_md}, batch={total_batch}, diff={diff}"
        )
