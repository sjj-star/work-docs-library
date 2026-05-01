"""JSONL 生成一致性测试.

使用 knowledge_base/parsed/ 下的已解析文档，通过 DocGraphPipeline.stage2_build_jsonl()
生成 JSONL，验证 batch content 与原始 Markdown 内容的一致性（无内容丢失）。
"""

from pathlib import Path

import pytest
from core.config import Config
from core.doc_graph_pipeline import ChapterParser, DocGraphPipeline


def _extract_original_content(full_content: str) -> str:
    """去掉标题路径前缀，提取原始 content."""
    if "\n\n" not in full_content:
        return full_content
    prefix, body = full_content.split("\n\n", 1)
    lines = [line for line in prefix.strip().split("\n") if line.strip()]
    if lines and all(line.startswith("#") for line in lines):
        return body
    return full_content


def _get_parsed_doc_dirs() -> list[Path]:
    """获取 knowledge_base/parsed/ 下的所有文档目录."""
    parsed_dir = Path(Config.DB_PATH).parent / "parsed"
    if not parsed_dir.exists():
        return []
    return [d for d in parsed_dir.iterdir() if d.is_dir()]


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
        jsonl_path, batches, requests = pipe.stage2_build_jsonl(doc_id)

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

        # 5. 字符数统计（允许切分导致的 \n\n 分隔符差异）
        total_md = sum(len(c) for c in md_contents)
        total_batch = sum(len(c) for c in batch_contents)
        diff = abs(total_md - total_batch)
        assert diff <= len(batch_contents) * 2, (
            f"字符数差异过大: md={total_md}, batch={total_batch}, diff={diff}"
        )
