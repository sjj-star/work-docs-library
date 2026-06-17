"""content_blocks 和 heading_maps 构建测试（方案C）."""

from core.doc_graph_pipeline import (
    ChapterNode,
    _build_content_blocks_and_maps,
    _collect_section_content,
)


def test_collect_section_content_with_children():
    """_collect_section_content 应递归收集节点及其所有子孙."""
    root = ChapterNode(level=2, title="Section 1", content="Section intro.")
    sub1 = ChapterNode(level=3, title="Sub 1.1", content="Sub content 1.")
    sub2 = ChapterNode(level=3, title="Sub 1.2", content="Sub content 2.")
    root.children = [sub1, sub2]

    result = _collect_section_content(root)
    assert "## Section 1" in result
    assert "Section intro." in result
    assert "### Sub 1.1" in result
    assert "Sub content 1." in result
    assert "### Sub 1.2" in result
    assert "Sub content 2." in result


def test_collect_section_content_empty_children():
    """无子节点时应只返回自身内容."""
    node = ChapterNode(level=2, title="Section", content="Content.")
    result = _collect_section_content(node)
    assert result == "## Section\n\nContent."


def test_build_content_blocks_basic():
    """基本文档应生成正确的 content_blocks 和 heading_maps."""
    root = ChapterNode(level=1, title="Doc", content="")
    sec1 = ChapterNode(level=2, title="Section 1", content="Content 1.")
    sec2 = ChapterNode(level=2, title="Section 2", content="Content 2.")
    root.children = [sec1, sec2]

    content_blocks, heading_maps = _build_content_blocks_and_maps([root])

    # 每个 section 一个 block
    assert len(content_blocks) == 2
    assert content_blocks[0]["section_title"] == "Section 1"
    assert content_blocks[1]["section_title"] == "Section 2"
    assert "Content 1." in content_blocks[0]["content"]
    assert "Content 2." in content_blocks[1]["content"]

    # heading_maps 包含所有 ## 标题
    titles = {hm["heading_title"] for hm in heading_maps}
    assert "Section 1" in titles
    assert "Section 2" in titles


def test_build_content_blocks_with_preface():
    """Root 有 content 时，第一个 section 应包含 preface."""
    root = ChapterNode(level=1, title="Doc", content="Introduction.")
    sec1 = ChapterNode(level=2, title="Section 1", content="Content 1.")
    root.children = [sec1]

    content_blocks, _ = _build_content_blocks_and_maps([root])

    assert len(content_blocks) == 1
    block = content_blocks[0]
    assert "# Doc" in block["content"]
    assert "Introduction." in block["content"]
    assert "## Section 1" in block["content"]
    assert "Content 1." in block["content"]


def test_build_content_blocks_with_subsections():
    """### 子标题应共享 section 的 blocks（简化处理）."""
    root = ChapterNode(level=1, title="Doc", content="")
    sec1 = ChapterNode(level=2, title="Section 1", content="Section intro.")
    sub1 = ChapterNode(level=3, title="Sub 1.1", content="Sub content 1.")
    sub2 = ChapterNode(level=3, title="Sub 1.2", content="Sub content 2.")
    sec1.children = [sub1, sub2]
    root.children = [sec1]

    content_blocks, heading_maps = _build_content_blocks_and_maps([root])

    # 只有一个 section，一个 block
    assert len(content_blocks) == 1
    assert "### Sub 1.1" in content_blocks[0]["content"]
    assert "### Sub 1.2" in content_blocks[0]["content"]

    # heading_maps 包含 Section 1, Sub 1.1, Sub 1.2
    hm_titles = {hm["heading_title"] for hm in heading_maps}
    assert hm_titles == {"Section 1", "Sub 1.1", "Sub 1.2"}

    # Sub 1.1 的 heading_map 指向 section 的所有 blocks
    sub11_hm = next(hm for hm in heading_maps if hm["heading_title"] == "Sub 1.1")
    assert sub11_hm["block_ids"] == [content_blocks[0]["block_id"]]


def test_build_content_blocks_split_by_max_chars():
    """超长的 section content 应按 BLOCK_MAX_CHARS 切分为多个 blocks."""
    from core.config import Config

    original_max = Config.BLOCK_MAX_CHARS
    Config.BLOCK_MAX_CHARS = 400
    try:
        root = ChapterNode(level=1, title="Doc", content="")
        long_content = "A" * 300 + "\n\n" + "B" * 300
        sec1 = ChapterNode(level=2, title="Section 1", content=long_content)
        root.children = [sec1]

        content_blocks, heading_maps = _build_content_blocks_and_maps([root])

        # 应切分为至少 2 个 blocks
        assert len(content_blocks) >= 2
        assert all(block["section_title"] == "Section 1" for block in content_blocks)

        # heading_maps 中 Section 1 应指向所有 blocks
        sec1_hm = next(hm for hm in heading_maps if hm["heading_title"] == "Section 1")
        assert len(sec1_hm["block_ids"]) == len(content_blocks)
    finally:
        Config.BLOCK_MAX_CHARS = original_max


def test_build_content_blocks_seq_index():
    """seq_index 应全局递增."""
    root = ChapterNode(level=1, title="Doc", content="")
    sec1 = ChapterNode(level=2, title="Section 1", content="A.")
    sec2 = ChapterNode(level=2, title="Section 2", content="B.")
    root.children = [sec1, sec2]

    content_blocks, _ = _build_content_blocks_and_maps([root])

    seq_indices = [block["seq_index"] for block in content_blocks]
    assert seq_indices == list(range(len(content_blocks)))


def test_build_content_blocks_block_id_format():
    """block_id 应为 b_{seq_index} 格式."""
    root = ChapterNode(level=1, title="Doc", content="")
    sec1 = ChapterNode(level=2, title="Section 1", content="A.")
    root.children = [sec1]

    content_blocks, _ = _build_content_blocks_and_maps([root])

    assert len(content_blocks) == 1
    assert content_blocks[0]["block_id"] == "b_0"


def test_build_content_blocks_parent_heading_none():
    """parent_heading 应使用 None 而非空字符串表示无父标题."""
    root = ChapterNode(level=1, title="", content="")
    sec1 = ChapterNode(level=2, title="Section 1", content="A.")
    root.children = [sec1]

    _, heading_maps = _build_content_blocks_and_maps([root])

    sec1_hm = next(hm for hm in heading_maps if hm["heading_title"] == "Section 1")
    assert sec1_hm["parent_heading"] is None


def test_build_content_blocks_content_hash_consistency():
    """原始 chapter content hash 与聚合 section content hash 应不同.

    _save_blocks_to_db 应使用原始 hash 以保持增量更新一致性.
    """
    root = ChapterNode(level=1, title="Doc", content="Preface.")
    sec1 = ChapterNode(level=2, title="Section 1", content="Original content.")
    sub1 = ChapterNode(level=3, title="Sub 1.1", content="Sub content.")
    sec1.children = [sub1]
    root.children = [sec1]

    content_blocks, _ = _build_content_blocks_and_maps([root])

    # content_blocks 中 Section 1 包含聚合内容（Original + Sub）
    sec_block = next(b for b in content_blocks if b["section_title"] == "Section 1")
    aggregated_content = sec_block["content"]
    assert "Sub content." in aggregated_content

    # 原始 chapter content 不含子章节
    original_content = "Original content."
    assert "Sub content." not in original_content

    # 验证 hash 不同（说明为什么 _save_blocks_to_db 需要使用原始 chapter hash）
    import hashlib

    original_hash = hashlib.md5(original_content.encode()).hexdigest()[:16]
    aggregated_hash = hashlib.md5(aggregated_content.encode()).hexdigest()[:16]
    assert original_hash != aggregated_hash, "原始 hash 和聚合 hash 应不同，否则增量更新失效"
