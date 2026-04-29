"""BatchBuilder 单元测试."""

from core.doc_graph_pipeline import BatchBuilder, ChapterNode


def test_split_by_sentences_protects_html_table():
    r"""HTML 表格不应被 \n\n+ 切分截断."""
    text = (
        "Paragraph 1.\n\n"
        "<table>\n"
        "  <tr><td>A</td></tr>\n\n"
        "  <tr><td>B</td></tr>\n"
        "</table>\n\n"
        "Paragraph 2."
    )
    chunks = BatchBuilder._split_by_sentences(text, max_len=500)
    # 总长度 < max_len，应合并为一个 chunk，但表格内容必须完整保留
    full_text = "\n\n".join(chunks)
    assert "<table>" in full_text
    assert "<tr><td>A</td></tr>" in full_text
    assert "<tr><td>B</td></tr>" in full_text
    assert "</table>" in full_text


def test_split_by_sentences_protects_html_table_across_chunks():
    """当文本超长时，HTML 表格应完整保留在单个 chunk 中."""
    table_html = "<table>\n  <tr><td>X</td></tr>\n\n  <tr><td>Y</td></tr>\n</table>"
    text = "A. " * 100 + "\n\n" + table_html + "\n\n" + "B. " * 100
    chunks = BatchBuilder._split_by_sentences(text, max_len=200)
    # 找到包含表格的 chunk
    table_chunks = [c for c in chunks if "<table>" in c]
    assert len(table_chunks) == 1
    assert "<tr><td>X</td></tr>" in table_chunks[0]
    assert "<tr><td>Y</td></tr>" in table_chunks[0]
    # 表格不应被拆分到多个 chunk
    for c in chunks:
        open_count = c.count("<table>")
        close_count = c.count("</table>")
        assert open_count == close_count, "HTML 表格被截断到多个 chunk 中"


def test_split_by_sentences_protects_markdown_table():
    r"""Markdown 表格不应被 \n\n+ 切分截断."""
    text = "Para 1.\n\n| Reg | Offset |\n|-----|--------|\n| A   | 0x00   |\n\nPara 2."
    chunks = BatchBuilder._split_by_sentences(text, max_len=500)
    table_chunk = [c for c in chunks if "| Reg | Offset |" in c]
    assert len(table_chunk) == 1
    assert "| A   | 0x00   |" in table_chunk[0]


def test_split_by_sentences_protects_code_block():
    r"""代码块不应被 \n\n+ 切分截断."""
    text = "Intro.\n\n```c\nvoid init() {\n\n    setup();\n}\n```\n\nOutro."
    chunks = BatchBuilder._split_by_sentences(text, max_len=500)
    code_chunk = [c for c in chunks if "```c" in c]
    assert len(code_chunk) == 1
    assert "setup();" in code_chunk[0]
    assert "```" in code_chunk[0]


def test_split_by_sentences_no_hard_truncate_fallback():
    """Fallback 不应硬截断文本."""
    text = ""
    chunks = BatchBuilder._split_by_sentences(text, max_len=100)
    assert chunks == [""]


def test_split_by_sentences_respects_max_len():
    """正常段落应按 max_len 切分."""
    text = "A. " * 200  # 约 800 chars
    chunks = BatchBuilder._split_by_sentences(text, max_len=200)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 200 + 50  # 允许少量超限（单个 sentence 超过 max_len 时）


def test_split_by_sentences_html_table_case_insensitive():
    """HTML 表格标签大小写不敏感."""
    text = "Intro.\n\n<TABLE>\n  <TR><TD>X</TD></TR>\n\n  <TR><TD>Y</TD></TR>\n</TABLE>\n\nOutro."
    chunks = BatchBuilder._split_by_sentences(text, max_len=500)
    table_chunk = [c for c in chunks if "<TABLE>" in c]
    assert len(table_chunk) == 1
    assert "<TR><TD>X</TD></TR>" in table_chunk[0]
    assert "<TR><TD>Y</TD></TR>" in table_chunk[0]


class TestBatchBuilderFiltersEmptyContent:
    """测试 BatchBuilder 过滤空 content 节点."""

    def test_filters_root_with_empty_content(self):
        """无子节点且 content 为空的 root 应被过滤."""
        empty_root = ChapterNode(level=1, title="Empty Root", content="")
        real_root = ChapterNode(level=1, title="Real Root", content="Real content here.")
        batches = BatchBuilder.build_batches([empty_root, real_root], max_chars=1000)
        assert len(batches) == 1
        assert batches[0][0]["title"] == "Real Root"

    def test_filters_section_with_empty_content(self):
        """无子节点且 content 为空的 section 应被过滤."""
        root = ChapterNode(level=1, title="Doc", content="")
        sec_empty = ChapterNode(level=2, title="Empty Section", content="")
        sec_real = ChapterNode(level=2, title="Real Section", content="Real content.")
        root.children = [sec_empty, sec_real]
        batches = BatchBuilder.build_batches([root], max_chars=1000)
        assert len(batches) == 1
        assert batches[0][0]["title"] == "Real Section"

    def test_filters_empty_chunk_node(self):
        """Content 为空的 chunk 节点应被过滤，不影响后续 chunk."""
        root = ChapterNode(level=1, title="Doc", content="")
        sec = ChapterNode(level=2, title="Section", content="")
        sec.children = [
            ChapterNode(level=3, title="Empty Chunk", content=""),
            ChapterNode(level=3, title="Real Chunk", content="Real content."),
        ]
        root.children = [sec]
        batches = BatchBuilder.build_batches([root], max_chars=1000)
        assert len(batches) == 1
        assert batches[0][0]["title"] == "Real Chunk"

    def test_all_empty_nodes_yield_empty_batches(self):
        """全部节点都为空时应返回空列表."""
        root = ChapterNode(level=1, title="Doc", content="")
        sec = ChapterNode(level=2, title="Section", content="")
        root.children = [sec]
        batches = BatchBuilder.build_batches([root], max_chars=1000)
        assert batches == []
