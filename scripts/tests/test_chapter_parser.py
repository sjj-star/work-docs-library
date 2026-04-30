"""test_chapter_parser 模块."""

from core.doc_graph_pipeline import ChapterParser


class TestChapterParserIsHeading:
    """测试 ChapterParser._is_heading 的标题识别逻辑."""

    def test_markdown_heading_basic(self):
        """基本 Markdown 标题识别."""
        result = ChapterParser._is_heading("# Introduction")
        assert result == (1, "Introduction")

    def test_markdown_heading_level2(self):
        """二级 Markdown 标题识别."""
        result = ChapterParser._is_heading("## 1 Introduction")
        assert result == (2, "1 Introduction")

    def test_markdown_heading_level3(self):
        """三级 Markdown 标题识别."""
        result = ChapterParser._is_heading("### 2.1 Controlling the HRPWM Capabilities")
        assert result == (3, "2.1 Controlling the HRPWM Capabilities")

    def test_rejects_pure_number_heading(self):
        r"""拒绝纯数字 Markdown 标题（页码噪声）."""
        assert ChapterParser._is_heading("# 1") is None
        assert ChapterParser._is_heading("## 42") is None
        assert ChapterParser._is_heading("### 999") is None

    def test_rejects_date_heading_month_year(self):
        r"""拒绝日期格式 Markdown 标题（封面日期噪声）."""
        assert ChapterParser._is_heading("# 16 March 2001") is None
        assert ChapterParser._is_heading("# 10 May 2004") is None
        assert ChapterParser._is_heading("# 1 January 2024") is None
        assert ChapterParser._is_heading("## 31 December 1999") is None

    def test_rejects_date_heading_iso(self):
        r"""拒绝 ISO 日期格式 Markdown 标题."""
        assert ChapterParser._is_heading("# 2024-01-15") is None
        assert ChapterParser._is_heading("# 2001-03-16") is None

    def test_accepts_heading_with_number_prefix(self):
        """接受带编号前缀的有效标题（非纯数字）."""
        result = ChapterParser._is_heading("# 1 Introduction")
        assert result == (1, "1 Introduction")

        result = ChapterParser._is_heading("# Chapter 1: Overview")
        assert result == (1, "Chapter 1: Overview")

    def test_rejects_lowercase_date(self):
        r"""拒绝小写日期格式."""
        assert ChapterParser._is_heading("# 16 march 2001") is None

    def test_rejects_toc_line_without_hash(self):
        r"""拒绝没有 # 前缀的目录行（如 "1 Introduction 7"）."""
        assert ChapterParser._is_heading("1 Introduction 7") is None
        assert ChapterParser._is_heading("2.1 Controlling the HRPWM Capabilities 10") is None
        assert ChapterParser._is_heading("2 Operational Description of HRPWM 9") is None

    def test_rejects_code_comment_line(self):
        r"""代码注释行（如 C 的 #include）不被识别为标题."""
        # Markdown 标题要求 # 后必须有空格，因此 #include 不会被匹配
        assert ChapterParser._is_heading('#include "DSP280x_Device.h"') is None
        assert ChapterParser._is_heading("#define MAX 100") is None
        assert ChapterParser._is_heading("#ifdef DEBUG") is None
        # 有效的 Markdown 标题应被匹配（# 后有空格）
        assert ChapterParser._is_heading("# include") == (1, "include")

    def test_rejects_plain_numbered_line(self):
        r"""拒绝纯数字编号行（不再识别数字编号标题）."""
        assert ChapterParser._is_heading("1.1 Overview") is None
        assert ChapterParser._is_heading("2.3 Register Descriptions") is None
        assert ChapterParser._is_heading("3: Conclusion") is None

    def test_rejects_chinese_numbered_line(self):
        r"""拒绝中文编号行（不再识别中文编号标题）."""
        assert ChapterParser._is_heading("一、概述") is None
        assert ChapterParser._is_heading("（一）特性") is None


class TestChapterParserParseFlat:
    """测试 ChapterParser.parse_flat 的代码块保护."""

    def test_skips_heading_inside_fenced_code_block(self):
        r"""Fenced code block 内的 # 行不应被识别为标题."""
        text = """# Main Title

Some intro.

```c
#include "header.h"
#define MAX 100
```

## Next Section

Content here.
"""
        flat = ChapterParser.parse_flat(text)
        titles = [ch["title"] for ch in flat]
        assert titles == ["Main Title", "Next Section"]
        # #include 应该被归入 Main Title 的 content
        main_content = flat[0]["content"]
        assert '#include "header.h"' in main_content
        assert "#define MAX 100" in main_content

    def test_skips_heading_inside_tilde_code_block(self):
        r"""~~~ 代码块内的 # 行不应被识别为标题."""
        text = """# Title

~~~python
# This is a comment
x = 1
~~~

## Section
"""
        flat = ChapterParser.parse_flat(text)
        titles = [ch["title"] for ch in flat]
        assert titles == ["Title", "Section"]

    def test_toc_lines_become_content(self):
        r"""没有 # 前缀的目录行应成为正文 content，而非独立章节."""
        text = """# Document

1 Introduction 7
2 Operational Description 9
2.1 Capabilities 10

## 1 Introduction

This is the real intro.
"""
        flat = ChapterParser.parse_flat(text)
        titles = [ch["title"] for ch in flat]
        # 目录行不应被识别为 heading
        assert "1 Introduction 7" not in titles
        assert "2 Operational Description 9" not in titles
        assert "2.1 Capabilities 10" not in titles
        # 真正的 Markdown 标题应被识别
        assert "Document" in titles
        assert "1 Introduction" in titles
        # 目录行应归入 Document 的 content
        doc_content = flat[0]["content"]
        assert "1 Introduction 7" in doc_content
        assert "2 Operational Description 9" in doc_content

    def test_empty_input(self):
        """空文本应返回空列表."""
        flat = ChapterParser.parse_flat("")
        assert flat == []

    def test_no_headings(self):
        """没有标题的文本应返回一个包含全部内容的章节."""
        flat = ChapterParser.parse_flat("Just some text.\n\nMore text.")
        assert len(flat) == 0


class TestChapterParserParseTree:
    """测试 ChapterParser.parse_tree 的树形结构和 preface 传播."""

    def test_no_preface_propagation(self):
        """去掉 preface 传播后，每个节点保留自己的 content."""
        text = """# Title
## Section 1
section intro
### Sub 1.1
sub content
"""
        tree = ChapterParser.parse_tree(text)

        def _collect_all_nodes(node, ancestors=None):
            ancestors = ancestors or []
            result = []
            if node.content:
                result.append(node)
            for child in node.children:
                result.extend(_collect_all_nodes(child, ancestors + [node.title]))
            return result

        all_nodes = []
        for root in tree:
            all_nodes.extend(_collect_all_nodes(root))

        assert len(all_nodes) == 2
        titles = [n.title for n in all_nodes]
        assert "Section 1" in titles
        assert "Sub 1.1" in titles

        # Sub 1.1 只包含自己的 content
        sub11 = next(n for n in all_nodes if n.title == "Sub 1.1")
        assert sub11.content == "sub content"

        # Section 1 保留自己的 content
        sec1 = next(n for n in all_nodes if n.title == "Section 1")
        assert sec1.content == "section intro"

    def test_multiple_nodes_preserve_own_content(self):
        """所有有 content 的节点应各自保留独立内容."""
        text = """# Title
## Section 1
### Sub 1.1
content 1
### Sub 1.2
content 2
## Section 2
content 3
"""
        tree = ChapterParser.parse_tree(text)

        def _collect_all_nodes(node, ancestors=None):
            ancestors = ancestors or []
            result = []
            if node.content:
                result.append(node)
            for child in node.children:
                result.extend(_collect_all_nodes(child, ancestors + [node.title]))
            return result

        all_nodes = []
        for root in tree:
            all_nodes.extend(_collect_all_nodes(root))

        assert len(all_nodes) == 3
        titles = {n.title for n in all_nodes}
        assert titles == {"Sub 1.1", "Sub 1.2", "Section 2"}

        sub11 = next(n for n in all_nodes if n.title == "Sub 1.1")
        assert sub11.content == "content 1"

        sub12 = next(n for n in all_nodes if n.title == "Sub 1.2")
        assert sub12.content == "content 2"

        sec2 = next(n for n in all_nodes if n.title == "Section 2")
        assert sec2.content == "content 3"

    def test_no_children_node_keeps_content(self):
        """没有子节点的节点应保持自身内容."""
        text = """# Title
## Section 1
content 1
## Section 2
content 2
"""
        tree = ChapterParser.parse_tree(text)

        def _collect_all_nodes(node, ancestors=None):
            ancestors = ancestors or []
            result = []
            if node.content:
                result.append(node)
            for child in node.children:
                result.extend(_collect_all_nodes(child, ancestors + [node.title]))
            return result

        all_nodes = []
        for root in tree:
            all_nodes.extend(_collect_all_nodes(root))

        assert len(all_nodes) == 2
        for n in all_nodes:
            assert n.content != ""
