"""test_chapter_parser 模块."""

from core.doc_graph_pipeline import ChapterParser


class TestChapterParserIsHeading:
    """测试 ChapterParser._is_heading 的标题识别逻辑."""

    def test_markdown_heading_basic(self):
        """基本 Markdown 标题识别."""
        result = ChapterParser._is_heading("# Introduction")
        assert result == (1, "Introduction")

    def test_numbered_heading(self):
        """数字编号标题."""
        result = ChapterParser._is_heading("1.1 Overview")
        assert result == (2, "1.1 Overview")

    def test_chinese_heading(self):
        """中文编号标题."""
        result = ChapterParser._is_heading("一、概述")
        assert result == (1, "一、概述")

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

    def test_non_markdown_number_still_works(self):
        """非 Markdown 格式的数字编号标题仍然正常工作."""
        result = ChapterParser._is_heading("2.3 Register Descriptions")
        assert result == (2, "2.3 Register Descriptions")

    def test_rejects_date_in_numbered_format(self):
        r"""拒绝数字编号分支中的日期格式（如 \"16 March 2001\"）."""
        assert ChapterParser._is_heading("16 March 2001") is None
        assert ChapterParser._is_heading("10 May 2004") is None

    def test_rejects_page_number_in_numbered_format(self):
        r"""拒绝数字编号分支中的页码格式（如 \"1 1\"）."""
        assert ChapterParser._is_heading("1 1") is None
        assert ChapterParser._is_heading("1 2") is None
        assert ChapterParser._is_heading("2 15") is None

    def test_rejects_pure_number_after_separator(self):
        r"""拒绝数字编号后仅跟数字的标题（如 \"Page 5\" 中的 \"5\"）."""
        # "5" 本身不会被数字编号分支匹配（没有前缀数字），
        # 但 "1 5" 会被匹配为 number=1, title="5"
        assert ChapterParser._is_heading("1 5") is None
