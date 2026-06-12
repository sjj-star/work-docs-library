"""test_table_utils 模块."""

from __future__ import annotations

from parsers.table_utils import normalize_markdown_table


def test_normalize_basic_table():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = normalize_markdown_table(md)
    lines = result.splitlines()
    assert len(lines) == 3
    assert all(line.startswith("|") and line.endswith("|") for line in lines)
    assert "| A | B |" in result
    assert "| 1 | 2 |" in result


def test_normalize_table_without_outer_pipes():
    md = "A | B\n---|---\n1 | 2"
    result = normalize_markdown_table(md)
    assert result.startswith("|")
    assert "| A | B |" in result
    assert "| 1 | 2 |" in result


def test_normalize_table_collapses_whitespace_and_br():
    md = "  | Header 1 | Header 2 |  \n|---|---|\n| line1<br>line2 |   cell   |"
    result = normalize_markdown_table(md)
    assert "<br>" not in result
    assert "line1 line2" in result
    assert "| Header 1 | Header 2 |" in result
    assert "| line1 line2 | cell |" in result


def test_normalize_empty_lines_removed():
    md = "\n| A |\n\n| B |\n"
    result = normalize_markdown_table(md)
    lines = result.splitlines()
    assert lines == ["| A |", "| B |"]
