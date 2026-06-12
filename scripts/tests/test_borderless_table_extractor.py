"""BorderlessTableExtractor 测试."""

from __future__ import annotations

import fitz
from parsers.borderless_table_extractor import BorderlessTableExtractor


def _make_borderless_table_pdf(path: str, caption_y: float = 150) -> fitz.Document:
    """生成一个仅有横线、无竖线的类 AMBA 表格 PDF."""
    doc = fitz.open()
    page = doc.new_page()

    # caption
    page.insert_text((72, caption_y), "Table 1-1. Borderless style table")

    x0, y0 = 72, 200
    col_lefts = [72, 220, 360]
    row_heights = [30, 60, 30]
    row_tops = [y0]
    for h in row_heights:
        row_tops.append(row_tops[-1] + h)

    # horizontal lines (full width)
    table_width = 460
    for y in row_tops:
        page.draw_line((x0, y), (x0 + table_width, y), color=(0, 0, 0))

    # header row
    page.insert_text((col_lefts[0], y0 + 20), "Layer")
    page.insert_text((col_lefts[1], y0 + 20), "Granularity")
    page.insert_text((col_lefts[2], y0 + 20), "Function")

    # data row 1: multi-line description in col 3
    row1_y = row_tops[1]
    page.insert_text((col_lefts[0], row1_y + 20), "Protocol")
    page.insert_text((col_lefts[1], row1_y + 20), "Transaction")
    page.insert_text((col_lefts[2], row1_y + 20), "Generates requests")
    page.insert_text((col_lefts[2], row1_y + 35), "and processes responses.")

    # data row 2
    row2_y = row_tops[2]
    page.insert_text((col_lefts[0], row2_y + 20), "Network")
    page.insert_text((col_lefts[1], row2_y + 20), "Packet")
    page.insert_text((col_lefts[2], row2_y + 20), "Packetizes messages.")

    doc.save(path)
    doc.close()
    return fitz.open(path)


def test_extract_horizontal_line_table(tmp_path):
    pdf_path = tmp_path / "borderless.pdf"
    doc = _make_borderless_table_pdf(str(pdf_path))
    page = doc[0]

    extractor = BorderlessTableExtractor()
    results = extractor.extract(page, fitz.Rect(0, 140, page.rect.width, 340))

    assert len(results) == 1
    md, bbox = results[0]
    assert "| Layer | Granularity | Function |" in md
    assert "| Protocol | Transaction |" in md
    assert "Network" in md
    assert "Packetizes messages" in md
    # multi-line description cell should be merged into a single cell, not split
    assert "Generates requests" in md
    assert "processes responses" in md

    # bbox should cover the table area
    assert bbox.y0 <= 200
    assert bbox.y1 >= 320

    doc.close()


def test_extract_ignores_lines_outside_search_rect(tmp_path):
    pdf_path = tmp_path / "borderless.pdf"
    doc = _make_borderless_table_pdf(str(pdf_path))
    page = doc[0]

    extractor = BorderlessTableExtractor()
    # search rect entirely above the table
    results = extractor.extract(page, fitz.Rect(0, 50, page.rect.width, 120))
    assert results == []

    doc.close()


def test_extract_requires_multiple_lines(tmp_path):
    pdf_path = tmp_path / "single_line.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 200), "Table 1-1. Single")
    page.draw_line((72, 240), (500, 240), color=(0, 0, 0))
    doc.save(str(pdf_path))
    doc.close()

    doc = fitz.open(str(pdf_path))
    page = doc[0]
    extractor = BorderlessTableExtractor()
    results = extractor.extract(page, fitz.Rect(0, 180, page.rect.width, 300))
    assert results == []
    doc.close()
