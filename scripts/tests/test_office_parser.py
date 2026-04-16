import docx
import openpyxl
import pytest

from parsers.office_parser import OfficeParser


def test_parse_docx(tmp_path):
    doc_path = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_paragraph("Hello world")
    doc.add_paragraph("Second paragraph")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "C"
    table.cell(1, 1).text = "D"
    doc.save(str(doc_path))

    parser = OfficeParser()
    result = parser.parse(str(doc_path))
    assert result.file_type == "docx"
    assert "Hello world" in result.chunks[0].content
    assert "A | B" in result.chunks[0].content
    assert result.total_pages == 1


def test_parse_xlsx(tmp_path):
    xlsx_path = tmp_path / "sample.xlsx"
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["Name", "Age"])
    ws1.append(["Alice", "30"])
    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["City", "Country"])
    ws2.append(["Beijing", "China"])
    wb.save(str(xlsx_path))

    parser = OfficeParser()
    result = parser.parse(str(xlsx_path))
    assert result.file_type == "xlsx"
    assert result.total_pages == 2
    assert "Sheet: Sheet1" in result.chunks[0].content
    assert "Alice | 30" in result.chunks[0].content
    assert "Sheet: Sheet2" in result.chunks[0].content
    assert len(result.chapters) == 2


def test_unsupported_suffix(tmp_path):
    bad = tmp_path / "bad.pptx"
    bad.write_text("x")
    parser = OfficeParser()
    with pytest.raises(ValueError):
        parser.parse(str(bad))
