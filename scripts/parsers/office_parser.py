import hashlib
from pathlib import Path
from typing import List

import docx
import openpyxl

from core.models import Chunk, Document, Chapter


class OfficeParser:
    SUPPORTED = (".docx", ".xlsx")

    def parse(self, path: str) -> Document:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".docx":
            return self._parse_docx(path)
        if suffix == ".xlsx":
            return self._parse_xlsx(path)
        raise ValueError(f"Unsupported office file: {suffix}")

    def _parse_docx(self, path: str) -> Document:
        document = docx.Document(path)
        title = Path(path).stem
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        tables = []
        for table in document.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            tables.append("\n".join(rows))

        content = "\n\n".join(paragraphs)
        if tables:
            content += "\n\n[Tables]\n" + "\n\n".join(tables)

        file_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()
        return Document(
            doc_id=file_hash,
            title=title,
            source_path=str(Path(path).resolve()),
            file_type="docx",
            total_pages=1,
            chapters=[Chapter(title="全文", start_page=1, end_page=1, level=1)],
            chunks=[Chunk(
                doc_id=file_hash,
                chunk_id="docx_full",
                content=content,
                chunk_type="text",
                page_start=1,
                page_end=1,
            )],
            file_hash=file_hash,
            status="pending",
        )

    def _parse_xlsx(self, path: str) -> Document:
        wb = openpyxl.load_workbook(path, data_only=True)
        title = Path(path).stem
        sheets_text = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                if row_text.strip():
                    rows.append(row_text)
            sheets_text.append(f"Sheet: {sheet_name}\n" + "\n".join(rows))

        content = "\n\n".join(sheets_text)
        file_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()
        total = len(wb.sheetnames)
        return Document(
            doc_id=file_hash,
            title=title,
            source_path=str(Path(path).resolve()),
            file_type="xlsx",
            total_pages=total,
            chapters=[Chapter(title=sheet, start_page=i + 1, end_page=i + 1, level=1) for i, sheet in enumerate(wb.sheetnames)],
            chunks=[Chunk(
                doc_id=file_hash,
                chunk_id="xlsx_full",
                content=content,
                chunk_type="table",
                page_start=1,
                page_end=total,
            )],
            file_hash=file_hash,
            status="pending",
        )
