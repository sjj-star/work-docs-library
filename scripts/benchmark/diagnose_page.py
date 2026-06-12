"""PDFParser 单页深度诊断脚本。

输出指定页面的完整内部状态，用于分析 _find_figure_regions / _detect_and_convert_tables 的行为。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from parsers.pdf_parser import PDFParser


def diagnose_page(pdf_path: str, page_1idx: int, output_path: str | None = None) -> dict:
    """诊断指定页面的图片提取内部状态。

    Args:
        pdf_path: PDF 文件路径
        page_1idx: 1-indexed 页码
        output_path: 可选，JSON 输出路径
    """
    parser = PDFParser()
    doc = fitz.open(pdf_path)
    page_idx = page_1idx - 1
    page = doc.load_page(page_idx)
    page_rect = page.rect

    # 1. 文本块（含分类）
    text_dict = page.get_text("dict")
    text_blocks = []
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        txt = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
        if not txt:
            continue
        sizes = []
        for line in block["lines"]:
            for span in line["spans"]:
                sizes.extend([span["size"]] * len(span["text"]))
        avg_size = sum(sizes) / len(sizes) if sizes else 12.0
        text_blocks.append((txt, fitz.Rect(block["bbox"]), avg_size))
    text_blocks.sort(key=lambda x: x[1].y0)
    classified = parser._classify_text_blocks_for_figures(text_blocks, page_rect)

    classified_json = []
    for txt, rect, size, cat in classified:
        classified_json.append({
            "text": txt[:200],
            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
            "avg_size": size,
            "category": cat,
        })

    # 2. diagram regions
    diagram_regions = parser._find_figure_regions(page, page_rect, 60, 60)
    diagram_captions = parser._get_diagram_captions(page, page_rect, diagram_regions)

    # 3. find_tables
    table_elements = []
    try:
        tabs = page.find_tables(strategy="lines_strict")
        if tabs:
            for tab in tabs.tables:
                table_elements.append({
                    "bbox": list(tab.bbox),
                    "row_count": tab.row_count,
                    "col_count": tab.col_count,
                    "to_markdown": tab.to_markdown(clean=False)[:300],
                })
    except Exception as e:
        table_elements = [{"error": str(e)}]

    # 4. raw drawings
    raw_drawings = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r:
            raw_drawings.append(list(r))

    result = {
        "pdf_path": str(pdf_path),
        "page_1idx": page_1idx,
        "page_rect": [page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1],
        "classified_blocks": classified_json,
        "diagram_regions": [list(r) for r in diagram_regions],
        "diagram_captions": {str(k): v for k, v in diagram_captions.items()},
        "find_tables_raw": table_elements,
        "raw_drawings_count": len(raw_drawings),
        "raw_drawings_sample": raw_drawings[:50],
    }

    doc.close()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_page.py <pdf_path> <page_1idx> [output.json]")
        sys.exit(1)
    diagnose_page(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
