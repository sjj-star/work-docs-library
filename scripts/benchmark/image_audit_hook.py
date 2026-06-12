"""PDFParser 图片提取策略审计 Hook。

用法：
    python scripts/benchmark/image_audit_hook.py <pdf_path> <out_dir>

输出：
    - audit_report.json   每页策略激活统计
    - audit_pages/        代表性页面截图（原PDF vs 提取图片）
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from parsers.pdf_parser import PDFParser


class AuditingPDFParser(PDFParser):
    """继承 PDFParser，在关键方法中注入审计统计。"""

    def __init__(self):
        """初始化审计日志与页码跟踪状态."""
        super().__init__()
        self._audit_log: list[dict] = []
        self._current_page: int = -1

    def _audit(self, event: str, data: dict | None = None):
        self._audit_log.append(
            {
                "page": self._current_page,
                "event": event,
                "data": data or {},
                "ts": time.perf_counter(),
            }
        )

    def parse(self, pdf_path, output_dir):
        """包装 PDFParser.parse 并记录各阶段耗时."""
        self._audit_log = []
        # Hook doc.load_page 以追踪当前页码
        original_load_page = fitz.Document.load_page

        def hooked_load_page(doc, page_id):
            # page_id 可以是页码(int)或页名(str)；仅记录整数页码
            if isinstance(page_id, int):
                self._current_page = page_id
            return original_load_page(doc, page_id)

        fitz.Document.load_page = hooked_load_page
        try:
            result = super().parse(pdf_path, output_dir)
        finally:
            fitz.Document.load_page = original_load_page
        return result

    def _find_figure_regions(self, page, page_rect, header_margin=0.0, footer_margin=0.0):
        t0 = time.perf_counter()
        result = super()._find_figure_regions(page, page_rect, header_margin, footer_margin)
        elapsed = time.perf_counter() - t0
        self._audit(
            "find_figure_regions",
            {
                "elapsed_ms": round(elapsed * 1000, 2),
                "regions_found": len(result),
            },
        )
        return result

    def _extract_raster_images(self, doc, page, page_idx, img_dir, diagram_regions=None):
        t0 = time.perf_counter()
        result = super()._extract_raster_images(doc, page, page_idx, img_dir, diagram_regions)
        elapsed = time.perf_counter() - t0
        self._audit(
            "extract_raster_images",
            {
                "elapsed_ms": round(elapsed * 1000, 2),
                "images_found": len(result),
                "image_names": [img["path"] for img in result],
            },
        )
        return result

    def _render_diagrams(self, page, page_idx, diagram_regions, diagram_captions, img_dir):
        t0 = time.perf_counter()
        result = super()._render_diagrams(
            page, page_idx, diagram_regions, diagram_captions, img_dir
        )
        elapsed = time.perf_counter() - t0
        self._audit(
            "render_diagrams",
            {
                "elapsed_ms": round(elapsed * 1000, 2),
                "images_rendered": len(result),
                "image_names": [img["path"] for img in result],
            },
        )
        return result

    def _detect_and_convert_tables(self, page, text_blocks, diagram_regions):
        t0 = time.perf_counter()
        result = super()._detect_and_convert_tables(page, text_blocks, diagram_regions)
        elapsed = time.perf_counter() - t0
        self._audit(
            "detect_and_convert_tables",
            {
                "elapsed_ms": round(elapsed * 1000, 2),
                "tables_found": len(result),
            },
        )
        return result

    def get_audit_summary(self) -> dict:
        """汇总审计日志为每页统计。"""
        pages: dict[int, dict] = {}
        for entry in self._audit_log:
            p = entry["page"]
            if p not in pages:
                pages[p] = {
                    "page": p,
                    "find_figure_regions_ms": 0,
                    "find_figure_regions_count": 0,
                    "extract_raster_images_ms": 0,
                    "extract_raster_images_count": 0,
                    "render_diagrams_ms": 0,
                    "render_diagrams_count": 0,
                    "detect_tables_ms": 0,
                    "detect_tables_count": 0,
                }
            ev = entry["event"]
            d = entry["data"]
            if ev == "find_figure_regions":
                pages[p]["find_figure_regions_ms"] = d["elapsed_ms"]
                pages[p]["find_figure_regions_count"] = d["regions_found"]
            elif ev == "extract_raster_images":
                pages[p]["extract_raster_images_ms"] = d["elapsed_ms"]
                pages[p]["extract_raster_images_count"] = d["images_found"]
            elif ev == "render_diagrams":
                pages[p]["render_diagrams_ms"] = d["elapsed_ms"]
                pages[p]["render_diagrams_count"] = d["images_rendered"]
            elif ev == "detect_and_convert_tables":
                pages[p]["detect_tables_ms"] = d["elapsed_ms"]
                pages[p]["detect_tables_count"] = d["tables_found"]

        return {
            "total_events": len(self._audit_log),
            "pages": list(pages.values()),
        }


def run_audit(pdf_path: str, out_dir: str) -> dict:
    """对指定 PDF 运行 AuditingPDFParser 并输出审计报告."""
    pdf_path_obj = Path(pdf_path)
    out_dir_obj = Path(out_dir)
    out_dir_obj.mkdir(parents=True, exist_ok=True)

    parser = AuditingPDFParser()
    t0 = time.perf_counter()
    md_text, image_paths = parser.parse(pdf_path_obj, out_dir_obj)
    elapsed = time.perf_counter() - t0

    summary = parser.get_audit_summary()
    summary["pdf_name"] = pdf_path_obj.name
    summary["total_pages"] = len(fitz.open(str(pdf_path_obj)))
    summary["total_elapsed_sec"] = round(elapsed, 2)
    summary["total_images_extracted"] = len(image_paths)
    summary["output_md_chars"] = len(md_text)

    report_path = out_dir_obj / "audit_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 选择有代表性的页面截图
    _snapshot_representative_pages(pdf_path_obj, out_dir_obj, summary["pages"])

    print(
        f"[AUDIT] {pdf_path_obj.name}: {elapsed:.1f}s, "
        f"{summary['total_images_extracted']} images, "
        f"report -> {report_path}"
    )
    return summary


def _snapshot_representative_pages(pdf_path: Path, out_dir: Path, pages: list[dict]):
    """对代表性页面截图：原PDF页面 + 提取的图片并列展示。"""
    doc = fitz.open(str(pdf_path))
    snap_dir = out_dir / "audit_pages"
    snap_dir.mkdir(exist_ok=True)

    # 选择标准：
    # 1. diagram 渲染最多的页面
    # 2. raster 图片最多的页面
    # 3. table 检测最多的页面
    # 4. 随机抽样（每100页抽1页）
    selected: set[int] = set()

    by_diagram = sorted(pages, key=lambda x: x["render_diagrams_count"], reverse=True)[:3]
    by_raster = sorted(pages, key=lambda x: x["extract_raster_images_count"], reverse=True)[:3]
    by_table = sorted(pages, key=lambda x: x["detect_tables_count"], reverse=True)[:3]
    by_random = [pages[i] for i in range(0, len(pages), max(1, len(pages) // 5))][:3]

    for p in by_diagram + by_raster + by_table + by_random:
        selected.add(p["page"])

    for pnum in sorted(selected):
        if pnum < 0 or pnum >= len(doc):
            continue
        page = doc.load_page(pnum)
        pix = page.get_pixmap(dpi=150)
        snap_path = snap_dir / f"page_{pnum + 1:04d}_original.png"
        pix.save(str(snap_path))

    doc.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python image_audit_hook.py <pdf_path> <out_dir>")
        sys.exit(1)
    run_audit(sys.argv[1], sys.argv[2])
