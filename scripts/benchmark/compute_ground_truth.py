#!/usr/bin/env python3
"""预计算 PDF 基准数据（ground truth），供评测对比使用."""

import json
import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

DOCS = {
    "tms320f28335": "/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/tms320f28335.pdf",
    "amba_chi": "/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/IHI0050G_amba_chi_architecture_spec.pdf",
    "dc_ug": "/mnt/c/Users/SJJ22/Downloads/Doc/EDA Doc/Design Compiler User Guide.pdf",
    "sprui07": "/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/sprui07.pdf",
}


def compute_ground_truth(doc_name: str, pdf_path: str, out_dir: Path) -> Path:
    """为单个 PDF 计算基准数据."""
    logger.info(f"[{doc_name}] 开始计算 ground truth: {pdf_path}")
    doc = fitz.open(pdf_path)

    total_chars = 0
    total_images = 0
    total_drawings = 0
    tables_per_page = []
    toc = doc.get_toc()

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # 文本字符数
        text = page.get_text()
        total_chars += len(text)

        # 图片数
        imgs = page.get_images(full=True)
        total_images += len(imgs)

        # Drawing primitives (矢量图)
        drawings = page.get_drawings()
        total_drawings += len(drawings)

        # 表格检测
        tables = page.find_tables()
        tables_per_page.append(len(tables.tables))

    gt = {
        "doc_name": doc_name,
        "pdf_path": pdf_path,
        "pages": len(doc),
        "total_chars": total_chars,
        "total_images": total_images,
        "total_drawings": total_drawings,
        "tables_per_page": tables_per_page,
        "total_tables": sum(tables_per_page),
        "toc": toc,
        "toc_entries": len(toc),
    }

    out_path = out_dir / f"{doc_name}_gt.json"
    out_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"[{doc_name}] 完成: {gt['pages']}页, "
        f"{gt['total_chars']}字符, {gt['total_images']}图片, "
        f"{gt['total_drawings']}drawings, {gt['total_tables']}表格, "
        f"{gt['toc_entries']} TOC entries -> {out_path}"
    )
    doc.close()
    return out_path


def main() -> None:
    """为所有配置文档计算并保存 ground truth 统计."""
    out_dir = Path("/tmp/workdocs_benchmark/ground_truth")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path in DOCS.items():
        compute_ground_truth(name, path, out_dir)

    logger.info("所有文档 ground truth 计算完成")


if __name__ == "__main__":
    main()
