#!/usr/bin/env python3
"""PyMuPDF4LLM 解析执行脚本."""

import logging
import sys
import time
import tracemalloc
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def run(pdf_path: str, out_dir: str) -> dict:
    """运行 PyMuPDF4LLM 解析，返回性能指标."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        # PyMuPDF4LLM to_markdown 支持图片提取到目录
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            write_images=True,
            image_path=str(out_dir / "images"),
        )
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        md_path = out_dir / "result.md"
        md_path.write_text(md_text, encoding="utf-8")

        # 统计提取的图片
        img_dir = out_dir / "images"
        img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0

        result = {
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "peak_memory_mb": round(peak / (1024 * 1024), 2),
            "output_chars": len(md_text),
            "output_images": img_count,
            "output_md_path": str(md_path),
            "output_img_dir": str(img_dir),
        }
        logger.info(
            f"PyMuPDF4LLM [{pdf_path.name}] 成功: {elapsed:.1f}s, "
            f"{result['peak_memory_mb']}MB, {result['output_chars']} chars, "
            f"{result['output_images']} images"
        )
        return result

    except Exception as e:
        tracemalloc.stop()
        elapsed = time.perf_counter() - t0
        logger.exception(f"PyMuPDF4LLM [{pdf_path.name}] 失败: {e}")
        return {"status": "failed", "error": str(e), "elapsed_sec": round(elapsed, 2)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_pymupdf4llm.py <pdf_path> <out_dir>")
        sys.exit(1)
    result = run(sys.argv[1], sys.argv[2])
    print(result)
