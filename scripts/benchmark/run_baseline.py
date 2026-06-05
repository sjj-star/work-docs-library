#!/usr/bin/env python3
"""Baseline PDFParser 解析执行脚本."""

import logging
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.pdf_parser import PDFParser

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def run(pdf_path: str, out_dir: str) -> dict:
    """运行 Baseline 解析，返回性能指标."""
    pdf_path_obj = Path(pdf_path)
    out_dir_obj = Path(out_dir)
    out_dir_obj.mkdir(parents=True, exist_ok=True)

    parser = PDFParser()

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        # PDFParser.parse() 内部会自动创建 output_dir/images/ 子目录
        md_text, image_paths = parser.parse(pdf_path_obj, out_dir_obj)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        md_path = out_dir_obj / "result.md"
        md_path.write_text(md_text, encoding="utf-8")

        result = {
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "peak_memory_mb": round(peak / (1024 * 1024), 2),
            "output_chars": len(md_text),
            "output_images": len(image_paths),
            "output_md_path": str(md_path),
            "output_img_dir": str(out_dir_obj / "images"),
        }
        logger.info(
            f"Baseline [{pdf_path_obj.name}] 成功: {elapsed:.1f}s, "
            f"{result['peak_memory_mb']}MB, {result['output_chars']} chars, "
            f"{result['output_images']} images"
        )
        return result

    except Exception as e:
        tracemalloc.stop()
        elapsed = time.perf_counter() - t0
        logger.exception(f"Baseline [{pdf_path_obj.name}] 失败: {e}")
        return {"status": "failed", "error": str(e), "elapsed_sec": round(elapsed, 2)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_baseline.py <pdf_path> <out_dir>")
        sys.exit(1)
    result = run(sys.argv[1], sys.argv[2])
    print(result)
