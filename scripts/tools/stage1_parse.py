#!/usr/bin/env python3
"""阶段1: PDF → Markdown.

用法:
    venv/bin/python scripts/tools/stage1_parse.py <pdf_path>

输出:
    doc_id, parsed_dir, chars
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.doc_graph_pipeline import DocGraphPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def main() -> None:
    """执行阶段1解析."""
    parser = argparse.ArgumentParser(description="PDF 解析为 Markdown")
    parser.add_argument("pdf_path", help="PDF 文件路径")
    args = parser.parse_args()

    pipe = DocGraphPipeline()
    doc_id, output_dir, text, images = pipe.stage1_parse(args.pdf_path)

    print(f"doc_id={doc_id}")
    print(f"parsed_dir={output_dir}")
    print(f"chars={len(text)}")
    print(f"images={len(images)}")
    print(f"\n可手动编辑: {output_dir}/result.md")
    print(f"下一阶段: venv/bin/python scripts/tools/stage2_build_jsonl.py {doc_id}")


if __name__ == "__main__":
    main()
