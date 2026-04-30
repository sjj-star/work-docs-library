#!/usr/bin/env python3
"""阶段2: Markdown → Batch JSONL.

用法:
    venv/bin/python scripts/tools/stage2_build_jsonl.py <doc_id> [--max-chars 10000]

输出:
    jsonl_path, batch_count, request_count
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
    """执行阶段2构建 JSONL."""
    parser = argparse.ArgumentParser(description="Markdown 生成 Batch JSONL")
    parser.add_argument("doc_id", help="文档 ID（即 parsed 目录名）")
    parser.add_argument("--max-chars", type=int, default=10000, help="每个 batch 最大字符数")
    args = parser.parse_args()

    pipe = DocGraphPipeline()
    jsonl_path, batches, requests = pipe.stage2_build_jsonl(
        args.doc_id, max_chars=args.max_chars
    )

    print(f"jsonl={jsonl_path}")
    print(f"batch_count={len(batches)}")
    print(f"request_count={len(requests)}")
    print(f"\n可审查 JSONL: {jsonl_path}")
    print(f"下一阶段: venv/bin/python scripts/tools/stage3_ingest.py {args.doc_id}")


if __name__ == "__main__":
    main()
