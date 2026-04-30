#!/usr/bin/env python3
"""阶段3: JSONL → API → 入库.

用法:
    venv/bin/python scripts/tools/stage3_ingest.py <doc_id> \
        [--jsonl PATH] [--file-path PATH] [--force]

输出:
    doc_id
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.doc_graph_pipeline import DocGraphPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def main() -> None:
    """执行阶段3入库."""
    parser = argparse.ArgumentParser(description="提交 JSONL 并完成入库")
    parser.add_argument("doc_id", help="文档 ID")
    parser.add_argument("--jsonl", help="自定义 JSONL 路径")
    parser.add_argument("--file-path", help="原始 PDF 文件路径（若数据库中无记录则必填）")
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    args = parser.parse_args()

    pipe = DocGraphPipeline()

    # 获取原始文件路径
    file_path = args.file_path
    if not file_path:
        doc = pipe.db.get_document(args.doc_id)
        if doc:
            file_path = doc.source_path
        else:
            print("error: 无法找到源文件路径，请提供 --file-path", file=sys.stderr)
            sys.exit(1)

    parsed_output_dir = Path(Config.DB_PATH).parent / "parsed" / args.doc_id
    result_md = parsed_output_dir / "result.md"
    if not result_md.exists():
        print(f"error: result.md 不存在 | path={result_md}", file=sys.stderr)
        sys.exit(1)

    extracted_text = result_md.read_text(encoding="utf-8")
    jsonl_path = Path(args.jsonl) if args.jsonl else None

    result_doc_id = pipe.stage3_ingest(
        file_path=file_path,
        doc_id=args.doc_id,
        parsed_output_dir=parsed_output_dir,
        extracted_text=extracted_text,
        bigmodel_images=[],
        jsonl_path=jsonl_path,
        force=args.force,
    )
    print(f"done | doc_id={result_doc_id}")


if __name__ == "__main__":
    main()
