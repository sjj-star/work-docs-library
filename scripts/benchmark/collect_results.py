#!/usr/bin/env python3
"""收集三路解析结果，生成汇总 JSON."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DOCS = {
    "tms320f28335": "/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/tms320f28335.pdf",
    "amba_chi": "/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/IHI0050G_amba_chi_architecture_spec.pdf",
    "dc_ug": "/mnt/c/Users/SJJ22/Downloads/Doc/EDA Doc/Design Compiler User Guide.pdf",
    "sprui07": "/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/sprui07.pdf",
}


def collect_doc_results(doc_name: str, base_dir: Path) -> dict:
    """收集单个文档的三路结果."""
    doc_dir = base_dir / doc_name
    result = {"doc_name": doc_name, "parsers": {}}

    for parser in ["baseline", "pymupdf4llm", "mineru"]:
        parser_dir = doc_dir / parser
        md_path = parser_dir / "result.md"
        perf_path = doc_dir / f"{parser}_result.json"

        parser_result = {"status": "not_run"}

        if md_path.exists():
            text = md_path.read_text(encoding="utf-8")
            parser_result["status"] = "success"
            parser_result["output_chars"] = len(text)
            parser_result["output_lines"] = text.count("\n")
            parser_result["output_words"] = len(text.split())
            # 统计 Markdown 结构
            parser_result["heading_count"] = text.count("\n#") + (1 if text.startswith("#") else 0)
            parser_result["table_rows"] = text.count("\n|")
            parser_result["image_refs"] = text.count("![")
            parser_result["md_path"] = str(md_path)

        if perf_path.exists():
            try:
                with open(perf_path) as f:
                    content = f.read()
                    if content.strip():
                        perf = json.loads(content)
                        parser_result.update(perf)
            except Exception:
                pass

        result["parsers"][parser] = parser_result

    return result


def main():
    base_dir = Path("/tmp/workdocs_benchmark/outputs")
    all_results = {}

    for name in DOCS:
        all_results[name] = collect_doc_results(name, base_dir)

    out_path = base_dir / "collected_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果已收集到 {out_path}")

    # 打印摘要
    for name, res in all_results.items():
        print(f"\n=== {name} ===")
        for parser, data in res["parsers"].items():
            status = data.get("status", "?")
            chars = data.get("output_chars", 0)
            elapsed = data.get("elapsed_sec", "?")
            mem = data.get("peak_memory_mb", "?")
            if status == "success":
                print(f"  {parser}: {chars} chars, {elapsed}s, {mem}MB ✅")
            elif status == "failed":
                print(f"  {parser}: FAILED ({data.get('error', 'unknown')}) ❌")
            else:
                print(f"  {parser}: {status}")


if __name__ == "__main__":
    main()
