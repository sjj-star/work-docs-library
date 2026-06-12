#!/usr/bin/env python3
"""子 Agent 分析入口脚本：读取三路输出，生成分维度评估报告."""

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def load_ground_truth(doc_name: str, base_dir: Path) -> dict:
    gt_path = base_dir / "ground_truth" / f"{doc_name}_gt.json"
    if gt_path.exists():
        return json.loads(gt_path.read_text(encoding="utf-8"))
    return {}


def load_parser_output(doc_name: str, parser: str, base_dir: Path) -> dict:
    parser_dir = base_dir / doc_name / parser
    md_path = parser_dir / "result.md"
    result = {"status": "not_found", "text": "", "metrics": {}}

    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        result["status"] = "success"
        result["text"] = text
        result["metrics"] = {
            "chars": len(text),
            "lines": text.count("\n"),
            "headings": text.count("\n#") + (1 if text.startswith("#") else 0),
            "heading_l1": text.count("\n# ") + (1 if text.startswith("# ") else 0),
            "heading_l2": text.count("\n## ") + (1 if text.startswith("## ") else 0),
            "heading_l3": text.count("\n### ") + (1 if text.startswith("### ") else 0),
            "table_rows": text.count("\n|"),
            "image_refs": text.count("!["),
            "bullet_lists": len(re.findall(r"^\s*[-*]\s+", text, re.MULTILINE)),
            "numbered_lists": len(re.findall(r"^\s*\d+\.\s+", text, re.MULTILINE)),
            "bold_texts": len(re.findall(r"\*\*[^*]+\*\*", text)),
        }

    return result


def analyze_doc(doc_name: str, base_dir: Path) -> dict:
    """分析单个文档的三路输出."""
    gt = load_ground_truth(doc_name, base_dir)
    baseline = load_parser_output(doc_name, "baseline", base_dir)
    mineru = load_parser_output(doc_name, "mineru", base_dir)

    report = {
        "doc_name": doc_name,
        "ground_truth": {
            "pages": gt.get("pages", 0),
            "total_chars": gt.get("total_chars", 0),
            "total_images": gt.get("total_images", 0),
            "total_tables": gt.get("total_tables", 0),
            "toc_entries": gt.get("toc_entries", 0),
        },
        "parsers": {
            "baseline": baseline,
            "mineru": mineru,
        },
    }

    # 计算对比指标
    gt_chars = gt.get("total_chars", 1)
    for name, data in report["parsers"].items():
        if data["status"] == "success":
            m = data["metrics"]
            m["char_extraction_rate"] = round(m["chars"] / gt_chars * 100, 2)

    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_doc.py <doc_name>")
        sys.exit(1)

    doc_name = sys.argv[1]
    base_dir = Path("/tmp/workdocs_benchmark")
    report = analyze_doc(doc_name, base_dir)

    out_path = base_dir / doc_name / "analysis_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"分析报告已保存到 {out_path}")

    # 打印摘要
    print(f"\n=== {doc_name} 分析摘要 ===")
    gt = report["ground_truth"]
    print(f"Ground Truth: {gt['pages']}页, {gt['total_chars']}字符, {gt['total_tables']}表格")
    for name, data in report["parsers"].items():
        if data["status"] == "success":
            m = data["metrics"]
            print(f"  {name}: {m['chars']} chars ({m['char_extraction_rate']:.1f}%), "
                  f"{m['headings']} headings, {m['table_rows']} table rows, "
                  f"{m['image_refs']} images")
        else:
            print(f"  {name}: {data['status']}")


if __name__ == "__main__":
    main()
