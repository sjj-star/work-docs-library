#!/usr/bin/env python3
"""Benchmark 回归对比脚本：对比改进前后的 Baseline 指标."""

import json
import re
from pathlib import Path


def analyze_md(md_path: Path) -> dict:
    """分析 Markdown 文件的关键指标."""
    text = md_path.read_text(encoding="utf-8")
    return {
        "chars": len(text),
        "lines": text.count("\n"),
        "headings": text.count("\n#") + (1 if text.startswith("#") else 0),
        "heading_l1": text.count("\n# ") + (1 if text.startswith("# ") else 0),
        "heading_l2": text.count("\n## ") + (1 if text.startswith("## ") else 0),
        "heading_l3": text.count("\n### ") + (1 if text.startswith("### ") else 0),
        "table_rows": text.count("\n|"),
        # 统计完整的 Markdown 表格块（避免跨行重复计数）
        "table_blocks": len(
            re.findall(
                r"(?:^|\n)(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)",
                text,
            )
        ),
        "image_refs": text.count("!["),
        "bullet_lists": len(re.findall(r"^\s*[-*]\s+", text, re.MULTILINE)),
        "numbered_lists": len(re.findall(r"^\s*\d+\.\s+", text, re.MULTILINE)),
    }


def load_result_json(result_path: Path) -> dict:
    """加载 run_baseline.py 输出的 result.json."""
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "status" in data:
                return data
        except Exception:
            pass
    return {}


def compare_doc(doc_name: str, base_dir: Path) -> dict:
    """对比单个文档改进前后的指标."""
    out_dir = base_dir / "outputs" / doc_name
    before_dir = out_dir / "baseline_before"
    after_dir = out_dir / "baseline"

    before_md = before_dir / "result.md"
    after_md = after_dir / "result.md"

    before_metrics = analyze_md(before_md) if before_md.exists() else {}
    after_metrics = analyze_md(after_md) if after_md.exists() else {}

    before_result = load_result_json(out_dir / "baseline_result.json")
    after_result = load_result_json(out_dir / "baseline_regression.json")

    return {
        "doc_name": doc_name,
        "before": {"metrics": before_metrics, "result": before_result},
        "after": {"metrics": after_metrics, "result": after_result},
    }


def print_comparison(report: dict) -> None:
    """打印对比表格."""
    doc = report["doc_name"]
    b = report["before"]["metrics"]
    a = report["after"]["metrics"]
    br = report["before"].get("result", {})
    ar = report["after"].get("result", {})

    if not b or not a:
        print(f"\n=== {doc} === 数据缺失，跳过")
        return

    def diff(k):
        bv = b.get(k, 0)
        av = a.get(k, 0)
        delta = av - bv
        pct = f"{delta / bv * 100:+.1f}%" if bv else "N/A"
        return bv, av, delta, pct

    print(f"\n=== {doc} ===")
    print(f"{'指标':<20} {'改进前':>12} {'改进后':>12} {'变化':>10} {'变化率':>10}")
    print("-" * 70)

    for key, label in [
        ("chars", "字符数"),
        ("lines", "行数"),
        ("headings", "标题总数"),
        ("heading_l1", "  L1 标题"),
        ("heading_l2", "  L2 标题"),
        ("heading_l3", "  L3 标题"),
        ("table_rows", "表格行数"),
        ("table_blocks", "表格块数"),
        ("image_refs", "图片引用"),
        ("bullet_lists", "列表项"),
        ("numbered_lists", "编号列表"),
    ]:
        bv, av, delta, pct = diff(key)
        print(f"{label:<20} {bv:>12,} {av:>12,} {delta:>+10,} {pct:>10}")

    # 性能指标
    if br and ar:
        print("-" * 70)
        bt = br.get("elapsed_sec", 0)
        at = ar.get("elapsed_sec", 0)
        td = at - bt
        tp = f"{td / bt * 100:+.1f}%" if bt else "N/A"
        print(f"{'解析时间(s)':<20} {bt:>12.1f} {at:>12.1f} {td:>+10.1f} {tp:>10}")

        bm = br.get("peak_memory_mb", 0)
        am = ar.get("peak_memory_mb", 0)
        md = am - bm
        mp = f"{md / bm * 100:+.1f}%" if bm else "N/A"
        print(f"{'峰值内存(MB)':<20} {bm:>12.1f} {am:>12.1f} {md:>+10.1f} {mp:>10}")


def main() -> None:
    """主入口."""
    base_dir = Path("/tmp/workdocs_benchmark")
    docs = ["tms320f28335", "amba_chi", "dc_ug", "sprui07"]

    print("=" * 70)
    print("PDF Parser Benchmark 回归验证报告")
    print("改进项: Layer1 表格检测 + Layer2 图片提取 + Layer3 P4L fallback")
    print("=" * 70)

    all_reports = []
    for doc_name in docs:
        report = compare_doc(doc_name, base_dir)
        all_reports.append(report)
        print_comparison(report)

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    total_before_chars = sum(r["before"]["metrics"].get("chars", 0) for r in all_reports)
    total_after_chars = sum(r["after"]["metrics"].get("chars", 0) for r in all_reports)
    total_before_tables = sum(r["before"]["metrics"].get("table_blocks", 0) for r in all_reports)
    total_after_tables = sum(r["after"]["metrics"].get("table_blocks", 0) for r in all_reports)
    total_before_images = sum(r["before"]["metrics"].get("image_refs", 0) for r in all_reports)
    total_after_images = sum(r["after"]["metrics"].get("image_refs", 0) for r in all_reports)

    print(f"{'指标':<20} {'改进前':>12} {'改进后':>12} {'变化':>10}")
    print("-" * 60)
    tc_delta = total_after_chars - total_before_chars
    tt_delta = total_after_tables - total_before_tables
    ti_delta = total_after_images - total_before_images
    print(f"{'总字符数':<20} {total_before_chars:>12,} {total_after_chars:>12,} {tc_delta:>+10,}")
    print(f"{'总表格块数':<20} {total_before_tables:>12,} {total_after_tables:>12,} "
          f"{tt_delta:>+10,}")
    print(f"{'总图片引用':<20} {total_before_images:>12,} {total_after_images:>12,} "
          f"{ti_delta:>+10,}")

    # 保存 JSON 报告
    report_path = base_dir / "reports" / "regression_report.json"
    report_path.write_text(
        json.dumps(all_reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()
