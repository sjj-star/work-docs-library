"""多进程 find_tables() 详细性能分析脚本。

统计维度：
1. 每页处理时间分解（pre-filter / find_tables / to_markdown / overhead）
2. 预筛选命中率（caption-gated vs heuristic-triggered）
3. find_tables() 调用次数 vs 页面总数
4. 不同 worker 数（1/2/3/4/6/8）下的详细对比
5. CPU 核心数与加速比关系
6. 进程启动/通信开销
7. 慢页面识别（耗时分布直方图）
"""

import gc
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import Any, cast

import fitz

TI_DOC = Path("/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/tms320f28335.pdf")
AMBA_DOC = Path("/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/IHI0050G_amba_chi_architecture_spec.pdf")

TABLE_LIKELY_INDICATORS = [
    r"^\s*[-=]{3,}\s*$",
    r"\|\s*\w",
    r"\w+\s{3,}\w+\s{3,}\w+\s{3,}\w+",
]
TABLE_CAPTION_RE = r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2


def _get_page_text_blocks(page: fitz.Page) -> list[dict]:
    """简化的文本块提取（用于预筛选）。"""
    text_dict = cast(dict[str, Any], page.get_text("dict"))
    blocks = []
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        text = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
        if text:
            blocks.append({"text": text})
    return blocks


def _is_page_likely_has_tables(text_blocks: list[dict]) -> bool:
    """版本 A 预筛选逻辑。"""
    import re

    for block in text_blocks:
        text = block.get("text", "")
        for pattern in TABLE_LIKELY_INDICATORS:
            if re.search(pattern, text, re.MULTILINE):
                return True
    return False


def _find_tables_with_timing(
    file_path: str,
    page_nums: list[int],
) -> dict:
    """单进程：逐页检测表格并统计详细耗时。"""
    import re

    doc = fitz.open(file_path)
    page_stats = []
    total_tables = 0
    triggered_by_caption = 0
    triggered_by_heuristic = 0
    skipped_by_filter = 0

    for p in page_nums:
        page = doc.load_page(p)
        page_stat: dict[str, Any] = {"page": p + 1}

        t0 = time.perf_counter()
        text_blocks = _get_page_text_blocks(page)
        t_after_text = time.perf_counter()

        has_caption = any(re.match(TABLE_CAPTION_RE, b.get("text", "")) for b in text_blocks)
        likely = has_caption or _is_page_likely_has_tables(text_blocks)
        t_after_filter = time.perf_counter()

        page_stat["text_extract_ms"] = round((t_after_text - t0) * 1000, 2)
        page_stat["filter_ms"] = round((t_after_filter - t_after_text) * 1000, 2)

        if not likely:
            skipped_by_filter += 1
            page_stat["skipped"] = True
            page_stat["find_tables_ms"] = 0.0
            page_stat["to_markdown_ms"] = 0.0
            page_stat["tables"] = 0
            page_stats.append(page_stat)
            continue

        if has_caption:
            triggered_by_caption += 1
            page_stat["trigger"] = "caption"
        else:
            triggered_by_heuristic += 1
            page_stat["trigger"] = "heuristic"

        t0_ft = time.perf_counter()
        tabs = page.find_tables(strategy="lines_strict")
        t_after_ft = time.perf_counter()

        md_time = 0.0
        table_count = 0
        if tabs is not None:
            for tab in tabs.tables:
                if tab.row_count < TABLE_MIN_ROWS or tab.col_count < TABLE_MIN_COLS:
                    continue
                t0_md = time.perf_counter()
                md = tab.to_markdown(clean=False)
                md_time += time.perf_counter() - t0_md
                if md.strip():
                    table_count += 1

        page_stat["find_tables_ms"] = round((t_after_ft - t0_ft) * 1000, 2)
        page_stat["to_markdown_ms"] = round(md_time * 1000, 2)
        page_stat["tables"] = table_count
        total_tables += table_count
        page_stats.append(page_stat)

    doc.close()
    return {
        "page_stats": page_stats,
        "total_tables": total_tables,
        "triggered_by_caption": triggered_by_caption,
        "triggered_by_heuristic": triggered_by_heuristic,
        "skipped_by_filter": skipped_by_filter,
    }


def _worker_find_tables(args: tuple[str, list[int]]) -> dict:
    """多进程 worker。"""
    file_path, page_nums = args
    return _find_tables_with_timing(file_path, page_nums)


def _merge_worker_results(results: list[dict]) -> dict:
    """合并多进程 worker 的结果。"""
    merged = {
        "page_stats": [],
        "total_tables": 0,
        "triggered_by_caption": 0,
        "triggered_by_heuristic": 0,
        "skipped_by_filter": 0,
    }
    for r in results:
        merged["page_stats"].extend(r["page_stats"])
        merged["total_tables"] += r["total_tables"]
        merged["triggered_by_caption"] += r["triggered_by_caption"]
        merged["triggered_by_heuristic"] += r["triggered_by_heuristic"]
        merged["skipped_by_filter"] += r["skipped_by_filter"]
    merged["page_stats"].sort(key=lambda x: x["page"])
    return merged


def _run_mp_benchmark(file_path: str, total_pages: int, workers: int) -> dict:
    """运行指定 worker 数的多进程基准测试。"""
    chunk_size = total_pages // workers
    chunks = [
        list(range(i * chunk_size, min((i + 1) * chunk_size, total_pages))) for i in range(workers)
    ]
    # 将余数页分配给最后一个 chunk
    if total_pages % workers and chunks:
        last_end = chunks[-1][-1] + 1 if chunks[-1] else (workers - 1) * chunk_size
        chunks[-1].extend(range(last_end, total_pages))

    gc.collect()
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.map(_worker_find_tables, [(file_path, c) for c in chunks])
    total_time = time.perf_counter() - t0

    merged = _merge_worker_results(results)
    merged["workers"] = workers
    merged["total_time"] = round(total_time, 2)
    return merged


def _analyze_page_distribution(page_stats: list[dict]) -> dict:
    """分析每页耗时分布。"""
    all_ft = [p["find_tables_ms"] for p in page_stats if p["find_tables_ms"] > 0]
    all_md = [p["to_markdown_ms"] for p in page_stats if p["to_markdown_ms"] > 0]
    all_total = [
        p["text_extract_ms"] + p["filter_ms"] + p["find_tables_ms"] + p["to_markdown_ms"]
        for p in page_stats
    ]

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "sum": 0, "mean": 0, "median": 0, "p95": 0, "max": 0}
        values_sorted = sorted(values)
        n = len(values_sorted)
        p95_idx = int(n * 0.95)
        return {
            "count": n,
            "sum": round(sum(values_sorted), 2),
            "mean": round(statistics.mean(values_sorted), 2),
            "median": round(statistics.median(values_sorted), 2),
            "p95": round(values_sorted[min(p95_idx, n - 1)], 2),
            "max": round(values_sorted[-1], 2),
        }

    # 识别最慢的页面
    slowest = sorted(
        [
            {
                "page": p["page"],
                "total_ms": p["text_extract_ms"]
                + p["filter_ms"]
                + p["find_tables_ms"]
                + p["to_markdown_ms"],
                "find_tables_ms": p["find_tables_ms"],
                "tables": p["tables"],
            }
            for p in page_stats
        ],
        key=lambda x: x["total_ms"],
        reverse=True,
    )[:10]

    return {
        "find_tables_ms": _stats(all_ft),
        "to_markdown_ms": _stats(all_md),
        "per_page_total_ms": _stats(all_total),
        "slowest_pages": slowest,
    }


def _print_report(doc_name: str, total_pages: int, single_result: dict, mp_results: list[dict]):
    """打印详细报告。"""
    cpu_count = mp.cpu_count()
    print(f"\n{'=' * 70}")
    print(f"  详细性能报告: {doc_name}")
    print(f"  总页数: {total_pages} | CPU 核心数: {cpu_count}")
    print(f"{'=' * 70}")

    # 1. 预筛选统计
    sr = single_result
    triggered = sr["triggered_by_caption"] + sr["triggered_by_heuristic"]
    print("\n【1. 预筛选命中率】")
    print(f"  页面总数:           {total_pages}")
    pct_triggered = triggered / total_pages * 100
    pct_caption = sr["triggered_by_caption"] / total_pages * 100
    pct_heuristic = sr["triggered_by_heuristic"] / total_pages * 100
    pct_skipped = sr["skipped_by_filter"] / total_pages * 100
    print(f"  触发 find_tables:   {triggered} ({pct_triggered:.1f}%)")
    print(f"    - caption-gated:  {sr['triggered_by_caption']} ({pct_caption:.1f}%)")
    print(f"    - heuristic:      {sr['triggered_by_heuristic']} ({pct_heuristic:.1f}%)")
    print(f"  预筛选跳过:         {sr['skipped_by_filter']} ({pct_skipped:.1f}%)")
    print(f"  检测表格总数:       {sr['total_tables']}")

    # 2. 耗时分布
    dist = _analyze_page_distribution(sr["page_stats"])
    print("\n【2. 单进程逐页耗时分解 (ms)】")
    print(
        f"  {'阶段':<20} {'count':>8} {'sum':>10} {'mean':>8} {'median':>8} {'p95':>8} {'max':>8}"
    )
    print(f"  {'-' * 70}")
    for name, key in [
        ("find_tables", "find_tables_ms"),
        ("to_markdown", "to_markdown_ms"),
        ("per_page_total", "per_page_total_ms"),
    ]:
        s = dist[key]
        print(
            f"  {name:<20} {s['count']:>8} {s['sum']:>10.1f} {s['mean']:>8.1f} "
            f"{s['median']:>8.1f} {s['p95']:>8.1f} {s['max']:>8.1f}"
        )

    # 3. 最慢页面
    print("\n【3. 最慢 10 个页面】")
    print(f"  {'page':>6} {'total_ms':>10} {'find_tables_ms':>16} {'tables':>8}")
    for sp in dist["slowest_pages"]:
        print(
            f"  {sp['page']:>6} {sp['total_ms']:>10.1f} "
            f"{sp['find_tables_ms']:>16.1f} {sp['tables']:>8}"
        )

    # 4. 多进程对比
    print("\n【4. 多进程加速比对比】")
    single_time = single_result.get("total_time", 0)
    print(f"  {'workers':>8} {'total_time':>12} {'speedup':>10} {'efficiency':>12}")
    print(f"  {'-' * 50}")
    print(f"  {'1 (单进程)':>8} {single_time:>10.2f}s {'1.00x':>10} {'100.0%':>12}")
    for r in mp_results:
        workers = r["workers"]
        t = r["total_time"]
        speedup = single_time / t if t > 0 else 0
        efficiency = speedup / workers * 100
        print(f"  {workers:>8} {t:>10.2f}s {speedup:>9.2f}x {efficiency:>11.1f}%")

    # 5. 结果一致性校验
    print("\n【5. 多进程结果一致性】")
    for r in mp_results:
        match = (
            r["total_tables"] == sr["total_tables"]
            and r["triggered_by_caption"] == sr["triggered_by_caption"]
            and r["triggered_by_heuristic"] == sr["triggered_by_heuristic"]
        )
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"  workers={r['workers']:>2}: tables={r['total_tables']:>3} {status}")


def main():
    """主函数。"""
    cpu_count = mp.cpu_count()
    print(f"系统 CPU 核心数: {cpu_count}")

    for doc_path, doc_name in [(TI_DOC, "TI TMS320F28335"), (AMBA_DOC, "AMBA CHI")]:
        if not doc_path.exists():
            print(f"\n跳过 {doc_name}: 文件不存在")
            continue

        doc = fitz.open(str(doc_path))
        total_pages = len(doc)
        doc.close()

        all_pages = list(range(total_pages))

        # 单进程详细统计
        print(f"\n{'=' * 70}")
        print(f"  开始测试: {doc_name} ({total_pages} 页)")
        print(f"{'=' * 70}")

        gc.collect()
        t0 = time.perf_counter()
        single_result = _find_tables_with_timing(str(doc_path), all_pages)
        single_result["total_time"] = round(time.perf_counter() - t0, 2)

        # 多进程测试（2, 4, 6, 8 workers）
        mp_results = []
        for workers in [2, 4]:
            if workers > total_pages:
                continue
            mp_results.append(_run_mp_benchmark(str(doc_path), total_pages, workers))

        # 如果有更多核心，测试 6, 8
        for workers in [6, 8]:
            if workers <= cpu_count and workers <= total_pages:
                mp_results.append(_run_mp_benchmark(str(doc_path), total_pages, workers))

        _print_report(doc_name, total_pages, single_result, mp_results)

    print(f"\n{'=' * 70}")
    print("  测试完成")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
