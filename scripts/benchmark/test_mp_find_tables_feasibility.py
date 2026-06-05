"""多进程 find_tables() 可行性验证脚本。

测试目标：
1. fitz.Rect 是否可 pickle（跨进程传递）
2. 多进程独立 fitz.open() + find_tables() 稳定性
3. 单进程 vs 多进程结果一致性
4. 实际加速比测量
5. 内存占用评估

约束：
- fitz.Page 非线程安全 → 多线程不可行
- 多进程方案：每个进程独立 open 文档
"""

import gc
import multiprocessing as mp
import pickle
import time
import tracemalloc
from pathlib import Path
from typing import Any, cast

import fitz
import pytest

# 测试文档路径（需在环境中存在）
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


def _is_page_likely_has_tables(text_blocks: list[dict]) -> bool:
    """版本 A 预筛选逻辑。"""
    import re

    for block in text_blocks:
        text = block.get("text", "")
        for pattern in TABLE_LIKELY_INDICATORS:
            if re.search(pattern, text, re.MULTILINE):
                return True
    return False


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


def _find_tables_single(
    file_path: str,
    page_nums: list[int],
) -> dict[int, list[dict]]:
    """单进程：在指定页面上检测表格。"""
    doc = fitz.open(file_path)
    results: dict[int, list[dict]] = {}
    for p in page_nums:
        page = doc.load_page(p)
        text_blocks = _get_page_text_blocks(page)
        has_caption = any(
            __import__("re").match(TABLE_CAPTION_RE, b.get("text", "")) for b in text_blocks
        )
        likely = has_caption or _is_page_likely_has_tables(text_blocks)
        if not likely:
            continue
        try:
            tabs = page.find_tables(strategy="lines_strict")
            if tabs is None:
                continue
            tables = []
            for tab in tabs.tables:
                if tab.row_count < TABLE_MIN_ROWS or tab.col_count < TABLE_MIN_COLS:
                    continue
                md = tab.to_markdown(clean=False)
                if md.strip():
                    tables.append(
                        {
                            "bbox": tuple(tab.bbox),
                            "rows": tab.row_count,
                            "cols": tab.col_count,
                            "md": md.strip(),
                        }
                    )
            if tables:
                results[p] = tables
        except Exception as e:
            results[p] = [{"error": str(e)}]
    doc.close()
    return results


def _worker_find_tables(args: tuple[str, list[int]]) -> dict[int, list[dict]]:
    """多进程 worker：独立打开文档并检测表格。"""
    file_path, page_nums = args
    return _find_tables_single(file_path, page_nums)


class TestPickleSerialization:
    """测试 1: fitz.Rect 跨进程序列化可行性。"""

    def test_rect_pickle(self):
        """fitz.Rect 必须可 pickle，否则无法跨进程传递 bbox。"""
        rect = fitz.Rect(10, 20, 100, 200)
        dumped = pickle.dumps(rect)
        loaded = pickle.loads(dumped)
        assert loaded == rect
        assert loaded.x0 == 10.0

    def test_rect_tuple_roundtrip(self):
        """Bbox 用 tuple 传递是更安全的备选方案。"""
        rect = fitz.Rect(10, 20, 100, 200)
        as_tuple = (rect.x0, rect.y0, rect.x1, rect.y1)
        restored = fitz.Rect(as_tuple)
        assert restored == rect

    def test_table_result_pickle(self):
        """表格检测结果必须可 pickle。"""
        result = {
            0: [
                {
                    "bbox": (10.0, 20.0, 100.0, 200.0),
                    "rows": 3,
                    "cols": 4,
                    "md": "| A | B |\n|---|---|\n| 1 | 2 |",
                }
            ]
        }
        dumped = pickle.dumps(result)
        loaded = pickle.loads(dumped)
        assert loaded == result


class TestResultConsistency:
    """测试 2: 单进程 vs 多进程结果一致性。"""

    @pytest.mark.skipif(not TI_DOC.exists(), reason="TI doc not found")
    def test_ti_single_vs_mp_consistency(self):
        """TI 文档：单进程和多进程检测结果必须一致。"""
        doc = fitz.open(str(TI_DOC))
        total = len(doc)
        doc.close()

        # 单进程
        single_result = _find_tables_single(str(TI_DOC), list(range(total)))

        # 多进程：2 workers
        mid = total // 2
        chunks = [list(range(0, mid)), list(range(mid, total))]
        with mp.Pool(2) as pool:
            mp_results = pool.map(_worker_find_tables, [(str(TI_DOC), c) for c in chunks])

        mp_merged: dict[int, list[dict]] = {}
        for r in mp_results:
            mp_merged.update(r)

        # 比较：页面数、每页表格数、每页表格 bbox
        assert set(single_result.keys()) == set(mp_merged.keys())
        for p in single_result:
            s_tables = single_result[p]
            m_tables = mp_merged[p]
            assert len(s_tables) == len(m_tables), f"Page {p + 1} table count mismatch"
            for i, (s, m) in enumerate(zip(s_tables, m_tables)):
                assert s["rows"] == m["rows"], f"Page {p + 1} table {i} rows mismatch"
                assert s["cols"] == m["cols"], f"Page {p + 1} table {i} cols mismatch"
                assert s["bbox"] == m["bbox"], f"Page {p + 1} table {i} bbox mismatch"
                assert s["md"] == m["md"], f"Page {p + 1} table {i} markdown mismatch"

    @pytest.mark.skipif(not AMBA_DOC.exists(), reason="AMBA doc not found")
    def test_amba_single_vs_mp_consistency(self):
        """AMBA 文档：单进程和多进程检测结果必须一致。"""
        doc = fitz.open(str(AMBA_DOC))
        total = len(doc)
        doc.close()

        single_result = _find_tables_single(str(AMBA_DOC), list(range(total)))

        # 4 workers
        chunk_size = total // 4
        chunks = [list(range(i * chunk_size, min((i + 1) * chunk_size, total))) for i in range(4)]
        with mp.Pool(4) as pool:
            mp_results = pool.map(_worker_find_tables, [(str(AMBA_DOC), c) for c in chunks])

        mp_merged: dict[int, list[dict]] = {}
        for r in mp_results:
            mp_merged.update(r)

        assert set(single_result.keys()) == set(mp_merged.keys())
        for p in single_result:
            s_tables = single_result[p]
            m_tables = mp_merged[p]
            assert len(s_tables) == len(m_tables), f"Page {p + 1} table count mismatch"


class TestStability:
    """测试 3: 多进程稳定性（重复运行不崩溃）。"""

    @pytest.mark.skipif(not TI_DOC.exists(), reason="TI doc not found")
    def test_ti_mp_repeated_runs(self):
        """TI 文档重复 3 次多进程运行，验证稳定性。"""
        doc = fitz.open(str(TI_DOC))
        total = len(doc)
        doc.close()

        for run in range(3):
            chunks = [list(range(0, total // 2)), list(range(total // 2, total))]
            with mp.Pool(2) as pool:
                results = pool.map(_worker_find_tables, [(str(TI_DOC), c) for c in chunks])
            assert all(isinstance(r, dict) for r in results)
            # 简单验证：至少有一些页面检测到表格
            total_pages_with_tables = sum(len(r) for r in results)
            assert total_pages_with_tables > 0, f"Run {run + 1}: no tables found"


class TestPerformance:
    """测试 4: 实际加速比测量。"""

    @pytest.mark.skipif(not TI_DOC.exists(), reason="TI doc not found")
    def test_ti_performance_speedup(self):
        """测量 TI 文档多进程加速比。"""
        doc = fitz.open(str(TI_DOC))
        total = len(doc)
        doc.close()

        all_pages = list(range(total))

        # 单进程 warm-up + 测量
        _find_tables_single(str(TI_DOC), all_pages[:5])  # warm-up
        gc.collect()
        t0 = time.perf_counter()
        _find_tables_single(str(TI_DOC), all_pages)
        t_single = time.perf_counter() - t0

        # 多进程：2 workers
        mid = total // 2
        chunks = [list(range(0, mid)), list(range(mid, total))]
        gc.collect()
        t0 = time.perf_counter()
        with mp.Pool(2) as pool:
            pool.map(_worker_find_tables, [(str(TI_DOC), c) for c in chunks])
        t_mp2 = time.perf_counter() - t0

        # 多进程：4 workers（如果页面足够）
        if total >= 4:
            chunk_size = total // 4
            chunks4 = [
                list(range(i * chunk_size, min((i + 1) * chunk_size, total))) for i in range(4)
            ]
            gc.collect()
            t0 = time.perf_counter()
            with mp.Pool(4) as pool:
                pool.map(_worker_find_tables, [(str(TI_DOC), c) for c in chunks4])
            t_mp4 = time.perf_counter() - t0
        else:
            t_mp4 = None

        sp2 = t_single / t_mp2 if t_mp2 > 0 else 0
        print(f"\n[TI] pages={total} single={t_single:.2f}s mp2={t_mp2:.2f}s speedup2={sp2:.2f}x")
        if t_mp4:
            sp4 = t_single / t_mp4 if t_mp4 > 0 else 0
            print(f"[TI] mp4={t_mp4:.2f}s speedup4={sp4:.2f}x")

        # 断言：多进程应更快（允许 10% 容差，因为进程启动开销）
        assert t_mp2 < t_single * 1.1, "2-worker mp should not be significantly slower"

    @pytest.mark.skipif(not AMBA_DOC.exists(), reason="AMBA doc not found")
    def test_amba_performance_speedup(self):
        """测量 AMBA 文档多进程加速比。"""
        doc = fitz.open(str(AMBA_DOC))
        total = len(doc)
        doc.close()

        all_pages = list(range(total))

        _find_tables_single(str(AMBA_DOC), all_pages[:5])
        gc.collect()
        t0 = time.perf_counter()
        _find_tables_single(str(AMBA_DOC), all_pages)
        t_single = time.perf_counter() - t0

        # 2 workers
        mid = total // 2
        chunks = [list(range(0, mid)), list(range(mid, total))]
        gc.collect()
        t0 = time.perf_counter()
        with mp.Pool(2) as pool:
            pool.map(_worker_find_tables, [(str(AMBA_DOC), c) for c in chunks])
        t_mp2 = time.perf_counter() - t0

        # 4 workers
        chunk_size = total // 4
        chunks4 = [list(range(i * chunk_size, min((i + 1) * chunk_size, total))) for i in range(4)]
        gc.collect()
        t0 = time.perf_counter()
        with mp.Pool(4) as pool:
            pool.map(_worker_find_tables, [(str(AMBA_DOC), c) for c in chunks4])
        t_mp4 = time.perf_counter() - t0

        sp2 = t_single / t_mp2 if t_mp2 > 0 else 0
        sp4 = t_single / t_mp4 if t_mp4 > 0 else 0
        print(f"\n[AMBA] pages={total} single={t_single:.2f}s mp2={t_mp2:.2f}s speedup2={sp2:.2f}x")
        print(f"[AMBA] mp4={t_mp4:.2f}s speedup4={sp4:.2f}x")

        assert t_mp4 < t_single * 1.1, "4-worker mp should not be significantly slower"


class TestMemory:
    """测试 5: 内存占用评估。"""

    @pytest.mark.skipif(not TI_DOC.exists(), reason="TI doc not found")
    def test_ti_memory_overhead(self):
        """评估多进程的内存开销。"""
        doc = fitz.open(str(TI_DOC))
        total = len(doc)
        doc.close()

        # 单进程内存基线
        gc.collect()
        tracemalloc.start()
        _find_tables_single(str(TI_DOC), list(range(total)))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_single = peak / 1024 / 1024  # MB

        # 多进程 2 workers（每个进程独立打开文档 → 约 2x 内存）
        gc.collect()
        mid = total // 2
        chunks = [list(range(0, mid)), list(range(mid, total))]
        tracemalloc.start()
        with mp.Pool(2) as pool:
            pool.map(_worker_find_tables, [(str(TI_DOC), c) for c in chunks])
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mp = peak / 1024 / 1024  # MB

        ratio = mem_mp / mem_single if mem_single > 0 else 0
        print(
            f"\n[TI Memory] single_peak={mem_single:.1f}MB "
            f"mp_peak={mem_mp:.1f}MB ratio={ratio:.2f}x"
        )

        # 多进程峰值内存不应超过单进程的 3 倍（2 workers + 主进程开销）
        assert mem_mp < mem_single * 3.0, "Memory overhead too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
