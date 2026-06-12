"""AMBA 风格无边框表格提取器.

仅依赖横线（通常是零高度水平线）和文字 x 坐标对齐来重建表格，
用于处理 find_tables() 无法检测的 "lines_strict/lines/text" 都失效的无竖线表格。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import fitz

from parsers.table_utils import normalize_markdown_table


@dataclass
class _Column:
    x0: float
    x1: float


class BorderlessTableExtractor:
    """从仅有水平线的区域中提取 Markdown 表格."""

    # 水平线筛选
    LINE_MIN_WIDTH_RATIO = 0.20
    LINE_MAX_HEIGHT_PT = 2.0
    LINE_WIDTH_HEIGHT_RATIO = 3.0
    LINE_DEDUP_TOLERANCE_PT = 2.0

    # 表格规模下限
    MIN_ROWS = 2
    MIN_COLS = 2

    # header 行同一单元格内单词聚类
    HEADER_WORD_GAP_PT = 12.0
    HEADER_BASELINE_TOLERANCE_PT = 3.0

    def extract(self, page: fitz.Page, search_rect: fitz.Rect) -> list[tuple[str, fitz.Rect]]:
        """在 search_rect 范围内查找并提取无边框表格.

        Args:
            page: PyMuPDF 页面对象。
            search_rect: 搜索区域（通常从 table caption 下方到页脚/下一硬分割）。

        Returns:
            列表，每个元素为 (markdown_table, bbox)。目前一页内通常返回 0 或 1 个结果。
        """
        page_rect = page.rect
        line_ys = self._collect_horizontal_line_ys(page, search_rect, page_rect)
        if len(line_ys) < self.MIN_ROWS + 1:
            return []

        # 先用线范围获取文字，再裁剪掉页脚/装饰线产生的空尾行
        draft_bbox = fitz.Rect(search_rect.x0, line_ys[0], search_rect.x1, line_ys[-1])
        words = self._get_words_in_bbox(page, draft_bbox)
        if not words:
            return []

        line_ys = self._prune_empty_boundary_lines(line_ys, words)
        if len(line_ys) < self.MIN_ROWS + 1:
            return []

        table_bbox = self._build_table_bbox(page, line_ys, search_rect, page_rect)
        rows = self._build_rows(line_ys)
        columns = self._detect_columns_from_header(words, rows[0])
        if len(columns) < self.MIN_COLS:
            return []

        md_table = self._build_markdown_table(words, rows, columns)
        if not md_table:
            return []

        return [(normalize_markdown_table(md_table), table_bbox)]

    def _collect_horizontal_line_ys(
        self, page: fitz.Page, search_rect: fitz.Rect, page_rect: fitz.Rect
    ) -> list[float]:
        """收集搜索范围内的水平线 y 坐标（去重后排序）."""
        min_width = page_rect.width * self.LINE_MIN_WIDTH_RATIO
        ys: list[float] = []
        for d in page.get_drawings():
            rect = fitz.Rect(d.get("rect", [0, 0, 0, 0]))
            if rect.y1 < search_rect.y0 or rect.y0 > search_rect.y1:
                continue
            if rect.width < min_width:
                continue
            if rect.height > self.LINE_MAX_HEIGHT_PT:
                continue
            if rect.width < rect.height * self.LINE_WIDTH_HEIGHT_RATIO:
                continue
            ys.append((rect.y0 + rect.y1) / 2)

        if not ys:
            return []
        ys.sort()

        # 去重：容差内合并
        deduped: list[float] = []
        current_sum = ys[0]
        current_count = 1
        for y in ys[1:]:
            if abs(y - current_sum / current_count) <= self.LINE_DEDUP_TOLERANCE_PT:
                current_sum += y
                current_count += 1
            else:
                deduped.append(current_sum / current_count)
                current_sum = y
                current_count = 1
        deduped.append(current_sum / current_count)
        return deduped

    def _build_table_bbox(
        self,
        page: fitz.Page,
        line_ys: list[float],
        search_rect: fitz.Rect,
        page_rect: fitz.Rect,
    ) -> fitz.Rect:
        """根据水平线范围构建表格 bbox."""
        # 使用 page 中的实际横线长度；若拿不到，则回退到页面主体宽度
        x0, x1 = page_rect.x1, page_rect.x0
        for d in page.get_drawings():
            rect = fitz.Rect(d.get("rect", [0, 0, 0, 0]))
            cy = (rect.y0 + rect.y1) / 2
            if any(abs(cy - ly) <= self.LINE_DEDUP_TOLERANCE_PT for ly in line_ys):
                x0 = min(x0, rect.x0)
                x1 = max(x1, rect.x1)
        if x0 >= x1:
            x0, x1 = search_rect.x0, search_rect.x1
        return fitz.Rect(x0, line_ys[0], x1, line_ys[-1])

    def _get_words_in_bbox(self, page: fitz.Page, bbox: fitz.Rect) -> list[tuple]:
        """获取 bbox 内所有 word（PyMuPDF words 格式）."""
        return cast(list[tuple], page.get_text("words", clip=bbox))

    @staticmethod
    def _has_words_between(words: list[tuple], y0: float, y1: float) -> bool:
        return any(y0 < (w[1] + w[3]) / 2 < y1 for w in words)

    def _prune_empty_boundary_lines(self, line_ys: list[float], words: list[tuple]) -> list[float]:
        """去掉表格上下边界没有文字的空行区间对应的线."""
        while len(line_ys) > 2 and not self._has_words_between(words, line_ys[0], line_ys[1]):
            line_ys = line_ys[1:]
        while len(line_ys) > 2 and not self._has_words_between(words, line_ys[-2], line_ys[-1]):
            line_ys = line_ys[:-1]
        return line_ys

    @staticmethod
    def _build_rows(line_ys: list[float]) -> list[tuple[float, float]]:
        """由相邻水平线生成行区间."""
        return [(line_ys[i], line_ys[i + 1]) for i in range(len(line_ys) - 1)]

    def _detect_columns_from_header(
        self, words: list[tuple], header_row: tuple[float, float]
    ) -> list[_Column]:
        """从 header 行单词的 x 位置聚类出列."""
        y0, y1 = header_row
        header_words = [w for w in words if y0 < (w[1] + w[3]) / 2 < y1]
        if not header_words:
            return []

        # 按从左到右、从上到下排序
        header_words.sort(key=lambda w: (w[1], w[0]))

        clusters: list[list[tuple]] = []
        for w in header_words:
            if not clusters:
                clusters.append([w])
                continue
            last = clusters[-1][-1]
            gap = w[0] - last[2]
            baseline_diff = abs(((w[1] + w[3]) / 2) - ((last[1] + last[3]) / 2))
            same_cell = (
                gap <= self.HEADER_WORD_GAP_PT
                and baseline_diff <= self.HEADER_BASELINE_TOLERANCE_PT
            )
            if same_cell:
                clusters[-1].append(w)
            else:
                clusters.append([w])

        columns = []
        for cluster in clusters:
            x0 = min(w[0] for w in cluster)
            x1 = max(w[2] for w in cluster)
            columns.append(_Column(x0, x1))
        return columns

    def _build_markdown_table(
        self,
        words: list[tuple],
        rows: list[tuple[float, float]],
        columns: list[_Column],
    ) -> str:
        """根据行区间和列中心重建 Markdown 表格."""
        # 用 header 单元格的几何中心作为列代表位置
        col_centers = [(col.x0 + col.x1) / 2 for col in columns]

        md_rows: list[list[str]] = []
        for y0, y1 in rows:
            row_words = [w for w in words if y0 < (w[1] + w[3]) / 2 < y1]
            cells: list[str] = []
            for col_idx, _ in enumerate(columns):
                # 取 word 中心与该列中心最近的单词
                cell_words: list[tuple] = []
                for w in row_words:
                    word_center = (w[0] + w[2]) / 2
                    distances = [abs(word_center - c) for c in col_centers]
                    if min(distances) == distances[col_idx]:
                        cell_words.append(w)
                # 按阅读顺序（从上到下、从左到右）排序后合并
                cell_words.sort(key=lambda w: (w[1], w[0]))
                cell_text = " ".join(w[4] for w in cell_words).strip()
                cells.append(cell_text)
            md_rows.append(cells)

        if not md_rows:
            return ""

        lines: list[str] = []
        lines.append("| " + " | ".join(md_rows[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in md_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
