"""AMBA 风格无边框表格提取器.

仅依赖横线（通常是零高度水平线）和文字 x 坐标对齐来重建表格，
用于处理 find_tables() 无法检测的 "lines_strict/lines/text" 都失效的无竖线表格。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

import fitz

from parsers.table_utils import normalize_markdown_table


@dataclass
class _Column:
    x0: float
    x1: float


@dataclass
class _Line:
    y: float
    x0: float
    x1: float
    full_width: bool = False


class BorderlessTableExtractor:
    """从仅有水平线的区域中提取 Markdown 表格."""

    # 水平线筛选
    LINE_MIN_WIDTH_RATIO = 0.20
    LINE_MAX_HEIGHT_PT = 2.0
    LINE_WIDTH_HEIGHT_RATIO = 3.0
    LINE_DEDUP_TOLERANCE_PT = 2.0

    # 全宽线判定：x0 与表格左边界接近
    FULL_WIDTH_X0_TOLERANCE_PT = 8.0

    # 同一页内多个表格的 x 聚类阈值
    # 需覆盖“全宽线 + 右侧短 ITEM 线”的组合（如 AMBA B1.2：308pt 与 405pt）
    TABLE_X_CLUSTER_GAP_PT = 150.0

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
        all_lines = self._collect_horizontal_lines(page, search_rect, page_rect)
        if len(all_lines) < self.MIN_ROWS + 1:
            return []

        # 按 x0 聚类，取 caption 下方第一个（最上方）线所属的簇作为目标表格
        all_lines.sort(key=lambda ln: ln.y)
        clusters = self._cluster_lines_by_x(all_lines)
        target_lines = self._select_table_cluster(clusters)
        if len(target_lines) < self.MIN_ROWS + 1:
            return []

        # 用搜索区域取文字，再裁剪掉页脚/装饰线产生的空尾行
        draft_bbox = fitz.Rect(
            min(ln.x0 for ln in target_lines),
            target_lines[0].y,
            max(ln.x1 for ln in target_lines),
            search_rect.y1,
        )
        words = self._get_words_in_bbox(page, draft_bbox)
        words = self._filter_excluded_words(page, words, search_rect)
        if not words:
            return []

        target_lines = self._prune_empty_boundary_lines(target_lines, words)
        if len(target_lines) < self.MIN_ROWS + 1:
            return []

        table_bbox = fitz.Rect(
            min(ln.x0 for ln in target_lines),
            target_lines[0].y,
            max(ln.x1 for ln in target_lines),
            target_lines[-1].y,
        )

        # 在目标表格簇内区分全宽分组线与右侧短 ITEM 线
        for ln in target_lines:
            ln.full_width = ln.x0 <= table_bbox.x0 + self.FULL_WIDTH_X0_TOLERANCE_PT

        columns = self._detect_columns_from_header(words, target_lines)
        if len(columns) < self.MIN_COLS:
            return []

        md_table = self._build_markdown_table(words, target_lines, columns, search_rect.y1)
        if not md_table:
            return []

        return [(normalize_markdown_table(md_table), table_bbox)]

    def _collect_horizontal_lines(
        self, page: fitz.Page, search_rect: fitz.Rect, page_rect: fitz.Rect
    ) -> list[_Line]:
        """收集搜索范围内的水平线（去重后排序）."""
        min_width = page_rect.width * self.LINE_MIN_WIDTH_RATIO
        raw: list[fitz.Rect] = []
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
            raw.append(rect)

        if not raw:
            return []

        raw.sort(key=lambda r: (r.y0 + r.y1) / 2)
        groups: list[list[fitz.Rect]] = []
        for r in raw:
            cy = (r.y0 + r.y1) / 2
            if (
                groups
                and abs(cy - sum((rr.y0 + rr.y1) / 2 for rr in groups[-1]) / len(groups[-1]))
                <= self.LINE_DEDUP_TOLERANCE_PT
            ):
                groups[-1].append(r)
            else:
                groups.append([r])

        lines: list[_Line] = []
        for g in groups:
            y = sum((r.y0 + r.y1) / 2 for r in g) / len(g)
            x0 = min(r.x0 for r in g)
            x1 = max(r.x1 for r in g)
            lines.append(_Line(y=y, x0=x0, x1=x1))
        return lines

    def _cluster_lines_by_x(self, lines: list[_Line]) -> list[list[_Line]]:
        """按 x0 位置把线分成不同表格簇."""
        sorted_lines = sorted(lines, key=lambda ln: ln.x0)
        clusters: list[list[_Line]] = []
        for ln in sorted_lines:
            if not clusters:
                clusters.append([ln])
                continue
            last = clusters[-1][-1]
            if abs(ln.x0 - last.x0) <= self.TABLE_X_CLUSTER_GAP_PT:
                clusters[-1].append(ln)
            else:
                clusters.append([ln])
        return clusters

    def _select_table_cluster(self, clusters: list[list[_Line]]) -> list[_Line]:
        """选择最上方线所在的簇作为当前表格."""
        if not clusters:
            return []
        # 所有线按 y 排序后取第一条线，返回它所在的簇
        candidates = [(ln.y, idx) for idx, cluster in enumerate(clusters) for ln in cluster]
        candidates.sort()
        target_idx = candidates[0][1]
        return sorted(clusters[target_idx], key=lambda ln: ln.y)

    def _get_words_in_bbox(self, page: fitz.Page, bbox: fitz.Rect) -> list[tuple]:
        """获取 bbox 内所有 word（PyMuPDF words 格式）."""
        return cast(list[tuple], page.get_text("words", clip=bbox))

    # 需要排除的跨页标记与相邻 caption 正则
    _CONTINUATION_RE = re.compile(
        r"continued\s+(?:on\s+next\s+page|from\s+previous\s+page)",
        re.IGNORECASE,
    )
    _CAPTION_RE = re.compile(
        r"^(Table|表|Figure)\s*[A-Z]?\d+(?:[-\.]\d+)?(?:[:]\s*\S|\s+[A-Z]\S*)",
        re.IGNORECASE,
    )

    def _filter_excluded_words(
        self, page: fitz.Page, words: list[tuple], search_rect: fitz.Rect
    ) -> list[tuple]:
        """过滤掉跨页标记与相邻 caption 文字，避免进入单元格."""
        exclude_bboxes: list[fitz.Rect] = []
        for block in page.get_text("blocks", clip=search_rect):
            txt = block[4]
            if self._CONTINUATION_RE.search(txt) or self._CAPTION_RE.match(txt.strip()):
                exclude_bboxes.append(fitz.Rect(block[:4]))
        if not exclude_bboxes:
            return words
        return [w for w in words if not any(fitz.Rect(w[:4]).intersects(b) for b in exclude_bboxes)]

    @staticmethod
    def _has_words_between(words: list[tuple], y0: float, y1: float) -> bool:
        return any(y0 < (w[1] + w[3]) / 2 < y1 for w in words)

    def _prune_empty_boundary_lines(self, lines: list[_Line], words: list[tuple]) -> list[_Line]:
        """去掉表格上下边界没有文字的空行区间对应的线."""
        while len(lines) > 2 and not self._has_words_between(words, lines[0].y, lines[1].y):
            lines = lines[1:]
        while len(lines) > 2 and not self._has_words_between(words, lines[-2].y, lines[-1].y):
            lines = lines[:-1]
        return lines

    def _detect_columns_from_header(self, words: list[tuple], lines: list[_Line]) -> list[_Column]:
        """从 header 行单词的 x 位置聚类出列.

        header 行位于最上方全宽线与紧随其后的下一条全宽线之间。
        """
        # 此时 full_width 尚未设置，用 x0 接近表格左边界的线作为全宽线
        table_left = min(ln.x0 for ln in lines)
        full_ys = [ln.y for ln in lines if ln.x0 <= table_left + self.FULL_WIDTH_X0_TOLERANCE_PT]
        full_ys.sort()

        y0 = lines[0].y
        if len(full_ys) >= 2:
            y1 = full_ys[1]
        else:
            y1 = y0 + 30.0

        header_words = [w for w in words if y0 < (w[1] + w[3]) / 2 < y1]
        if not header_words:
            return []

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
        lines: list[_Line],
        columns: list[_Column],
        search_bottom: float,
    ) -> str:
        """根据全宽分组线和右侧短 ITEM 线重建 Markdown 表格."""
        full_ys = [ln.y for ln in lines if ln.full_width]
        if len(full_ys) < 2:
            # 没有明显的全宽分组线：退化为每根线作为一个行边界
            rows = [(lines[i].y, lines[i + 1].y) for i in range(len(lines) - 1)]
            return self._render_rows(words, rows, columns)

        # 分组：相邻全宽线之间为一个 group；最后一个 group 可能延伸到搜索底部
        groups: list[tuple[float, float, list[float]]] = []
        for i in range(len(full_ys) - 1):
            y0, y1 = full_ys[i], full_ys[i + 1]
            partials = [ln.y for ln in lines if not ln.full_width and y0 < ln.y < y1]
            groups.append((y0, y1, sorted(partials)))
        # 最后一个 group：从最后一条全宽线到 search_bottom
        last_y = full_ys[-1]
        partials = [ln.y for ln in lines if not ln.full_width and ln.y > last_y]
        if partials or self._has_words_between(words, last_y, search_bottom):
            groups.append((last_y, search_bottom, sorted(partials)))

        if not groups:
            return ""

        # 第一组是 header 行
        header_texts = self._cell_texts(words, groups[0][0], groups[0][1], columns)
        if not any(header_texts):
            return ""

        md_rows: list[list[str]] = [header_texts]
        for y0, y1, partials in groups[1:]:
            classification = self._column_text_in_group(words, y0, y1, columns[0])
            # 如果该 group 没有右侧 ITEM 线，则按普通列提取整行
            if not partials:
                row_cells = self._cell_texts(words, y0, y1, columns)
                if classification and not row_cells[0]:
                    row_cells[0] = classification
                if any(row_cells):
                    md_rows.append(row_cells)
                continue
            # 由 partial 线切分 item 行
            item_bounds = [y0] + partials + [y1]
            for k in range(len(item_bounds) - 1):
                item_text = self._column_text_in_group(
                    words, item_bounds[k], item_bounds[k + 1], columns[-1]
                )
                if classification or item_text:
                    md_rows.append([classification, item_text])

        return self._rows_to_markdown(md_rows)

    def _render_rows(
        self,
        words: list[tuple],
        rows: list[tuple[float, float]],
        columns: list[_Column],
    ) -> str:
        """无全宽分组线时的退化渲染."""
        md_rows = []
        for y0, y1 in rows:
            md_rows.append(self._cell_texts(words, y0, y1, columns))
        return self._rows_to_markdown(md_rows)

    def _cell_texts(
        self,
        words: list[tuple],
        y0: float,
        y1: float,
        columns: list[_Column],
    ) -> list[str]:
        """按列中心分配一个行区间内的文字."""
        col_centers = [(col.x0 + col.x1) / 2 for col in columns]
        row_words = [w for w in words if y0 < (w[1] + w[3]) / 2 < y1]
        cells = []
        for col_idx, _ in enumerate(columns):
            cell_words = []
            for w in row_words:
                word_center = (w[0] + w[2]) / 2
                distances = [abs(word_center - c) for c in col_centers]
                if min(distances) == distances[col_idx]:
                    cell_words.append(w)
            cell_words.sort(key=lambda w: (w[1], w[0]))
            cells.append(" ".join(w[4] for w in cell_words).strip())
        return cells

    def _column_text_in_group(
        self, words: list[tuple], y0: float, y1: float, column: _Column
    ) -> str:
        """提取某一列在 group 区间内的全部文字（用于分类列跨多 ITEM 行的情况）."""
        col_words = [
            w
            for w in words
            if y0 < (w[1] + w[3]) / 2 < y1
            and abs((w[0] + w[2]) / 2 - (column.x0 + column.x1) / 2)
            <= (column.x1 - column.x0) / 2 + 20
        ]
        col_words.sort(key=lambda w: (w[1], w[0]))
        return " ".join(w[4] for w in col_words).strip()

    @staticmethod
    def _rows_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        # 过滤空数据行，保留 header 与 separator
        filtered = [rows[0]]
        filtered.append(["---" for _ in rows[0]])
        for row in rows[1:]:
            non_empty = [cell for cell in row if cell.strip()]
            # 只保留至少有两个非空单元格，或仅有非空 item 单元格（续页）的行
            if len(non_empty) >= 2 or (len(non_empty) == 1 and not row[0].strip()):
                filtered.append(row)
        lines = ["| " + " | ".join(r) + " |" for r in filtered]
        return "\n".join(lines)
