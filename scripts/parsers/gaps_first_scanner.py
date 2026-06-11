"""Caption-Driven Linear Extractor with Hard Separators.

核心设计：
1. 先确定页面的所有 y 轴硬分割位置（header/footer/heading/caption/body_text）
2. 硬分割之间形成 zone，任何搜索算法只在 zone 内执行
3. 不存在无 Figure Caption 的图 —— 无 Caption 的 drawing 只可能是表格或正文
4. 一个 Figure Caption 仅对应一个图片
"""

import io
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import fitz
from core.config import Config
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GapsPageResult:
    """单页处理结果."""

    images: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Zone:
    """硬分割之间的区间."""

    y0: float
    y1: float
    drawings: list[fitz.Rect] = field(default_factory=list)
    consumed: bool = False
    text_block_count: int = 0
    h_zero_height: int = 0  # Count of horizontal lines with height==0.0 (before _fix_drawing_rect)


class GapsFirstScanner:
    """Caption-driven linear extractor with hard separators."""

    FIGURE_CAPTION_RE = r"^Figure\s+[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
    TABLE_CAPTION_RE = r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
    CALLOUT_PREFIXES = ("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.")
    CALLOUT_NOTE_PREFIX = "Note:"
    CALLOUT_GAP_THRESHOLD = 45.0
    CALLOUT_SMALL_FONT_MAX = 9.0
    CALLOUT_SMALL_FONT_RATIO = 0.72
    DRAWING_RECT_MIN_SIZE = 0.1
    DRAWING_RECT_EPSILON = 0.5
    HEADER_MARGIN_PT = 50.0
    FOOTER_MARGIN_PT = 50.0
    CAPTION_OVERLAP_TOLERANCE = 5.0
    CLIP_PADDING_HORIZONTAL = 15.0
    CLIP_PADDING_BOTTOM = 15.0
    EDGE_LABEL_MARGIN = 12.0
    EDGE_LABEL_MARGIN_CALLOUT = 25.0
    BODY_TEXT_MAX_HEIGHT_RATIO = 0.35
    BODY_TEXT_MAX_WIDTH_RATIO = 0.85
    BODY_TEXT_MAX_HEIGHT = 80.0
    BODY_TEXT_WIDTH_RATIO = 0.52
    BODY_TEXT_MIN_RECT_HEIGHT = 8.0
    BODY_TEXT_MIN_LENGTH = 45
    TABLE_MIN_HEIGHT_PT = Config.PARSER_TABLE_MIN_HEIGHT_PT
    TABLE_MIN_WIDTH_RATIO = Config.PARSER_TABLE_MIN_WIDTH_RATIO
    TABLE_DETECTION_ENABLED = Config.PARSER_TABLE_DETECTION_ENABLED
    TABLE_MIN_ROWS = Config.PARSER_TABLE_MIN_ROWS
    TABLE_MIN_COLS = Config.PARSER_TABLE_MIN_COLS
    PAGE_RENDER_DPI = Config.PARSER_PAGE_RENDER_DPI
    MIN_IMAGE_WIDTH = Config.PARSER_MIN_IMAGE_WIDTH
    MIN_IMAGE_HEIGHT = Config.PARSER_MIN_IMAGE_HEIGHT
    FIGURE_MIN_SCORE = Config.PARSER_FIGURE_MIN_SCORE
    EDGE_LABEL_MAX_LEN = Config.PARSER_EDGE_LABEL_MAX_LEN
    CLUSTER_PROXIMITY = 10.0
    FULL_PAGE_SKIP_WIDTH_RATIO = 0.92
    FULL_PAGE_SKIP_HEIGHT_RATIO = 0.82
    HEADING_MIN_FONT = 12.0
    HEADING_WIDTH_RATIO = 0.3

    @classmethod
    def _is_likely_body_text(cls, txt: str, rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
        """Distinguish body paragraphs from diagram labels."""
        if rect.height > page_rect.height * cls.BODY_TEXT_MAX_HEIGHT_RATIO:
            return False
        if (
            rect.width > page_rect.width * cls.BODY_TEXT_MAX_WIDTH_RATIO
            and rect.height > cls.BODY_TEXT_MAX_HEIGHT
        ):
            return False
        width_ratio = rect.width / page_rect.width
        return (
            width_ratio > cls.BODY_TEXT_WIDTH_RATIO
            and rect.height > cls.BODY_TEXT_MIN_RECT_HEIGHT
            and len(txt) > cls.BODY_TEXT_MIN_LENGTH
        )

    @classmethod
    def _is_strict_figure_caption(cls, text: str) -> bool:
        """Exclude body-text reference sentences (e.g. 'Figure B2.2 shows...')."""
        if not re.match(cls.FIGURE_CAPTION_RE, text):
            return False
        ref_pattern = (
            r"^Figure\s*[A-Z]?\d+(?:[-\.]\d+)?\s+"
            r"(?:shows|describes|lists|illustrates|presents|gives|provides|details|summarizes)\b"
        )
        return not re.match(ref_pattern, text, re.IGNORECASE)

    @classmethod
    def _is_strict_table_caption(cls, text: str) -> bool:
        """Exclude body-text reference sentences (e.g. 'Table B1.3 shows...')."""
        if not re.match(cls.TABLE_CAPTION_RE, text):
            return False
        ref_pattern = (
            r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?\s+"
            r"(?:shows|describes|lists|illustrates|presents|gives|provides|details|summarizes)\b"
        )
        return not re.match(ref_pattern, text, re.IGNORECASE)

    @classmethod
    def _fix_drawing_rect(cls, rect: fitz.Rect) -> fitz.Rect:
        """Expand zero-width/height drawing rects so they can intersect."""
        eps = cls.DRAWING_RECT_EPSILON
        if rect.width < cls.DRAWING_RECT_MIN_SIZE and rect.height < cls.DRAWING_RECT_MIN_SIZE:
            return fitz.Rect(rect.x0 - eps, rect.y0 - eps, rect.x1 + eps, rect.y1 + eps)
        if rect.width < cls.DRAWING_RECT_MIN_SIZE:
            return fitz.Rect(rect.x0 - eps, rect.y0, rect.x1 + eps, rect.y1)
        if rect.height < cls.DRAWING_RECT_MIN_SIZE:
            return fitz.Rect(rect.x0, rect.y0 - eps, rect.x1, rect.y1 + eps)
        return rect

    @classmethod
    def _is_low_content_image(cls, pil_img: Image.Image) -> bool:
        """Detect nearly-blank or decorative images."""
        rgb = pil_img.convert("RGB")
        if rgb.width > 100 or rgb.height > 100:
            rgb.thumbnail((100, 100))
        pixels = list(rgb.get_flattened_data())
        total = len(pixels)
        if total == 0:
            return True
        from collections import Counter

        most_common_count = Counter(pixels).most_common(1)[0][1]
        return most_common_count / total > 0.95

    def _classify_text_blocks(
        self, text_blocks: list[tuple[str, fitz.Rect, float]], page_rect: fitz.Rect
    ) -> list[tuple[str, fitz.Rect, float, str]]:
        """Classify text blocks: caption, heading, callout, body_text, other."""
        n = len(text_blocks)
        categories: list[str | None] = [None] * n

        # First pass: captions (strict — exclude body-text reference sentences)
        caption_indices: list[int] = []
        for i, (txt, rect, _) in enumerate(text_blocks):
            if self._is_strict_figure_caption(txt):
                categories[i] = "figure_caption"
                caption_indices.append(i)
            elif self._is_strict_table_caption(txt):
                categories[i] = "table_caption"
                caption_indices.append(i)

        # Estimate body_size from non-caption blocks
        non_caption_sizes = [
            avg_size for i, (_, _, avg_size) in enumerate(text_blocks) if categories[i] is None
        ]
        estimated_body_size = (
            statistics.median(non_caption_sizes) if non_caption_sizes else 10.0
        )
        heading_threshold = max(estimated_body_size + 2, self.HEADING_MIN_FONT)

        # Second pass: heading
        for i, (txt, rect, avg_size) in enumerate(text_blocks):
            if categories[i] is not None:
                continue
            if (
                avg_size >= heading_threshold
                and rect.width > page_rect.width * self.HEADING_WIDTH_RATIO
            ):
                categories[i] = "heading"

        caption_ranges = []
        for idx in caption_indices:
            _, rect, _ = text_blocks[idx]
            caption_ranges.append((rect.y0 - 250, rect.y1 + 250))

        callout_pattern = re.compile(r"^[A-H]\.\s|^Note:\s*|^\([A-Z]\)")
        median_est = estimated_body_size

        # Third pass: callout seeds
        for i, (txt, rect, avg_size) in enumerate(text_blocks):
            if categories[i] is not None:
                continue
            near_caption = any(rect.y1 >= y0 and rect.y0 <= y1 for y0, y1 in caption_ranges)
            if not near_caption:
                continue
            has_prefix = (
                txt.startswith(self.CALLOUT_PREFIXES)
                or txt.startswith(self.CALLOUT_NOTE_PREFIX)
                or bool(callout_pattern.search(txt))
            )
            is_small = (
                avg_size <= self.CALLOUT_SMALL_FONT_MAX
                or avg_size <= self.CALLOUT_SMALL_FONT_RATIO * estimated_body_size + 2
            ) and not self._is_likely_body_text(txt, rect, page_rect)
            is_small_relaxed = avg_size <= (median_est + 1.5) and not self._is_likely_body_text(
                txt, rect, page_rect
            )
            if has_prefix or is_small or is_small_relaxed:
                categories[i] = "callout"

        # Fourth pass: spatial clustering of callouts
        changed = True
        while changed:
            changed = False
            for i, (txt, rect, avg_size) in enumerate(text_blocks):
                if categories[i] is not None:
                    continue
                if avg_size > (median_est + 1.5):
                    continue
                for j, (_, rect_j, _) in enumerate(text_blocks):
                    if categories[j] != "callout":
                        continue
                    gap = max(rect.y0 - rect_j.y1, rect_j.y0 - rect.y1, 0)
                    if gap < self.CALLOUT_GAP_THRESHOLD:
                        categories[i] = "callout"
                        changed = True
                        break

        # Fifth pass: remaining → body_text or other
        for i, (txt, rect, avg_size) in enumerate(text_blocks):
            if categories[i] is None:
                categories[i] = (
                    "body_text" if self._is_likely_body_text(txt, rect, page_rect) else "other"
                )

        return [
            (txt, rect, avg_size, cast(str, categories[i]))
            for i, (txt, rect, avg_size) in enumerate(text_blocks)
        ]

    def _build_hard_separators(
        self,
        classified: list[tuple[str, fitz.Rect, float, str]],
        page_rect: fitz.Rect,
        header_margin: float,
        footer_margin: float,
    ) -> list[tuple[str, fitz.Rect]]:
        """Build hard separators: header, footer, heading, caption, body_text."""
        effective_header = header_margin if header_margin > 0 else self.HEADER_MARGIN_PT
        effective_footer = footer_margin if footer_margin > 0 else self.FOOTER_MARGIN_PT
        separators: list[tuple[str, fitz.Rect]] = []
        separators.append(
            (
                "header",
                fitz.Rect(
                    page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y0 + effective_header
                ),
            )
        )
        separators.append(
            (
                "footer",
                fitz.Rect(
                    page_rect.x0, page_rect.y1 - effective_footer, page_rect.x1, page_rect.y1
                ),
            )
        )
        for _txt, rect, _avg_size, cat in classified:
            if cat in ("figure_caption", "table_caption", "heading"):
                separators.append((cat, rect))
            elif cat == "body_text" and (
                rect.height > 20.0 and rect.width > page_rect.width * 0.6
            ):
                separators.append((cat, rect))
        separators.sort(key=lambda x: x[1].y0)

        # Merge overlapping separators
        merged: list[tuple[str, fitz.Rect]] = []
        for sep_type, sep_rect in separators:
            if not merged:
                merged.append((sep_type, fitz.Rect(sep_rect)))
            else:
                last_type, last_rect = merged[-1]
                if sep_rect.y0 <= last_rect.y1 + self.CAPTION_OVERLAP_TOLERANCE:
                    last_rect.y1 = max(last_rect.y1, sep_rect.y1)
                    # Prefer caption/heading over body_text/header/footer
                    priority = {
                        "figure_caption": 4, "table_caption": 3, "heading": 2, "body_text": 1
                    }
                    if priority.get(sep_type, 0) > priority.get(last_type, 0):
                        merged[-1] = (sep_type, last_rect)
                else:
                    merged.append((sep_type, fitz.Rect(sep_rect)))
        return merged

    def _build_zones(
        self, separators: list[tuple[str, fitz.Rect]], page_rect: fitz.Rect
    ) -> list[_Zone]:
        """Build zones between hard separators."""
        zones: list[_Zone] = []
        for i in range(len(separators) - 1):
            y0 = separators[i][1].y1
            y1 = separators[i + 1][1].y0
            if y1 > y0 + 1.0:
                zones.append(_Zone(y0=max(y0, page_rect.y0), y1=min(y1, page_rect.y1)))
        return zones

    def _cluster_drawings(self, drawings: list[fitz.Rect]) -> list[list[fitz.Rect]]:
        """Group drawings into clusters by proximity."""
        if not drawings:
            return []
        rects = list(drawings)
        clusters: list[list[fitz.Rect]] = []
        while rects:
            cluster = [rects.pop(0)]
            changed = True
            while changed:
                changed = False
                remaining: list[fitz.Rect] = []
                for r in rects:
                    close = False
                    for c in cluster:
                        dx = max(c.x0 - r.x1, r.x0 - c.x1, 0)
                        dy = max(c.y0 - r.y1, r.y0 - c.y1, 0)
                        if dx <= self.CLUSTER_PROXIMITY and dy <= self.CLUSTER_PROXIMITY:
                            close = True
                            break
                    if close:
                        cluster.append(r)
                        changed = True
                    else:
                        remaining.append(r)
                rects = remaining
            clusters.append(cluster)
        return clusters

    def _build_clip(
        self,
        cluster_rects: list[fitz.Rect],
        zone_y0: float,
        zone_y1: float,
        classified: list[tuple[str, fitz.Rect, float, str]],
        page_rect: fitz.Rect,
    ) -> fitz.Rect:
        """Build final clip: drawing-bounded X, zone-clamped Y, edge-label expansion, padding."""
        min_y = max(min(r.y0 for r in cluster_rects), zone_y0)
        max_y = min(max(r.y1 for r in cluster_rects), zone_y1)
        min_x = min(r.x0 for r in cluster_rects)
        max_x = max(r.x1 for r in cluster_rects)
        # Edge-label expansion: expand clip to capture diagram labels
        # (e.g. legend on the side or short labels at edges).
        # Skip bullet lists — they are body text, not diagram labels.
        for txt, r, _avg_size, cat in classified:
            if cat in ("figure_caption", "table_caption", "heading"):
                continue
            # Allow short body_text near edges (diagram labels like "Example 1")
            if cat == "body_text":
                if len(txt) > self.EDGE_LABEL_MAX_LEN:
                    continue
                # Must be within horizontal range of the cluster
                if r.x1 < min_x - 50 or r.x0 > max_x + 50:
                    continue
            if "•" in txt:
                continue
            dx = max(min_x - r.x1, 0, r.x0 - max_x)
            dy = max(min_y - r.y1, 0, r.y0 - max_y)
            if cat == "callout":
                # Long callouts are likely body text, not diagram labels
                margin = (
                    self.EDGE_LABEL_MARGIN_CALLOUT
                    if len(txt) <= self.EDGE_LABEL_MAX_LEN
                    else self.EDGE_LABEL_MARGIN
                )
            else:
                margin = self.EDGE_LABEL_MARGIN
            if max(dx, dy) <= margin:
                min_x = min(min_x, r.x0)
                max_x = max(max_x, r.x1)
                min_y = min(min_y, r.y0)
                max_y = max(max_y, r.y1)
        # Clamp to zone and page, add padding
        return fitz.Rect(
            max(min_x - self.CLIP_PADDING_HORIZONTAL, page_rect.x0),
            max(min_y, zone_y0, page_rect.y0),
            min(max_x + self.CLIP_PADDING_HORIZONTAL, page_rect.x1),
            min(max_y + self.CLIP_PADDING_BOTTOM, zone_y1, page_rect.y1),
        )

    # ------------------------------------------------------------------
    # Claimed region management (top-down sequential processing)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_in_claimed(rect: fitz.Rect, claimed: list[tuple[float, float]]) -> bool:
        """Check if rect is almost fully contained in any claimed region."""
        if not claimed:
            return False
        rect_h = rect.y1 - rect.y0
        for c_y0, c_y1 in claimed:
            overlap = max(0, min(rect.y1, c_y1) - max(rect.y0, c_y0))
            if overlap > rect_h * 0.95:
                return True
        return False

    @staticmethod
    def _zone_is_claimed(zone: _Zone, claimed: list[tuple[float, float]]) -> bool:
        """Check if zone overlaps claimed regions by >50% of its height."""
        if not claimed:
            return False
        zone_h = zone.y1 - zone.y0
        for c_y0, c_y1 in claimed:
            overlap = max(0, min(zone.y1, c_y1) - max(zone.y0, c_y0))
            if overlap > zone_h * 0.5:
                return True
        return False

    @staticmethod
    def _add_claimed(y0: float, y1: float, claimed: list[tuple[float, float]]) -> None:
        """Add a claimed y-range and merge overlapping/adjacent regions."""
        claimed.append((y0, y1))
        claimed.sort(key=lambda x: x[0])
        merged: list[tuple[float, float]] = []
        for c0, c1 in claimed:
            if not merged:
                merged.append((c0, c1))
            else:
                last0, last1 = merged[-1]
                if c0 <= last1 + 5.0:  # merge if adjacent within 5pt
                    merged[-1] = (last0, max(last1, c1))
                else:
                    merged.append((c0, c1))
        claimed[:] = merged

    @staticmethod
    def _exclude_claimed(
        y0: float, y1: float, claimed: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Return sub-ranges of [y0, y1] that are not in claimed."""
        ranges: list[tuple[float, float]] = [(y0, y1)]
        for c0, c1 in claimed:
            new_ranges: list[tuple[float, float]] = []
            for r0, r1 in ranges:
                if r1 <= c0 or r0 >= c1:
                    new_ranges.append((r0, r1))
                else:
                    if r0 < c0:
                        new_ranges.append((r0, c0))
                    if r1 > c1:
                        new_ranges.append((c1, r1))
            ranges = new_ranges
            if not ranges:
                break
        return ranges

    # ------------------------------------------------------------------
    # Bidirectional figure search with drawing-density scoring
    # ------------------------------------------------------------------

    def _score_cluster(
        self,
        cluster_rects: list[fitz.Rect],
        clip: fitz.Rect,
        page: fitz.Page,
        classified: list[tuple[str, fitz.Rect, float, str]] | None = None,
    ) -> float:
        """Score a drawing cluster as a figure region.

        Higher score = more likely to be a real figure/diagram.
        Based on: drawing area ratio, body_text sparsity, image presence.
        Only 'body_text' counts against the score; callouts/diagram labels are ignored.
        """
        if not cluster_rects or clip.height < 10:
            return 0.0

        clip_area = max(clip.get_area(), 1.0)

        # 1. Drawing density: area covered by drawing rects vs clip area
        drawing_area = 0.0
        merged = []
        for r in sorted(cluster_rects, key=lambda x: x.y0):
            if merged and r.y0 <= merged[-1].y1 and r.x0 <= merged[-1].x1:
                merged[-1] = fitz.Rect(
                    min(merged[-1].x0, r.x0),
                    min(merged[-1].y0, r.y0),
                    max(merged[-1].x1, r.x1),
                    max(merged[-1].y1, r.y1),
                )
            else:
                merged.append(fitz.Rect(r))
        for r in merged:
            drawing_area += r.get_area()
        drawing_ratio = min(drawing_area / clip_area, 1.0)

        # 2. Text sparsity: count only body_text (not callouts/diagram labels)
        if classified is not None:
            text_area = sum(
                (r.x1 - r.x0) * (r.y1 - r.y0)
                for _txt, r, _avg_size, cat in classified
                if cat == "body_text" and r.intersects(clip)
            )
        else:
            raw_blocks = page.get_text("blocks", clip=clip)
            text_blocks = cast(
                list[tuple[float, float, float, float, str, int, int]], raw_blocks
            )
            text_area = sum(
                (b[2] - b[0]) * (b[3] - b[1]) for b in text_blocks
            )
        text_ratio = min(text_area / clip_area, 1.0)

        # 3. Image presence bonus
        img_bonus = 0.0
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                ibbox = fitz.Rect(pix.irect)
                if clip.intersects(ibbox):
                    img_bonus += 2.0
                pix = None
        except Exception:
            pass

        # 4. Penalty: too much body_text (likely body text region)
        text_penalty = 0.0
        if text_ratio > 0.7:
            text_penalty = 3.0
        elif text_ratio > 0.4:
            text_penalty = 1.0

        # 5. Height factor: very short clips are likely decorative lines
        height_factor = 1.0
        if clip.height < 30:
            height_factor = 0.2
        elif clip.height < 60:
            height_factor = 0.6
        elif clip.height > 150:
            height_factor = 1.2

        # 6. Table-like penalty: mostly horizontal lines = not a real figure
        table_penalty = 0.0
        if self._is_table_like_cluster(cluster_rects):
            table_penalty = 5.0

        # 7. Simple shape penalty: just a large rectangle (likely empty border)
        shape_penalty = 0.0
        if len(cluster_rects) <= 2:
            for r in cluster_rects:
                if r.width > r.height * 5 and r.height > 5:
                    shape_penalty = 3.0
                    break

        base_score = drawing_ratio * 8.0 + (1.0 - text_ratio) * 4.0
        score = (
            base_score + img_bonus - text_penalty - table_penalty - shape_penalty
        ) * height_factor
        return max(score, 0.0)

    def _is_table_like_cluster(self, cluster_rects: list[fitz.Rect]) -> bool:
        """Check if a cluster is mostly horizontal table lines.

        Returns True if the cluster appears to be table lines rather than
        a real figure/diagram.
        """
        if not cluster_rects:
            return False
        h_lines = 0
        for r in cluster_rects:
            if r.height < 2.0 and r.width > r.height * 10:
                h_lines += 1
        return len(cluster_rects) > 2 and h_lines / len(cluster_rects) > 0.8

    def _render_image(self, page: fitz.Page, clip_rect: fitz.Rect, img_path: Path) -> bool:
        """Render a page region to a PNG image (lossless, high fidelity)."""
        try:
            zoom = self.PAGE_RENDER_DPI / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            png_buf = io.BytesIO(pix.tobytes("png"))
            pil_img = Image.open(png_buf)
            pil_img.save(img_path, format="PNG")
            return True
        except Exception as e:
            logger.warning(f"Failed to render image to {img_path}: {e}")
            return False

    _UNSCANNED = object()

    def _find_tables_in_zone(
        self, page: fitz.Page, zone: _Zone, page_rect: fitz.Rect,
        precomputed_tables: list[Any] | object = _UNSCANNED,
        strategy: str = "lines_strict",
    ) -> list[tuple[str, fitz.Rect]]:
        """Find tables inside a zone.

        Returns list of (markdown, bbox) tuples.
        Uses precomputed_tables if provided to avoid repeated PyMuPDF calls.
        """
        tables_md: list[tuple[str, fitz.Rect]] = []
        if not self.TABLE_DETECTION_ENABLED:
            return tables_md
        try:
            clip_rect = fitz.Rect(page_rect.x0, zone.y0, page_rect.x1, zone.y1)
            if precomputed_tables is not self._UNSCANNED:
                tables = precomputed_tables if isinstance(precomputed_tables, list) else []
            else:
                tabs = page.find_tables(strategy=strategy, clip=clip_rect)
                if tabs is None:
                    return tables_md
                tables = tabs.tables
            for tab in tables:
                # Defensive: skip tables with empty cells
                # (PyMuPDF bug: tab.bbox crashes on empty cells)
                if not tab.cells:
                    continue
                tab_bbox = fitz.Rect(tab.bbox)
                # Skip tables not mostly inside this zone
                intersection = tab_bbox & clip_rect
                if (
                    not intersection
                    or intersection.get_area() / tab_bbox.get_area() < 0.5
                ):
                    continue
                if tab.row_count < self.TABLE_MIN_ROWS or tab.col_count < self.TABLE_MIN_COLS:
                    continue
                if tab_bbox.width < page_rect.width * self.TABLE_MIN_WIDTH_RATIO:
                    continue
                if tab_bbox.height < self.TABLE_MIN_HEIGHT_PT:
                    continue
                md_table = tab.to_markdown(clean=False)
                if md_table.strip():
                    tables_md.append((md_table.strip(), tab_bbox))
        except Exception as e:
            pnum = page.number + 1 if page.number is not None else 0
            logger.warning(f"Table detection failed on page {pnum}: {e}")
        return tables_md

    def _classify_table_style(self, zone: _Zone) -> str | None:
        """Classify zone's drawings into table style: 'grid', 'horizontal', or None.

        Analyzes drawing styles (horizontal lines, vertical lines, other shapes)
        to distinguish tables from diagrams/charts/drawings.

        - 'grid': multiple horizontal AND vertical lines, few non-line shapes,
                   reasonable text density
        - 'horizontal': no vertical lines, many horizontal lines + lots of text,
                        few non-line shapes, NOT zero-height lines
                        (find_tables cannot detect zero-height lines)
        """
        if not zone.drawings:
            return None

        h_lines = 0           # width > height * 3
        v_lines = 0           # height > width * 3
        other = 0             # curves, arrows, squares, circles, etc.

        for r in zone.drawings:
            if r.width > r.height * 3:
                h_lines += 1
            elif r.height > r.width * 3:
                v_lines += 1
            else:
                other += 1

        total_lines = h_lines + v_lines
        zone_height = max(zone.y1 - zone.y0, 1.0)

        # If no clear line structures at all, not a table
        if total_lines == 0:
            return None

        # If non-line shapes dominate (diagrams, charts, arrows), not a table
        if other > total_lines * 0.5:
            return None

        # Style A: Grid table
        if h_lines >= 4 and v_lines >= 3:
            # Bitfield diagrams have many lines but very few text blocks.
            # A real table should have at least ~1 text block per 50pt of height.
            if zone.text_block_count / zone_height < 0.02:
                logger.debug(
                    f"Grid zone filtered (low text density): "
                    f"y={zone.y0:.0f}-{zone.y1:.0f} text={zone.text_block_count} "
                    f"density={zone.text_block_count/zone_height:.3f}"
                )
                return None
            return "grid"

        # Style B: Horizontal-only table
        if v_lines <= 1 and h_lines >= 6 and zone.text_block_count >= 5:
            # Zero-height lines (AMBA style): find_tables cannot detect them at all.
            # If majority of horizontal lines are zero-height, skip to avoid empty runs.
            # NOTE: h_zero_height is counted BEFORE _fix_drawing_rect expands the rects.
            if zone.h_zero_height > 0 and zone.h_zero_height >= h_lines * 0.5:
                logger.debug(
                    f"Horizontal zone filtered (zero-height lines): "
                    f"y={zone.y0:.0f}-{zone.y1:.0f} h={h_lines} zero_h={zone.h_zero_height}"
                )
                return None
            return "horizontal"

        return None

    def process_page(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
        header_margin: float,
        footer_margin: float,
        img_dir: str | Path,
    ) -> GapsPageResult:
        """Process a single page. Returns GapsPageResult with images and tables."""
        page_idx = (page.number + 1) if page.number is not None else 0
        result = GapsPageResult()
        img_dir = Path(img_dir)

        # 1. Extract & classify text blocks
        text_dict = cast(dict[str, Any], page.get_text("dict"))
        text_blocks: list[tuple[str, fitz.Rect, float]] = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            txt = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
            if not txt:
                continue
            sizes: list[float] = []
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes.extend([span["size"]] * len(span["text"]))
            avg_size = sum(sizes) / len(sizes) if sizes else 12.0
            text_blocks.append((txt, fitz.Rect(block["bbox"]), avg_size))
        text_blocks.sort(key=lambda x: x[1].y0)
        classified = self._classify_text_blocks(text_blocks, page_rect)

        # 2. Collect captions
        figure_captions = [
            (txt, rect) for txt, rect, _, cat in classified if cat == "figure_caption"
        ]
        table_captions = [
            (txt, rect) for txt, rect, _, cat in classified if cat == "table_caption"
        ]

        # 3. Build hard separators and zones
        separators = self._build_hard_separators(
            classified, page_rect, header_margin, footer_margin
        )
        zones = self._build_zones(separators, page_rect)

        # 4. Collect drawings and assign to zones by centroid
        def _is_header_footer_rect(r: fitz.Rect, is_drawing: bool = False) -> bool:
            h = page_rect.y0 + (header_margin + 20 if is_drawing else header_margin)
            f = page_rect.y1 - (footer_margin + 20 if is_drawing else footer_margin)
            return r.y1 < h or r.y0 > f

        raw_drawings = [d.get("rect") for d in page.get_drawings() if d.get("rect")]
        # Count zero-height horizontal lines BEFORE _fix_drawing_rect expands them
        for r in raw_drawings:
            if r.height == 0.0 and r.width > r.height * 3:
                cy = (r.y0 + r.y1) / 2
                for zone in zones:
                    if zone.y0 <= cy < zone.y1:
                        zone.h_zero_height += 1
                        break
        drawing_rects = [self._fix_drawing_rect(r) for r in raw_drawings]
        for dr in drawing_rects:
            if _is_header_footer_rect(dr, is_drawing=True):
                continue
            cy = (dr.y0 + dr.y1) / 2
            for zone in zones:
                if zone.y0 <= cy < zone.y1:
                    zone.drawings.append(dr)
                    break

        # 4b. Count text blocks per zone (for horizontal-only table detection)
        for _txt, rect, _avg_size, cat in classified:
            if cat in ("figure_caption", "table_caption", "heading"):
                continue
            cy = (rect.y0 + rect.y1) / 2
            for zone in zones:
                if zone.y0 <= cy < zone.y1:
                    zone.text_block_count += 1
                    break

        # 5. Sequential top-down processing of all captions
        footer_y = page_rect.y1 - (
            footer_margin if footer_margin > 0 else self.FOOTER_MARGIN_PT
        )

        all_captions: list[tuple[float, str, str, fitz.Rect]] = []
        for txt, rect in figure_captions:
            all_captions.append((rect.y0, "figure", txt, rect))
        for txt, rect in table_captions:
            all_captions.append((rect.y0, "table", txt, rect))
        all_captions.sort(key=lambda x: x[0])

        claimed_y_ranges: list[tuple[float, float]] = []
        img_counter = 0

        for _, cap_type, caption_text, caption_rect in all_captions:
            if self._is_in_claimed(caption_rect, claimed_y_ranges):
                continue

            if cap_type == "figure":
                # Find zone above and below caption (zone-based, compatible with old logic)
                zone_above = None
                zone_below = None
                for zone in zones:
                    if zone.y1 <= caption_rect.y0 + self.CAPTION_OVERLAP_TOLERANCE:
                        zone_above = zone
                    elif zone.y0 >= caption_rect.y1 - self.CAPTION_OVERLAP_TOLERANCE:
                        if zone_below is None or zone.y0 < zone_below.y0:
                            zone_below = zone

                # Build candidates, excluding consumed and claimed zones
                candidates: list[tuple[str, _Zone]] = []
                if (
                    zone_above
                    and zone_above.drawings
                    and not zone_above.consumed
                    and not self._zone_is_claimed(zone_above, claimed_y_ranges)
                ):
                    candidates.append(("up", zone_above))
                if (
                    zone_below
                    and zone_below.drawings
                    and not zone_below.consumed
                    and not self._zone_is_claimed(zone_below, claimed_y_ranges)
                ):
                    candidates.append(("down", zone_below))

                if not candidates:
                    continue

                # Score each candidate zone to determine direction
                best_score = 0.0
                best_zone: _Zone | None = None
                for direction, zone in candidates:
                    clusters = self._cluster_drawings(zone.drawings)
                    all_rects = [r for cluster in clusters for r in cluster]
                    clip = self._build_clip(
                        all_rects, zone.y0, zone.y1, classified, page_rect
                    )
                    score = self._score_cluster(all_rects, clip, page, classified)
                    # Prefer closer zone when scores are close (< 0.5 apart)
                    dist = (
                        caption_rect.y0 - zone.y1
                        if direction == "up"
                        else zone.y0 - caption_rect.y1
                    )
                    effective_score = score - dist * 0.02
                    # Penalize zone that contains another figure caption
                    for _other_txt, other_rect in figure_captions:
                        if other_rect == caption_rect:
                            continue
                        # Check if caption is inside or touching the clip vertically
                        if (
                            clip.y0 <= other_rect.y0 < clip.y1 + 5
                            and other_rect.x1 > clip.x0
                            and other_rect.x0 < clip.x1
                        ):
                            effective_score -= 5.0
                            break
                    # Only switch if significantly better (>0.5 margin)
                    if effective_score > best_score + 0.5 or best_zone is None:
                        best_score = effective_score
                        best_zone = zone
                    elif (
                        best_zone is not None
                        and abs(effective_score - best_score) < 0.5
                        and len(zone.drawings) > len(best_zone.drawings)
                    ):
                        # Tie-breaker: prefer zone with more drawings
                        best_zone = zone
                        best_score = effective_score

                if best_zone is None or best_score < self.FIGURE_MIN_SCORE:
                    continue

                clusters = self._cluster_drawings(best_zone.drawings)
                if not clusters:
                    continue

                # Sort clusters by distance to caption
                if best_zone.y1 <= caption_rect.y0 + self.CAPTION_OVERLAP_TOLERANCE:
                    sorted_clusters = sorted(
                        clusters, key=lambda c: max(r.y1 for r in c), reverse=True
                    )
                else:
                    sorted_clusters = sorted(
                        clusters, key=lambda c: min(r.y0 for r in c)
                    )

                # Start with closest cluster, merge adjacent clusters if gap < 80pt
                target_cluster: list[fitz.Rect] = list(sorted_clusters[0])
                prev_y0 = min(r.y0 for r in sorted_clusters[0])
                prev_y1 = max(r.y1 for r in sorted_clusters[0])
                for next_cl in sorted_clusters[1:]:
                    next_y0 = min(r.y0 for r in next_cl)
                    next_y1 = max(r.y1 for r in next_cl)
                    gap = min(abs(next_y0 - prev_y1), abs(next_y1 - prev_y0))
                    if gap < 30:
                        target_cluster.extend(next_cl)
                        prev_y0 = min(prev_y0, next_y0)
                        prev_y1 = max(prev_y1, next_y1)
                    else:
                        break

                all_rects = target_cluster
                if not all_rects:
                    continue

                clip = self._build_clip(
                    all_rects, best_zone.y0, best_zone.y1, classified, page_rect
                )
                if (
                    clip.width > page_rect.width * self.FULL_PAGE_SKIP_WIDTH_RATIO
                    and clip.height > page_rect.height * self.FULL_PAGE_SKIP_HEIGHT_RATIO
                ):
                    continue

                img_counter += 1
                img_path = img_dir / f"page_{page_idx}_diagram_{img_counter:02d}.png"
                if self._render_image(page, clip, img_path):
                    result.images.append(
                        {
                            "path": str(img_path),
                            "page_idx": page_idx,
                            "caption": caption_text,
                            "bbox": [clip.x0, clip.y0, clip.x1, clip.y1],
                        }
                    )
                    self._add_claimed(clip.y0, clip.y1, claimed_y_ranges)
                    best_zone.consumed = True

            else:  # table
                table_y0 = caption_rect.y1
                table_y1 = min(footer_y, caption_rect.y1 + 300)
                search_ranges = self._exclude_claimed(
                    table_y0, table_y1, claimed_y_ranges
                )

                # Determine table strategy for this page (if not already)
                table_strategy = "lines_strict"
                for zone in zones:
                    if zone.consumed or not zone.drawings:
                        continue
                    style = self._classify_table_style(zone)
                    if style == "horizontal":
                        table_strategy = "lines"
                        break

                # Pre-compute tables once if needed
                all_page_tables: list[Any] = []
                if self.TABLE_DETECTION_ENABLED and search_ranges:
                    try:
                        tabs = page.find_tables(strategy=table_strategy)
                        if tabs is not None:
                            all_page_tables = tabs.tables
                    except Exception:
                        pass

                found_any = False
                for s_y0, s_y1 in search_ranges:
                    zone = _Zone(y0=s_y0, y1=s_y1)
                    tables_md = self._find_tables_in_zone(
                        page, zone, page_rect, all_page_tables, table_strategy
                    )
                    if tables_md:
                        # Only keep the first table (closest to caption)
                        md, tab_bbox = tables_md[0]
                        result.tables.append(
                            {
                                "page_idx": page_idx,
                                "markdown": md,
                                "caption": caption_text,
                                "bbox": [tab_bbox.x0, tab_bbox.y0, tab_bbox.x1, tab_bbox.y1],
                            }
                        )
                        found_any = True
                        self._add_claimed(tab_bbox.y0, tab_bbox.y1, claimed_y_ranges)
                        break

                # Mark caption itself as claimed if table found
                if found_any:
                    self._add_claimed(caption_rect.y0, caption_rect.y1, claimed_y_ranges)
                    # Also mark overlapping zones as consumed to avoid orphan re-processing
                    for z in zones:
                        if z.consumed:
                            continue
                        # noinspection PyUnboundLocalVariable
                        overlap = max(
                            0, min(z.y1, tab_bbox.y1) - max(z.y0, tab_bbox.y0)
                        )
                        if overlap > (z.y1 - z.y0) * 0.5:
                            z.consumed = True

        # 6. Handle orphan zones: no caption, but has drawings.
        #    These could be uncaptioned tables. If not tables, ignore.
        #    Skip zones overlapping claimed regions.
        table_strategy = "lines_strict"
        for zone in zones:
            if zone.consumed or not zone.drawings:
                continue
            # Skip if zone is mostly claimed
            search_ranges = self._exclude_claimed(
                zone.y0, zone.y1, claimed_y_ranges
            )
            if not search_ranges:
                continue
            if self._classify_table_style(zone):
                all_page_tables = []
                if self.TABLE_DETECTION_ENABLED:
                    try:
                        tabs = page.find_tables(strategy=table_strategy)
                        if tabs is not None:
                            all_page_tables = tabs.tables
                    except Exception:
                        pass
                tables_md = self._find_tables_in_zone(
                    page, zone, page_rect, all_page_tables, table_strategy
                )
                for md, tab_bbox in tables_md:
                    # Skip if this table was already extracted by a caption
                    if self._is_in_claimed(tab_bbox, claimed_y_ranges):
                        continue
                    result.tables.append(
                        {
                            "page_idx": page_idx,
                            "markdown": md,
                            "bbox": [tab_bbox.x0, tab_bbox.y0, tab_bbox.x1, tab_bbox.y1],
                        }
                    )
                    self._add_claimed(tab_bbox.y0, tab_bbox.y1, claimed_y_ranges)
                if tables_md and not all(
                    self._is_in_claimed(tab_bbox, claimed_y_ranges)
                    for _md, tab_bbox in tables_md
                ):
                    self._add_claimed(zone.y0, zone.y1, claimed_y_ranges)

        return result
