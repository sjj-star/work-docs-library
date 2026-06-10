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

        # First pass: captions
        caption_indices: list[int] = []
        for i, (txt, rect, _) in enumerate(text_blocks):
            if re.match(self.FIGURE_CAPTION_RE, txt):
                categories[i] = "figure_caption"
                caption_indices.append(i)
            elif re.match(self.TABLE_CAPTION_RE, txt):
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
        # Edge-label expansion
        for txt, r, _avg_size, cat in classified:
            if cat in ("body_text", "figure_caption", "table_caption", "heading"):
                continue
            dx = max(min_x - r.x1, 0, r.x0 - max_x)
            dy = max(min_y - r.y1, 0, r.y0 - max_y)
            if max(dx, dy) <= self.EDGE_LABEL_MARGIN:
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
    ) -> list[str]:
        """Find tables inside a zone.

        Uses precomputed_tables if provided to avoid repeated PyMuPDF calls.
        """
        tables_md: list[str] = []
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
                # Skip tables not overlapping this zone
                if not tab_bbox.intersects(clip_rect):
                    continue
                if tab.row_count < self.TABLE_MIN_ROWS or tab.col_count < self.TABLE_MIN_COLS:
                    continue
                if tab_bbox.width < page_rect.width * self.TABLE_MIN_WIDTH_RATIO:
                    continue
                if tab_bbox.height < self.TABLE_MIN_HEIGHT_PT:
                    continue
                md_table = tab.to_markdown(clean=False)
                if md_table.strip():
                    tables_md.append(md_table.strip())
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

        # 5. Extract figures by figure_caption
        img_counter = 0
        for caption_text, caption_rect in figure_captions:
            # Find zone above and below caption
            zone_above = None
            zone_below = None
            for zone in zones:
                if zone.y1 <= caption_rect.y0 + self.CAPTION_OVERLAP_TOLERANCE:
                    zone_above = zone
                elif zone.y0 >= caption_rect.y1 - self.CAPTION_OVERLAP_TOLERANCE:
                    if zone_below is None or zone.y0 < zone_below.y0:
                        zone_below = zone

            # Choose zone with drawings, prefer closer one
            candidates = []
            if zone_above and zone_above.drawings and not zone_above.consumed:
                candidates.append((zone_above, caption_rect.y0 - zone_above.y1))
            if zone_below and zone_below.drawings and not zone_below.consumed:
                candidates.append((zone_below, zone_below.y0 - caption_rect.y1))

            if not candidates:
                continue

            target_zone, _dist = min(candidates, key=lambda x: x[1])
            clusters = self._cluster_drawings(target_zone.drawings)
            all_rects = [r for cluster in clusters for r in cluster]
            if not all_rects:
                continue

            img_counter += 1
            clip = self._build_clip(
                all_rects, target_zone.y0, target_zone.y1, classified, page_rect
            )
            if (
                clip.width > page_rect.width * self.FULL_PAGE_SKIP_WIDTH_RATIO
                and clip.height > page_rect.height * self.FULL_PAGE_SKIP_HEIGHT_RATIO
            ):
                continue
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
                target_zone.consumed = True

        # 6. Determine if this page needs table detection and which strategy
        needs_table_detection = bool(table_captions)
        table_strategy = "lines_strict"
        for zone in zones:
            if zone.consumed or not zone.drawings:
                continue
            style = self._classify_table_style(zone)
            if style:
                needs_table_detection = True
                if style == "horizontal":
                    table_strategy = "lines"

        # 7. Pre-compute all tables on the page once (expensive PyMuPDF call)
        all_page_tables: list[Any] = []
        if self.TABLE_DETECTION_ENABLED and needs_table_detection:
            try:
                tabs = page.find_tables(strategy=table_strategy)
                if tabs is not None:
                    all_page_tables = tabs.tables
            except Exception:
                pass

        # 8. Extract tables by table_caption
        for caption_text, caption_rect in table_captions:
            # Table is below caption
            target_zone = None
            for zone in zones:
                if (
                    zone.y0 >= caption_rect.y1 - self.CAPTION_OVERLAP_TOLERANCE
                    and zone.y0 < caption_rect.y1 + 200
                    and not zone.consumed
                ):
                    target_zone = zone
                    break

            if target_zone is None:
                continue

            tables_md = self._find_tables_in_zone(
                page, target_zone, page_rect, all_page_tables, table_strategy
            )
            for md in tables_md:
                result.tables.append({"page_idx": page_idx, "markdown": md})
            if tables_md:
                target_zone.consumed = True

        # 9. Handle orphan zones: no caption, but has drawings.
        #    These could be uncaptioned tables. If not tables, ignore.
        for zone in zones:
            if zone.consumed or not zone.drawings:
                continue
            if self._classify_table_style(zone):
                tables_md = self._find_tables_in_zone(
                    page, zone, page_rect, all_page_tables, table_strategy
                )
                for md in tables_md:
                    result.tables.append({"page_idx": page_idx, "markdown": md})
                if tables_md:
                    zone.consumed = True

        return result
