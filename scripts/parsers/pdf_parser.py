"""pdf_parser 模块."""

import hashlib
import io
import logging
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import fitz  # pymupdf
from PIL import Image

logger = logging.getLogger(__name__)


class PDFParser:
    """PDFParser 类."""

    SUPPORTED = (".pdf",)
    MIN_IMAGE_WIDTH = 100
    MIN_IMAGE_HEIGHT = 100
    PAGE_RENDER_DPI = 100

    # Heuristics for distinguishing body text from diagram labels
    BODY_TEXT_MAX_HEIGHT_RATIO = 0.35
    BODY_TEXT_MAX_WIDTH_RATIO = 0.85
    BODY_TEXT_MAX_HEIGHT = 80
    BODY_TEXT_WIDTH_RATIO = 0.52
    BODY_TEXT_MIN_RECT_HEIGHT = 8
    BODY_TEXT_MIN_LENGTH = 45

    # Figure caption pattern
    # Matches true captions like "Figure 1-1. Description", "Figure A1.1: Description",
    # or "Figure 1-1 Description" (where the word after the number starts with a capital
    # letter). It excludes explanatory sentences like "Figure 1-1 shows..." or
    # "Figure 1-1 is a hierarchical...".
    FIGURE_CAPTION_RE = r"^Figure\s+[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
    TABLE_CAPTION_RE = r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
    CALLOUT_PREFIXES = ("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.")
    CALLOUT_NOTE_PREFIX = "Note:"
    CALLOUT_GAP_THRESHOLD = 45
    CAPTION_OVERLAP_TOLERANCE = 5

    # Drawing rect fix for zero-width/height lines
    DRAWING_RECT_MIN_SIZE = 0.1
    DRAWING_RECT_EPSILON = 0.5

    # Header/footer margins
    HEADER_MARGIN_PT = 50
    FOOTER_MARGIN_PT = 50
    HEADER_MARGIN_DRAWING_PT = 70
    FOOTER_MARGIN_DRAWING_PT = 70

    # Clip padding
    CLIP_PADDING_HORIZONTAL = 15
    CLIP_PADDING_BOTTOM = 15

    # Small-font callout detection
    CALLOUT_SMALL_FONT_MAX = 9.0
    CALLOUT_SMALL_FONT_RATIO = 0.72

    # Aspect-ratio filters to skip tables and near-full-page noise
    TABLE_SKIP_WIDTH_RATIO = 0.72
    TABLE_SKIP_ASPECT = 6.5
    TABLE_SKIP_HEIGHT_RATIO = 0.22
    FULL_PAGE_SKIP_WIDTH_RATIO = 0.92
    FULL_PAGE_SKIP_HEIGHT_RATIO = 0.82

    # Gap text-density guard and edge-label expansion
    GAP_TEXT_DENSITY_THRESHOLD = 0.35
    EDGE_LABEL_MARGIN = 12.0

    # Drawing cluster filtering
    MAX_DRAWING_CLUSTER_SPREAD = 250.0
    DIAGRAM_SIDE_AREA_RATIO = 2.5

    # Table-region detection for figure extraction
    TABLE_MIN_HEIGHT_PT = 20.0
    TABLE_MIN_WIDTH_RATIO = 0.15

    @classmethod
    def _is_likely_body_text(cls, txt: str, rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
        """Heuristic to distinguish body paragraphs from diagram labels."""
        if rect.height > page_rect.height * cls.BODY_TEXT_MAX_HEIGHT_RATIO:
            return False
        if (
            rect.width > page_rect.width * cls.BODY_TEXT_MAX_WIDTH_RATIO
            and rect.height > cls.BODY_TEXT_MAX_HEIGHT
        ):
            return False

        width_ratio = rect.width / page_rect.width
        if (
            width_ratio > cls.BODY_TEXT_WIDTH_RATIO
            and rect.height > cls.BODY_TEXT_MIN_RECT_HEIGHT
            and len(txt) > cls.BODY_TEXT_MIN_LENGTH
        ):
            return True
        return False

    @classmethod
    def _fix_drawing_rect(cls, rect: fitz.Rect) -> fitz.Rect:
        """Expand zero-width or zero-height drawing rects so they can intersect."""
        eps = cls.DRAWING_RECT_EPSILON
        if rect.width < cls.DRAWING_RECT_MIN_SIZE and rect.height < cls.DRAWING_RECT_MIN_SIZE:
            return fitz.Rect(rect.x0 - eps, rect.y0 - eps, rect.x1 + eps, rect.y1 + eps)
        if rect.width < cls.DRAWING_RECT_MIN_SIZE:
            return fitz.Rect(rect.x0 - eps, rect.y0, rect.x1 + eps, rect.y1)
        if rect.height < cls.DRAWING_RECT_MIN_SIZE:
            return fitz.Rect(rect.x0, rect.y0 - eps, rect.x1, rect.y1 + eps)
        return rect

    @classmethod
    def _has_drawing_spanning(cls, y: float, drawing_rects: list) -> bool:
        """Return True if any drawing crosses the horizontal line at y."""
        for dr in drawing_rects:
            if dr.y0 < y and dr.y1 > y:
                return True
        return False

    @classmethod
    def _is_low_content_image(cls, pil_img: Image.Image) -> bool:
        """Detect nearly-blank or decorative images (lines, dots, empty boxes)."""
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

    def _classify_text_blocks_for_figures(self, text_blocks, page_rect):
        """Classify text blocks into figure_caption, table_caption, callout, body_text, or other.

        Callout detection uses content-pattern seeds + spatial clustering so that
        multi-line note blocks (even when font size equals body text) are treated as one cluster.
        """
        n = len(text_blocks)
        categories: list[str | None] = [None] * n

        # First pass: identify captions
        caption_indices = []
        for i, (txt, rect, _) in enumerate(text_blocks):
            if re.match(self.FIGURE_CAPTION_RE, txt):
                categories[i] = "figure_caption"
                caption_indices.append(i)
            elif re.match(self.TABLE_CAPTION_RE, txt):
                categories[i] = "table_caption"
                caption_indices.append(i)

        # Build caption proximity ranges (±250 pt)
        caption_ranges = []
        for idx in caption_indices:
            _, rect, _ = text_blocks[idx]
            caption_ranges.append((rect.y0 - 250, rect.y1 + 250))

        # Helper to check callout content patterns
        callout_pattern = re.compile(r"^[A-H]\.\s|^Note:\s*|^\([A-Z]\)")

        # Second pass: identify callout seeds
        median_est = 8.0  # constant estimate for small-font docs like datasheets
        for i, (txt, rect, avg_size) in enumerate(text_blocks):
            if categories[i] is not None:
                continue
            # Must be near a caption to be considered a callout
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
                or avg_size <= self.CALLOUT_SMALL_FONT_RATIO * 12 + 2
            ) and not self._is_likely_body_text(txt, rect, page_rect)
            # Relaxed threshold for docs where body text is already small,
            # but exclude blocks that are clearly body paragraphs.
            is_small_relaxed = avg_size <= (median_est + 1.5) and not self._is_likely_body_text(
                txt, rect, page_rect
            )

            if has_prefix or is_small or is_small_relaxed:
                categories[i] = "callout"

        # Third pass: spatial clustering – absorb adjacent small-font blocks into callout clusters
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
                    if rect.y0 >= rect_j.y1:
                        gap = rect.y0 - rect_j.y1
                    elif rect_j.y0 >= rect.y1:
                        gap = rect_j.y0 - rect.y1
                    else:
                        gap = 0
                    if gap < self.CALLOUT_GAP_THRESHOLD:
                        categories[i] = "callout"
                        changed = True
                        break

        # Fourth pass: classify remaining blocks
        for i, (txt, rect, avg_size) in enumerate(text_blocks):
            if categories[i] is None:
                if self._is_likely_body_text(txt, rect, page_rect):
                    categories[i] = "body_text"
                else:
                    categories[i] = "other"

        return [
            (txt, rect, avg_size, categories[i])
            for i, (txt, rect, avg_size) in enumerate(text_blocks)
        ]

    def _find_figure_regions(
        self,
        page,
        page_rect,
        header_margin: float = 0.0,
        footer_margin: float = 0.0,
    ):
        """Use Figure captions to locate corresponding diagram areas.

        Returns a list of fitz.Rect clips ready for rendering.

        This implementation uses a *page-level horizontal zoning* strategy:
        headers, footers, figure captions and table captions act as hard
        separators. The page is sliced into gaps between separators, and each
        gap's drawings are assigned to the nearest figure caption. This avoids
        the fragility of per-caption upward/downward body-text probing.
        """
        # 使用动态边距（来自 _analyze_document_layout 的检测结果）
        # 若未传入则回退到硬编码常量
        effective_header = header_margin if header_margin > 0 else self.HEADER_MARGIN_PT
        effective_footer = footer_margin if footer_margin > 0 else self.FOOTER_MARGIN_PT
        text_dict = cast(dict[str, Any], page.get_text("dict"))
        text_blocks = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            txt = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
            if not txt:
                continue
            sizes = []
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes.extend([span["size"]] * len(span["text"]))
            avg_size = sum(sizes) / len(sizes) if sizes else 12.0
            text_blocks.append((txt, fitz.Rect(block["bbox"]), avg_size))

        text_blocks.sort(key=lambda x: x[1].y0)
        classified_blocks = self._classify_text_blocks_for_figures(text_blocks, page_rect)

        figure_captions = [
            (txt, rect) for txt, rect, _, cat in classified_blocks if cat == "figure_caption"
        ]
        if not figure_captions:
            return []

        def _is_header_footer_rect(r: fitz.Rect, is_drawing: bool = False) -> bool:
            h_margin = page_rect.y0 + (effective_header + 20 if is_drawing else effective_header)
            f_margin = page_rect.y1 - (effective_footer + 20 if is_drawing else effective_footer)
            return r.y1 < h_margin or r.y0 > f_margin

        # 1. Build hard separators (header strip, footer strip, captions, table bodies)
        separators = []
        separators.append(
            fitz.Rect(
                page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y0 + self.HEADER_MARGIN_PT
            )
        )
        separators.append(
            fitz.Rect(
                page_rect.x0, page_rect.y1 - self.FOOTER_MARGIN_PT, page_rect.x1, page_rect.y1
            )
        )
        for txt, rect, _, cat in classified_blocks:
            if cat in ("figure_caption", "table_caption"):
                separators.append(rect)

        # Detect actual table regions and treat them as separators so that
        # figure extraction does not swallow table bodies.
        # Only keep tables that have a matching table caption directly above them
        # (leveraging the prior that tables are always captioned immediately above).
        # Lazy: skip the expensive find_tables() call entirely if the page has no
        # table caption — this avoids severe timeouts on large vector-heavy docs.
        table_caption_rects = [r for _, r, _, cat in classified_blocks if cat == "table_caption"]
        if table_caption_rects:
            try:
                for tab in page.find_tables():
                    bbox = fitz.Rect(tab.bbox)
                    if not (
                        bbox.height > self.TABLE_MIN_HEIGHT_PT
                        and bbox.width > page_rect.width * self.TABLE_MIN_WIDTH_RATIO
                    ):
                        continue
                    # Require a table caption sitting just above the table
                    has_caption_above = any(
                        cap.y1 <= bbox.y0 + 5 and cap.y1 >= bbox.y0 - 100
                        for cap in table_caption_rects
                    )
                    if has_caption_above:
                        separators.append(bbox)
            except Exception:
                pass

        separators.sort(key=lambda r: r.y0)
        merged = []
        for s in separators:
            if not merged:
                merged.append(fitz.Rect(s))
            else:
                last = merged[-1]
                if s.y0 <= last.y1 + self.CAPTION_OVERLAP_TOLERANCE:
                    last.y1 = max(last.y1, s.y1)
                else:
                    merged.append(fitz.Rect(s))
        separators = merged

        # 2. Build gaps between separators
        gaps = []
        for i in range(len(separators) - 1):
            y0 = separators[i].y1
            y1 = separators[i + 1].y0
            if y1 > y0:
                gaps.append({"y0": y0, "y1": y1, "drawings": [], "assigned_caption_idx": None})

        # 3. Collect drawings and assign to gaps by centroid
        raw_drawings = [d.get("rect") for d in page.get_drawings() if d.get("rect")]
        drawing_rects = [self._fix_drawing_rect(r) for r in raw_drawings]
        for dr in drawing_rects:
            if _is_header_footer_rect(dr, is_drawing=True):
                continue
            cy = (dr.y0 + dr.y1) / 2
            for gap in gaps:
                if gap["y0"] <= cy < gap["y1"]:
                    gap["drawings"].append(dr)
                    break

        # 4. Assign each gap to the directly-adjacent figure caption(s).
        # A gap is bounded by two separators; if one of those separators IS a
        # figure caption, the gap naturally belongs to that caption.  This
        # prevents a figure from swallowing drawings that sit on the far side
        # of an intervening table caption.
        cap_rects = {(crect.y0, crect.y1): idx for idx, (_, crect) in enumerate(figure_captions)}
        for i, gap in enumerate(gaps):
            up_idx = cap_rects.get((separators[i].y0, separators[i].y1))
            down_idx = cap_rects.get((separators[i + 1].y0, separators[i + 1].y1))

            if up_idx is not None and down_idx is None:
                gap["assigned_caption_idx"] = up_idx
            elif down_idx is not None and up_idx is None:
                gap["assigned_caption_idx"] = down_idx
            elif up_idx is not None and down_idx is not None:
                gap_cy = (gap["y0"] + gap["y1"]) / 2
                up_cy = (figure_captions[up_idx][1].y0 + figure_captions[up_idx][1].y1) / 2
                down_cy = (figure_captions[down_idx][1].y0 + figure_captions[down_idx][1].y1) / 2
                d_up = abs(gap_cy - up_cy)
                d_down = abs(gap_cy - down_cy)
                if d_up < d_down:
                    gap["assigned_caption_idx"] = up_idx
                elif d_down < d_up:
                    gap["assigned_caption_idx"] = down_idx
                else:
                    # Tie-break by drawings inside the gap
                    if gap["drawings"]:
                        d_centroid = sum((dr.y0 + dr.y1) / 2 for dr in gap["drawings"]) / len(
                            gap["drawings"]
                        )
                        gap["assigned_caption_idx"] = up_idx if d_centroid <= gap_cy else down_idx
                    else:
                        gap["assigned_caption_idx"] = up_idx
            else:
                # Orphan gap – no directly-adjacent figure caption.
                # As a lightweight fallback, merge to the nearest figure caption
                # unless any caption (figure or table) blocks the way.
                all_caption_rects = [
                    r
                    for _, r, _, c in classified_blocks
                    if c in ("figure_caption", "table_caption")
                ]
                gap_cy = (gap["y0"] + gap["y1"]) / 2
                nearest = None
                best_dist = float("inf")
                for idx, (_, crect) in enumerate(figure_captions):
                    c_cy = (crect.y0 + crect.y1) / 2
                    blocked = False
                    if c_cy < gap["y0"]:
                        for r in all_caption_rects:
                            r_cy = (r.y0 + r.y1) / 2
                            if c_cy < r_cy < gap["y0"]:
                                blocked = True
                                break
                    elif c_cy > gap["y1"]:
                        for r in all_caption_rects:
                            r_cy = (r.y0 + r.y1) / 2
                            if gap["y1"] < r_cy < c_cy:
                                blocked = True
                                break
                    else:
                        blocked = True
                    if blocked:
                        continue
                    dist = abs(gap_cy - c_cy)
                    if dist < best_dist:
                        best_dist = dist
                        nearest = idx
                if nearest is not None:
                    gap["assigned_caption_idx"] = nearest

        # 5. Filter drawings per figure caption.
        # When a caption is flanked by another caption on one side, prefer the
        # side that is NOT blocked. This mirrors the old logic's
        # upper_hit_caption / lower_hit_caption behaviour.
        # First, map each caption to its separator index.
        caption_sep_idx = {}
        for cap_idx, (_, crect) in enumerate(figure_captions):
            for i, sep in enumerate(separators):
                if abs(sep.y0 - crect.y0) < 1 and abs(sep.y1 - crect.y1) < 1:
                    caption_sep_idx[cap_idx] = i
                    break

        # 6. Build clips per figure caption
        clips = []
        for cap_idx, (caption, caption_rect) in enumerate(figure_captions):
            assigned_drawings = []
            for gap in gaps:
                if gap["assigned_caption_idx"] == cap_idx:
                    assigned_drawings.extend(gap["drawings"])

            if not assigned_drawings:
                continue

            # 5b. Filter drawings per figure caption.
            ds_above = [d for d in assigned_drawings if (d.y0 + d.y1) / 2 < caption_rect.y0]
            ds_below = [d for d in assigned_drawings if (d.y0 + d.y1) / 2 >= caption_rect.y1]

            # Remove outlier drawings on each side that are separated from the
            # main cluster by a large gap. This discards header/footer lines and
            # decorative separators sitting in distant body text while keeping
            # genuine multi-part figures.
            def _prune_outlier_drawings(drawings, is_above):
                if not drawings:
                    return drawings
                centers = sorted([((d.y0 + d.y1) / 2, d) for d in drawings], key=lambda x: x[0])
                if is_above:
                    # Closest to caption is last (largest y)
                    centers.reverse()
                # Find first inter-drawing gap that exceeds the threshold
                split_idx = len(centers)
                for i in range(1, len(centers)):
                    gap = centers[i][0] - centers[i - 1][0]
                    if gap > self.MAX_DRAWING_CLUSTER_SPREAD:
                        split_idx = i
                        break
                return [d for _, d in centers[:split_idx]]

            ds_above = _prune_outlier_drawings(ds_above, is_above=True)
            ds_below = _prune_outlier_drawings(ds_below, is_above=False)

            # Body-text barrier: if there is a wide body-text block between the
            # caption and the closest drawing cluster on a side, discard that side.
            def _has_body_text_barrier(caption_y, cluster_y, is_above):
                barrier = False
                for txt, r, _, c in classified_blocks:
                    if c != "body_text":
                        continue
                    if r.width <= page_rect.width * 0.5:
                        continue
                    if is_above:
                        if cluster_y < r.y0 and r.y1 < caption_y:
                            barrier = True
                            break
                    else:
                        if caption_y < r.y0 and r.y1 < cluster_y:
                            barrier = True
                            break
                return barrier

            if ds_above:
                closest_cy = max((d.y0 + d.y1) / 2 for d in ds_above)
                if _has_body_text_barrier(caption_rect.y0, closest_cy, is_above=True):
                    ds_above = []
            if ds_below:
                closest_cy = min((d.y0 + d.y1) / 2 for d in ds_below)
                if _has_body_text_barrier(caption_rect.y1, closest_cy, is_above=False):
                    ds_below = []

            area_above = sum(d.width * d.height for d in ds_above)
            area_below = sum(d.width * d.height for d in ds_below)

            if ds_above and not ds_below:
                filtered_drawings = ds_above
            elif ds_below and not ds_above:
                filtered_drawings = ds_below
            elif ds_above and ds_below:
                if area_above > area_below * self.DIAGRAM_SIDE_AREA_RATIO:
                    filtered_drawings = ds_above
                elif area_below > area_above * self.DIAGRAM_SIDE_AREA_RATIO:
                    filtered_drawings = ds_below
                else:
                    filtered_drawings = ds_above + ds_below
            else:
                filtered_drawings = []

            if not filtered_drawings:
                continue

            # 5c. Text-density guard: reject individual gaps that are mostly body text.
            def _gap_is_text_heavy(gap):
                gap_drawings = [
                    d for d in filtered_drawings if gap["y0"] <= (d.y0 + d.y1) / 2 < gap["y1"]
                ]
                if not gap_drawings:
                    return False

                total_area = sum(d.width * d.height for d in gap_drawings)
                min_x = min(d.x0 for d in gap_drawings)
                max_x = max(d.x1 for d in gap_drawings)
                min_y = min(d.y0 for d in gap_drawings)
                max_y = max(d.y1 for d in gap_drawings)
                height = max(1.0, max_y - min_y)
                width = max_x - min_x
                if total_area < 100 and width / height > 50:
                    for txt, r, _, c in classified_blocks:
                        if c == "body_text" and r.y1 >= min_y - 15 and r.y0 <= max_y + 15:
                            return True

                band_y0 = min_y - 20
                band_y1 = max_y + 20
                band_area = max(1.0, (band_y1 - band_y0) * page_rect.width)
                body_area = sum(
                    r.width * r.height
                    for txt, r, _, c in classified_blocks
                    if c == "body_text" and r.y0 < band_y1 and r.y1 > band_y0
                )
                return body_area / band_area > self.GAP_TEXT_DENSITY_THRESHOLD

            kept_drawings = []
            for gap in gaps:
                if gap["assigned_caption_idx"] != cap_idx:
                    continue
                gap_drawings = [
                    d for d in filtered_drawings if gap["y0"] <= (d.y0 + d.y1) / 2 < gap["y1"]
                ]
                if not gap_drawings:
                    continue
                if _gap_is_text_heavy(gap):
                    continue
                kept_drawings.extend(gap_drawings)
            filtered_drawings = kept_drawings

            if not filtered_drawings:
                continue

            # 6. Build initial bounding box from caption + drawings
            region_items = [caption_rect] + filtered_drawings
            min_x = min(r.x0 for r in region_items)
            max_x = max(r.x1 for r in region_items)
            min_y = min(r.y0 for r in region_items)
            max_y = max(r.y1 for r in region_items)
            raw_clip = fitz.Rect(min_x, min_y, max_x, max_y)

            # 6b. Edge-label expansion: include small text blocks that sit just
            # outside the raw drawing bounds (callouts, pin labels, etc.).
            for txt, r, _, c in classified_blocks:
                if c in ("body_text", "figure_caption", "table_caption"):
                    continue
                if self._is_likely_body_text(txt, r, page_rect):
                    continue
                # Compute minimum distance from rect r to raw_clip
                dx = max(raw_clip.x0 - r.x1, 0, r.x0 - raw_clip.x1)
                dy = max(raw_clip.y0 - r.y1, 0, r.y0 - raw_clip.y1)
                dist = max(dx, dy)
                if dist <= self.EDGE_LABEL_MARGIN:
                    region_items.append(r)

            if len(region_items) > 1:
                min_x = min(r.x0 for r in region_items)
                max_x = max(r.x1 for r in region_items)
                min_y = min(r.y0 for r in region_items)
                max_y = max(r.y1 for r in region_items)

            clip = fitz.Rect(min_x, min_y, max_x, max_y)
            clip = (
                fitz.Rect(
                    clip.x0 - self.CLIP_PADDING_HORIZONTAL,
                    clip.y0,
                    clip.x1 + self.CLIP_PADDING_HORIZONTAL,
                    clip.y1 + self.CLIP_PADDING_BOTTOM,
                )
                & page_rect
            )

            # Skip clips that are clearly tables by caption text or aspect ratio fallback.
            has_table_caption = any(
                re.match(self.TABLE_CAPTION_RE, t)
                for t, r, _, c in classified_blocks
                if r.intersects(clip) and c == "table_caption"
            )
            if has_table_caption:
                continue

            aspect = clip.width / clip.height if clip.height > 0 else 999
            if (
                clip.width > page_rect.width * self.TABLE_SKIP_WIDTH_RATIO
                and aspect > self.TABLE_SKIP_ASPECT
                and clip.height < page_rect.height * self.TABLE_SKIP_HEIGHT_RATIO
            ):
                continue

            if (
                clip.width > page_rect.width * self.FULL_PAGE_SKIP_WIDTH_RATIO
                and clip.height > page_rect.height * self.FULL_PAGE_SKIP_HEIGHT_RATIO
            ):
                continue

            clips.append(clip)

        return clips

    def parse(
        self,
        file_path: str | Path,
        output_dir: str | Path | None = None,
        tool_type: str = "expert",
    ) -> tuple[str, list[Path]]:
        """解析 PDF，返回与 BigModelParserClient.parse_pdf() 一致的 (markdown_text, image_paths)。.

        markdown_text 包含 Markdown 章节标题（# / ##）和图片引用 ![alt](images/xxx.jpg)。
        图片保存到 output_dir/images/ 目录下，格式为 JPG。
        矢量图区域内的文本标签被包含在渲染的图片中，不在 Markdown 中单独输出。
        """
        file_path = Path(file_path)
        doc = fitz.open(str(file_path))
        total_pages = len(doc)

        # 1. 全局文档布局分析（最高优先级：页眉/页脚检测 + 字体统计）
        header_margin, footer_margin, body_size, heading_threshold, has_bold_fonts = (
            self._analyze_document_layout(doc)
        )
        logger.info(
            f"布局分析完成 | header={header_margin:.1f}pt | footer={footer_margin:.1f}pt | "
            f"body_size={body_size:.1f}pt | heading_threshold={heading_threshold:.1f}pt | "
            f"has_bold_fonts={has_bold_fonts}"
        )

        # 2. 确定输出目录
        if output_dir is None:
            file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
            output_dir = (
                Path(__file__).resolve().parent.parent.parent
                / "knowledge_base"
                / "parsed"
                / file_hash[:16]
            )
        else:
            output_dir = Path(output_dir)
        img_dir = output_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # 3. TOC
        toc = doc.get_toc()
        toc_by_page = self._build_toc_by_page(toc) if toc else {}
        chapter_map = self._build_chapter_map(toc, total_pages)

        # 4. 逐页处理
        all_image_paths: list[Path] = []
        page_markdowns: list[str] = []

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_idx = page_num + 1

            # 4a. 识别矢量图区域（使用动态边距）
            diagram_regions = self._find_figure_regions(
                page, page.rect, header_margin, footer_margin
            )
            diagram_captions = self._get_diagram_captions(page, page.rect, diagram_regions)

            # 4b. 获取文本块（过滤页眉/页脚 + diagram 区域内）
            text_blocks = self._get_page_text_blocks(
                page, diagram_regions, header_margin, footer_margin
            )

            # 4b2. 用 TOC 识别 heading
            page_toc_entries = toc_by_page.get(page_idx, [])
            if page_toc_entries:
                text_blocks = self._identify_headings_by_toc(text_blocks, page_toc_entries)
            else:
                text_blocks = self._fallback_heading_detection(
                    text_blocks, heading_threshold, has_bold_fonts
                )

            # 4c. 提取嵌入图片
            raster_images = self._extract_raster_images(doc, page, page_idx, img_dir)

            # 4d. 渲染矢量图
            diagram_images = self._render_diagrams(
                page, page_idx, diagram_regions, diagram_captions, img_dir
            )

            all_image_paths.extend([img["path"] for img in raster_images])
            all_image_paths.extend([img["path"] for img in diagram_images])

            # 4e. 合并排序生成 Markdown
            page_md = self._build_page_markdown(text_blocks, raster_images, diagram_images)
            page_markdowns.append(page_md)

        doc.close()

        # 5. 根据章节结构组织整体 Markdown
        markdown_text = self._organize_by_chapters(page_markdowns, chapter_map, toc)

        logger.info(
            f"PDF 解析完成 | file={file_path.name} | pages={total_pages} | "
            f"images={len(all_image_paths)} | md_chars={len(markdown_text)}"
        )
        return markdown_text, all_image_paths

    # -----------------------------------------------------------------------
    # 新增辅助方法
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_horizontal_decorations(page: fitz.Page) -> list[tuple[float, float]]:
        """提取页面中的水平装饰线（实线或细高矩形）.

        返回 [(y_center, length), ...].
        同时检测线条 ("l")、矩形 ("re") 和 drawing rect，
        筛选高度 < 2pt 且接近水平的元素。
        """
        page_w = page.rect.width
        min_length = page_w * 0.7
        max_height = 2.0
        results: list[tuple[float, float]] = []

        def _add_if_match(y0: float, y1: float, x0: float, x1: float) -> None:
            height = abs(y1 - y0)
            length = abs(x1 - x0)
            if height < max_height and length > min_length:
                y_center = (y0 + y1) / 2
                results.append((y_center, length))

        for d in page.get_drawings():
            # 1. drawing 级别的 rect
            if "rect" in d:
                r = fitz.Rect(d["rect"])
                _add_if_match(r.y0, r.y1, r.x0, r.x1)

            # 2. item 级别
            for item in d.get("items", []):
                kind = item[0]
                if kind == "l":
                    p1, p2 = item[1], item[2]
                    _add_if_match(p1.y, p2.y, p1.x, p2.x)
                elif kind == "re":
                    r = fitz.Rect(item[1])
                    _add_if_match(r.y0, r.y1, r.x0, r.x1)

        return results

    @staticmethod
    def _find_consistent_line(
        line_ys: list[float], total_pages: int, tolerance: float = 3.0
    ) -> float:
        """找到全局 80% 页面一致的水平线位置.

        使用聚类算法，容差 tolerance pt。返回线的 y 坐标，找不到返回 0.0。
        """
        if not line_ys:
            return 0.0

        min_count = max(3, int(total_pages * 0.8))
        line_ys.sort()

        # 聚类：按 y 坐标分组
        clusters: list[list[float]] = []
        current = [line_ys[0]]
        for y in line_ys[1:]:
            if y - current[0] <= tolerance:
                current.append(y)
            else:
                clusters.append(current)
                current = [y]
        clusters.append(current)

        # 找最大簇
        best = max(clusters, key=len, default=[])
        if len(best) >= min_count:
            return sum(best) / len(best)
        return 0.0

    def _analyze_document_layout(
        self, doc: fitz.Document
    ) -> tuple[float, float, float, float, bool]:
        """一次性全局分析文档布局.

        基于实线装饰线检测定位页眉/页尾边界，同时收集字体统计信息.
        检测失败时 margin = 0（视为无页眉/页尾）。

        返回: (header_margin_pt, footer_margin_pt, body_size, heading_threshold, has_bold_fonts)
        """
        if len(doc) < 3:
            return (0.0, 0.0, 12.0, 14.0, False)

        page_h = doc.load_page(0).rect.height
        all_sizes: list[float] = []
        bold_sizes: list[float] = []
        header_line_ys: list[float] = []
        footer_line_ys: list[float] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # --- 装饰线检测 ---
            decorations = self._extract_horizontal_decorations(page)
            for y_center, _length in decorations:
                if y_center < page_h * 0.25:
                    header_line_ys.append(y_center)
                elif y_center > page_h * 0.75:
                    footer_line_ys.append(y_center)

            # --- 字体信息：span 级别统计 ---
            text_dict = cast(dict[str, Any], page.get_text("dict"))
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_sizes.append(span["size"])
                        if span["flags"] & 2**4:  # 第4位 = bold
                            bold_sizes.append(span["size"])

        # --- 计算正文区域边距（纯装饰线检测） ---
        header_margin = self._find_consistent_line(header_line_ys, len(doc))
        footer_line_y = self._find_consistent_line(footer_line_ys, len(doc))
        footer_margin = page_h - footer_line_y if footer_line_y > 0 else 0.0

        # 页眉回退：页眉检测失败但页尾成功时，使用页尾对称
        if header_margin == 0.0 and footer_margin > 0.0:
            header_margin = footer_margin

        # --- 计算字体统计 ---
        body_size = statistics.median(all_sizes) if all_sizes else 12.0

        if bold_sizes:
            heading_threshold = max(body_size + 2.0, min(bold_sizes))
        else:
            heading_threshold = body_size + 2.0

        has_bold_fonts = bool(bold_sizes)

        return (header_margin, footer_margin, body_size, heading_threshold, has_bold_fonts)

    def _build_toc_lookup(self, toc: list) -> set[str]:
        """从 PDF 目录/书签构建标准化标题查找集合."""
        toc_titles: set[str] = set()
        for level, title, page_num in toc:
            # 清洗：去除尾部页码和装饰点（如 "Introduction .... 3"）
            cleaned = re.sub(r"[\.\s]+\d+\s*$", "", title).strip()
            normalized = re.sub(r"\s+", " ", cleaned).lower()
            if normalized:
                toc_titles.add(normalized)
                if len(normalized) > 20:
                    toc_titles.add(normalized[:20])
        return toc_titles

    def _build_toc_by_page(
        self, toc: list[tuple[int, str, int]]
    ) -> dict[int, list[tuple[int, str]]]:
        """按页码组织 TOC 条目，返回 {page_num: [(level, cleaned_title), ...]}."""
        toc_by_page: dict[int, list[tuple[int, str]]] = {}
        for level, title, page_num in toc:
            # 仅去除由连续点号分隔的尾部页码（如 "Introduction .... 3"）
            cleaned = re.sub(r"[\.]+\s*\d+\s*$", "", title).strip()
            # 将特殊 Unicode 空格（如 \u2002 EN SPACE）替换为普通空格
            cleaned = re.sub(r"[\s\u00A0\u2000-\u200F\u202F\u205F\u3000]+", " ", cleaned).strip()
            if not cleaned:
                continue
            toc_by_page.setdefault(page_num, []).append((level, cleaned))
        return toc_by_page

    # 非技术章节标题黑名单（目录、前言、索引等不应被标记为 heading）
    _SKIP_HEADING_TITLES: set[str] = {
        "table of contents",
        "contents",
        "preface",
        "index",
        "list of figures",
        "list of tables",
        "list of abbreviations",
        "acknowledgments",
        "acknowledgements",
        "foreword",
    }

    def _identify_headings_by_toc(
        self,
        text_blocks: list[dict[str, Any]],
        page_toc_entries: list[tuple[int, str]],
    ) -> list[dict[str, Any]]:
        """用 TOC 标题文本直接匹配页面文本块，标记 heading.

        匹配策略（按优先级）：
        1. 精确匹配：文本块内容 == TOC 完整标题（跳过非技术章节黑名单）
        2. 前缀匹配：文本块 == 编号（如 "1.1"），合并后续文本块直到匹配 TOC 完整标题
        """
        matched_toc_indices: set[int] = set()

        for i, block in enumerate(text_blocks):
            if block.get("_merged"):
                continue
            text = block["text"].strip()
            if not text:
                continue

            # 将 \n 替换为空格后进行匹配（PyMuPDF 常将编号和标题名分在不同行）
            text_normalized = text.replace("\n", " ").strip()

            # 1. 精确匹配
            for tidx, (level, toc_title) in enumerate(page_toc_entries):
                if text_normalized == toc_title:
                    # 跳过非技术章节（目录、前言、索引等）
                    if toc_title.lower() in self._SKIP_HEADING_TITLES:
                        continue
                    block["is_heading"] = True
                    block["heading_level"] = level
                    block["text"] = toc_title  # 用标准标题替换原始文本
                    matched_toc_indices.add(tidx)
                    break

            if block.get("is_heading"):
                continue

            # 2. 前缀匹配：文本块是纯编号（如 "1.1"）
            if re.match(r"^\d+(?:\.\d+)*$", text_normalized):
                for tidx, (level, toc_title) in enumerate(page_toc_entries):
                    if toc_title.startswith(text + " "):
                        # 跳过非技术章节
                        if toc_title.lower() in self._SKIP_HEADING_TITLES:
                            continue
                        combined = text
                        for j in range(i + 1, len(text_blocks)):
                            if text_blocks[j].get("_merged"):
                                continue
                            next_text = text_blocks[j]["text"].strip()
                            y_gap = abs(text_blocks[j]["y0"] - block["y0"])
                            if y_gap > 10.0:  # 跨行太远，停止
                                break
                            combined = f"{combined} {next_text}".strip()
                            if combined == toc_title:
                                block["text"] = toc_title
                                block["is_heading"] = True
                                block["heading_level"] = level
                                for k in range(i + 1, j + 1):
                                    text_blocks[k]["_merged"] = True
                                matched_toc_indices.add(tidx)
                                break
                            if len(combined) > len(toc_title):
                                break
                        if block.get("is_heading"):
                            break

        # 移除标记为已合并的文本块
        return [b for b in text_blocks if not b.get("_merged")]

    def _fallback_heading_detection(
        self,
        text_blocks: list[dict[str, Any]],
        heading_threshold: float,
        has_bold_fonts: bool,
    ) -> list[dict[str, Any]]:
        """无 TOC 页面时的严格 fallback heading 检测.

        仅识别看起来像编号标题的文本块，避免表格数据噪声。
        """
        for block in text_blocks:
            text = block["text"].strip()
            avg_size = block.get("avg_size", 0.0)
            is_bold = block.get("is_bold", False)

            looks_like_heading = bool(re.match(r"^\d+(?:\.\d+)*\s+.+", text))
            if (
                looks_like_heading
                and avg_size >= heading_threshold + 2.0
                and len(text) < 80
                and (not has_bold_fonts or is_bold)
            ):
                block["is_heading"] = True
                # 根据编号深度推断层级：1 -> 1, 1.1 -> 2, 1.1.1 -> 3
                m = re.match(r"^(\d+(?:\.\d+)*)", text)
                if m:
                    block["heading_level"] = m.group(1).count(".") + 1
                else:
                    block["heading_level"] = 2
        return text_blocks

    def _match_toc_title(self, text: str, toc_titles: set[str]) -> int:
        """返回匹配置信度：0=无匹配, 1=模糊, 2=子串, 3=精确."""
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if not normalized:
            return 0

        if normalized in toc_titles:
            return 3

        for toc_title in toc_titles:
            if normalized in toc_title or toc_title in normalized:
                return 2
            if len(normalized) > 10 and len(toc_title) > 10:
                if SequenceMatcher(None, normalized, toc_title).ratio() > 0.8:
                    return 1
        return 0

    def _build_chapter_map(
        self, toc: list[tuple[int, str, int]], total_pages: int
    ) -> dict[int, str]:
        """构建页码到章节标题的映射。."""
        if not toc:
            return {i: "全文" for i in range(1, total_pages + 1)}
        chapter_map = {}
        for i, (level, title, page) in enumerate(toc):
            start = max(1, page)
            end = toc[i + 1][2] - 1 if i + 1 < len(toc) else total_pages
            for p in range(start, end + 1):
                chapter_map[p] = title
        # 填充未映射的页面
        for p in range(1, total_pages + 1):
            if p not in chapter_map:
                chapter_map[p] = "(Untitled)"
        return chapter_map

    # 制表位/排版伪换行与真正段落换行的区分阈值（pt）
    # 目录页编号与标题的 y0 差通常在 2–4pt；正文段落折行在 9–12pt 以上
    TAB_MERGE_THRESHOLD_PT = 4.0

    def _get_page_text_blocks(
        self,
        page: fitz.Page,
        diagram_regions: list[fitz.Rect],
        header_margin: float = 0.0,
        footer_margin: float = 0.0,
    ) -> list[dict[str, Any]]:
        r"""获取页面文本块，过滤掉页眉/页脚和 diagram 区域内的文本.

        基于 get_text('dict') 的 block → line → span 结构构建文本块。
        对同一 block 内相邻 line，若 y0 差 < TAB_MERGE_THRESHOLD_PT（同一视觉行），
        用空格合并（恢复制表位/排版布局语义）；否则保留 \n（真正的段落内换行）。

        heading 判断由调用方通过 _identify_headings_by_toc 或
        _fallback_heading_detection 完成。
        """
        text_dict = cast(dict[str, Any], page.get_text("dict"))
        page_h = page.rect.height

        text_blocks = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue

            bbox = fitz.Rect(block["bbox"])

            # 1. 过滤：页眉区域
            if header_margin > 0 and bbox.y0 < header_margin + 5:
                continue

            # 2. 过滤：页脚区域
            if footer_margin > 0 and bbox.y0 > page_h - footer_margin - 5:
                continue

            # 3. 过滤：文本块中心落在任一 diagram 区域内
            center_y = (bbox.y0 + bbox.y1) / 2
            inside_diagram = False
            for dr in diagram_regions:
                if dr.y0 <= center_y <= dr.y1:
                    inside_diagram = True
                    break
            if inside_diagram:
                continue

            # 构建 block text：基于 line y0 差智能合并
            sizes: list[float] = []
            is_bold = False
            line_texts: list[str] = []
            line_y0s: list[float] = []

            for line in block["lines"]:
                line_text = "".join(s["text"] for s in line["spans"]).strip()
                if not line_text:
                    continue
                line_texts.append(line_text)
                line_y0s.append(line["bbox"][1])
                for span in line["spans"]:
                    sizes.extend([span["size"]] * len(span["text"]))
                    if span["flags"] & 2**4:
                        is_bold = True

            if not line_texts:
                continue

            parts = [line_texts[0]]
            for i in range(1, len(line_texts)):
                y0_diff = abs(line_y0s[i] - line_y0s[i - 1])
                if y0_diff < self.TAB_MERGE_THRESHOLD_PT:
                    parts.append(" ")
                else:
                    parts.append("\n")
                parts.append(line_texts[i])

            text = "".join(parts).strip()
            if not text:
                continue

            avg_size = sum(sizes) / len(sizes) if sizes else 10.0

            text_blocks.append(
                {
                    "type": "text",
                    "y0": bbox.y0,
                    "y1": bbox.y1,
                    "text": text,
                    "avg_size": avg_size,
                    "is_bold": is_bold,
                    "is_heading": False,
                }
            )
        return text_blocks

    def _get_diagram_captions(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
        diagram_regions: list[fitz.Rect],
    ) -> dict[int, str]:
        """获取每个 diagram 区域对应的 figure caption 文本。."""
        text_dict = cast(dict[str, Any], page.get_text("dict"))
        captions = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            txt = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
            if re.match(self.FIGURE_CAPTION_RE, txt):
                rect = fitz.Rect(block["bbox"])
                captions.append((txt, rect))

        # 将每个 diagram 区域与最近的 caption 匹配
        region_captions: dict[int, str] = {}
        for i, region in enumerate(diagram_regions):
            best_caption = ""
            best_dist = float("inf")
            region_cy = (region.y0 + region.y1) / 2
            for cap_text, cap_rect in captions:
                cap_cy = (cap_rect.y0 + cap_rect.y1) / 2
                dist = abs(region_cy - cap_cy)
                if dist < best_dist:
                    best_dist = dist
                    best_caption = cap_text
            region_captions[i] = best_caption
        return region_captions

    def _extract_raster_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_idx: int,
        img_dir: Path,
    ) -> list[dict[str, Any]]:
        """提取页面中的嵌入图片（raster），保存为 JPG。."""
        images = []
        img_list = page.get_images(full=True)
        for img_index, img_info in enumerate(img_list, start=1):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(image_bytes))

                if pil_img.width < self.MIN_IMAGE_WIDTH or pil_img.height < self.MIN_IMAGE_HEIGHT:
                    continue
                if self._is_low_content_image(pil_img):
                    continue

                # 获取图片 bbox
                try:
                    bbox: fitz.Rect = page.get_image_bbox(img_info)  # type: ignore[reportAssignmentType]
                except Exception:
                    continue

                img_path = img_dir / f"page_{page_idx}_img_{img_index}.jpg"
                if pil_img.mode in ("RGBA", "P"):
                    pil_img = pil_img.convert("RGB")
                pil_img.save(img_path, format="JPEG", quality=90)

                rel_path = f"images/{img_path.name}"
                images.append(
                    {
                        "type": "image",
                        "y0": bbox.y0,
                        "y1": bbox.y1,
                        "path": img_path,
                        "rel_path": rel_path,
                        "alt": f"Page {page_idx} Image {img_index}",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract image xref={xref} on page {page_idx}: {e}")
        return images

    def _render_diagrams(
        self,
        page: fitz.Page,
        page_idx: int,
        diagram_regions: list[fitz.Rect],
        diagram_captions: dict[int, str],
        img_dir: Path,
    ) -> list[dict[str, Any]]:
        """将矢量图区域渲染为图片，保存为 JPG。."""
        images = []
        for i, clip_rect in enumerate(diagram_regions, start=1):
            try:
                img_path = img_dir / f"page_{page_idx}_diagram_{i:02d}.jpg"
                zoom = self.PAGE_RENDER_DPI / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)

                # pixmap 保存为 PNG 后转 JPEG
                png_buf = io.BytesIO(pix.tobytes("png"))
                pil_img = Image.open(png_buf).convert("RGB")
                pil_img.save(img_path, format="JPEG", quality=90)

                rel_path = f"images/{img_path.name}"
                alt = diagram_captions.get(i - 1, f"Page {page_idx} Diagram {i}")
                images.append(
                    {
                        "type": "image",
                        "y0": clip_rect.y0,
                        "y1": clip_rect.y1,
                        "path": img_path,
                        "rel_path": rel_path,
                        "alt": alt,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to render diagram on page {page_idx}: {e}")
        return images

    def _build_page_markdown(
        self,
        text_blocks: list[dict[str, Any]],
        raster_images: list[dict[str, Any]],
        diagram_images: list[dict[str, Any]],
    ) -> str:
        r"""将文本块和图片按 y 坐标排序，生成页面 Markdown。.

        非 heading 文本块中的 \n 是 PyMuPDF 的段落内换行，替换为空格以保持同一段落。
        不同文本块之间的 \n\n 是段落分隔。
        """
        elements = text_blocks + raster_images + diagram_images
        elements.sort(key=lambda e: e["y0"])

        parts = []
        for elem in elements:
            if elem["type"] == "text":
                if elem.get("is_heading"):
                    level = elem.get("heading_level", 2)
                    prefix = "#" * level
                    parts.append(f"{prefix} {elem['text']}\n")
                else:
                    # 保留原始换行符，让下游 LLM 理解原始排版结构
                    parts.append(elem["text"])
            elif elem["type"] == "image":
                parts.append(f"\n![{elem['alt']}]({elem['rel_path']})\n")

        return "\n\n".join(parts)

    def _organize_by_chapters(
        self,
        page_markdowns: list[str],
        chapter_map: dict[int, str],
        toc: list[tuple[int, str, int]],
    ) -> str:
        """根据 TOC 在章节起始页面前插入 # Title，组织整体 Markdown。."""
        if not toc:
            # 无 TOC，整个文档作为一个章节
            return "# 全文\n\n" + "\n\n---\n\n".join(page_markdowns)

        # 构建章节起始页映射
        chapter_starts = {}
        for level, title, page in toc:
            if page not in chapter_starts:
                chapter_starts[page] = []
            chapter_starts[page].append((level, title))

        parts = []
        for page_idx, page_md in enumerate(page_markdowns, start=1):
            if page_idx in chapter_starts:
                for level, title in chapter_starts[page_idx]:
                    # 清洗：去除尾部页码 + 特殊 Unicode 空格替换为普通空格
                    cleaned_title = re.sub(r"[\.]+\s*\d+\s*$", "", title).strip()
                    cleaned_title = (
                        re.sub(r"[\s\u00A0\u2000-\u200F\u202F\u205F\u3000]+", " ", cleaned_title)
                        .strip()
                        .lower()
                    )
                    if cleaned_title in self._SKIP_HEADING_TITLES:
                        continue
                    # 用于检测和插入的标题也清洗特殊空格
                    title_cleaned = re.sub(
                        r"[\s\u00A0\u2000-\u200F\u202F\u205F\u3000]+", " ", title
                    ).strip()
                    expected = f"{'#' * level} {title_cleaned}\n"
                    # 如果页面正文中已存在该 heading，跳过插入避免重复
                    if expected in page_md:
                        continue
                    parts.append(expected)
            if page_md.strip():
                parts.append(page_md)

        return "\n\n".join(parts)
