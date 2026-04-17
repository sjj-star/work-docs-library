import hashlib
import io
import logging
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # pymupdf
from PIL import Image

from core.models import Chunk, Document, Chapter

logger = logging.getLogger(__name__)


class PDFParser:
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
    FIGURE_CAPTION_RE = r"^Figure\s+[A-Z]?\d+[-\.]\d+(?:[:\.]\s*\S|\s+[A-Z]\S*)"
    TABLE_CAPTION_RE = r"^(Table|表)\s*[A-Z]?\d+[-\.]\d+(?:[:\.]\s*\S|\s+[A-Z]\S*)"
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
        if rect.width > page_rect.width * cls.BODY_TEXT_MAX_WIDTH_RATIO and rect.height > cls.BODY_TEXT_MAX_HEIGHT:
            return False

        width_ratio = rect.width / page_rect.width
        if width_ratio > cls.BODY_TEXT_WIDTH_RATIO and rect.height > cls.BODY_TEXT_MIN_RECT_HEIGHT and len(txt) > cls.BODY_TEXT_MIN_LENGTH:
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
        categories = [None] * n

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
            near_caption = any(
                rect.y1 >= y0 and rect.y0 <= y1 for y0, y1 in caption_ranges
            )
            if not near_caption:
                continue

            has_prefix = (
                txt.startswith(self.CALLOUT_PREFIXES)
                or txt.startswith(self.CALLOUT_NOTE_PREFIX)
                or bool(callout_pattern.search(txt))
            )
            is_small = (
                avg_size <= self.CALLOUT_SMALL_FONT_MAX or avg_size <= self.CALLOUT_SMALL_FONT_RATIO * 12 + 2
            ) and not self._is_likely_body_text(txt, rect, page_rect)
            # Relaxed threshold for docs where body text is already small,
            # but exclude blocks that are clearly body paragraphs.
            is_small_relaxed = (
                avg_size <= (median_est + 1.5)
                and not self._is_likely_body_text(txt, rect, page_rect)
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

    def _find_figure_regions(self, page, page_rect):
        """Use Figure captions to locate corresponding diagram areas.
        Returns a list of fitz.Rect clips ready for rendering.

        This implementation uses a *page-level horizontal zoning* strategy:
        headers, footers, figure captions and table captions act as hard
        separators. The page is sliced into gaps between separators, and each
        gap's drawings are assigned to the nearest figure caption. This avoids
        the fragility of per-caption upward/downward body-text probing.
        """
        text_dict = page.get_text("dict")
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
            h_margin = page_rect.y0 + (
                self.HEADER_MARGIN_DRAWING_PT if is_drawing else self.HEADER_MARGIN_PT
            )
            f_margin = page_rect.y1 - (
                self.FOOTER_MARGIN_DRAWING_PT if is_drawing else self.FOOTER_MARGIN_PT
            )
            return r.y1 < h_margin or r.y0 > f_margin

        # 1. Build hard separators (header strip, footer strip, captions, table bodies)
        separators = []
        separators.append(
            fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y0 + self.HEADER_MARGIN_PT)
        )
        separators.append(
            fitz.Rect(page_rect.x0, page_rect.y1 - self.FOOTER_MARGIN_PT, page_rect.x1, page_rect.y1)
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
        table_caption_rects = [
            r for _, r, _, cat in classified_blocks if cat == "table_caption"
        ]
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
                    r for _, r, _, c in classified_blocks if c in ("figure_caption", "table_caption")
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
                centers = sorted(
                    [((d.y0 + d.y1) / 2, d) for d in drawings],
                    key=lambda x: x[0]
                )
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
            clip = fitz.Rect(
                clip.x0 - self.CLIP_PADDING_HORIZONTAL,
                clip.y0,
                clip.x1 + self.CLIP_PADDING_HORIZONTAL,
                clip.y1 + self.CLIP_PADDING_BOTTOM,
            ) & page_rect

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

    # Semantic chunking thresholds
    MERGE_MAX_CHARS = 6000
    MERGE_MAX_PAGES = 3

    def parse(self, path: str, extract_images: bool = True, output_dir: str | None = None) -> Document:
        doc = fitz.open(path)
        title = Path(path).stem
        total_pages = len(doc)

        toc = doc.get_toc()
        chapters = self._toc_to_chapters(toc, total_pages)

        file_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()

        page_records = []  # (page_num, text, images)

        if output_dir is not None:
            base_img_dir = Path(output_dir)
        else:
            base_img_dir = (
                Path(__file__).resolve().parent.parent.parent
                / "knowledge_base"
                / "images"
                / file_hash
            )

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            large_images = []

            if extract_images:
                img_list = page.get_images(full=True)
                for img_index, img in enumerate(img_list, start=1):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_img = Image.open(io.BytesIO(image_bytes))

                        if (
                            pil_img.width < self.MIN_IMAGE_WIDTH
                            or pil_img.height < self.MIN_IMAGE_HEIGHT
                        ):
                            continue

                        if self._is_low_content_image(pil_img):
                            continue

                        img_dir = base_img_dir
                        img_dir.mkdir(parents=True, exist_ok=True)
                        img_path = (
                            img_dir / f"page_{page_num + 1}_img_{img_index}.png"
                        )
                        if pil_img.mode in ("RGBA", "P"):
                            pil_img = pil_img.convert("RGB")
                        pil_img.save(img_path, format="PNG")
                        large_images.append({
                            "chunk_id": f"page_{page_num + 1}_img_{img_index}",
                            "path": str(img_path),
                        })
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image xref={xref} on page {page_num + 1}: {e}"
                        )

                try:
                    clips = self._find_figure_regions(page, page.rect)
                    for clip_index, clip_rect in enumerate(clips, start=1):
                        img_dir = base_img_dir
                        img_dir.mkdir(parents=True, exist_ok=True)
                        img_path = (
                            img_dir / f"page_{page_num + 1}_diagram_{clip_index:02d}.png"
                        )
                        zoom = self.PAGE_RENDER_DPI / 72
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                        pix.save(str(img_path))
                        large_images.append({
                            "chunk_id": f"page_{page_num + 1}_diagram_{clip_index:02d}",
                            "path": str(img_path),
                        })
                except Exception as e:
                    logger.warning(
                        f"Failed to render page {page_num + 1}: {e}"
                    )

            page_records.append((page_num + 1, text, large_images))

        doc.close()

        # Determine chapter for each page
        def _page_chapter(page_num: int):
            for ch in chapters:
                if ch.start_page <= page_num <= ch.end_page:
                    return ch.title
            return "(Untitled)"

        # Semantic grouping: merge adjacent pages if same chapter and within limits
        chunks: List[Chunk] = []
        current_pages = []
        current_texts = []
        current_images = []
        current_chars = 0

        def _flush_pages():
            nonlocal current_pages, current_texts, current_images, current_chars
            if not current_pages:
                return
            start_page = current_pages[0]
            end_page = current_pages[-1]
            # Build content
            contents = []
            for pn, txt, imgs in zip(current_pages, current_texts, current_images):
                parts = [txt] if txt else []
                if imgs:
                    img_block = "\n\n[IMAGES ON THIS PAGE]\n" + "\n".join(
                        f"- Image {i + 1} (Path: {img['path']})"
                        for i, img in enumerate(imgs)
                    )
                    parts.append(img_block)
                if parts:
                    contents.append("\n\n".join(parts))
            full_content = "\n\n[PAGE BREAK]\n\n".join(contents) if len(contents) > 1 else (contents[0] if contents else "")
            all_images = [img for imgs in current_images for img in imgs]
            meta = {}
            if all_images:
                meta["images"] = all_images
            chunks.append(Chunk(
                doc_id="",
                chunk_id=f"pages_{start_page}-{end_page}",
                content=full_content,
                chunk_type="text",
                page_start=start_page,
                page_end=end_page,
                metadata=meta,
            ))
            current_pages = []
            current_texts = []
            current_images = []
            current_chars = 0

        for page_num, text, imgs in page_records:
            ch_title = _page_chapter(page_num)
            prev_ch = _page_chapter(current_pages[-1]) if current_pages else None

            should_flush = False
            if current_pages:
                if ch_title != prev_ch:
                    should_flush = True
                elif current_chars + len(text) > self.MERGE_MAX_CHARS:
                    should_flush = True
                elif len(current_pages) >= self.MERGE_MAX_PAGES:
                    should_flush = True

            if should_flush:
                _flush_pages()

            current_pages.append(page_num)
            current_texts.append(text)
            current_images.append(imgs)
            current_chars += len(text)

        _flush_pages()

        return Document(
            doc_id=file_hash,
            title=title,
            source_path=str(Path(path).resolve()),
            file_type="pdf",
            total_pages=total_pages,
            chapters=chapters,
            chunks=chunks,
            file_hash=file_hash,
            status="pending",
        )

    @staticmethod
    def _toc_to_chapters(toc: List[Tuple[int, str, int]], total_pages: int) -> List[Chapter]:
        if not toc:
            return [Chapter(title="全文", start_page=1, end_page=total_pages, level=1)]
        chapters = []
        for i, (level, title, page) in enumerate(toc):
            start = max(1, page)
            end = total_pages
            if i + 1 < len(toc):
                end = max(start, toc[i + 1][2] - 1)
            chapters.append(Chapter(title=title, start_page=start, end_page=end, level=level))
        return chapters
