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
    FIGURE_CAPTION_RE = r"^Figure\s+\d+[-.]\d+\."
    TABLE_CAPTION_RE = r"^(Table|表)\s*\d+[-.]\d+"
    CALLOUT_PREFIXES = ("A.", "B.", "C.", "D.")
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

    # Aspect-ratio filters to skip tables and near-full-page noise
    TABLE_SKIP_WIDTH_RATIO = 0.72
    TABLE_SKIP_ASPECT = 6.5
    TABLE_SKIP_HEIGHT_RATIO = 0.22
    FULL_PAGE_SKIP_WIDTH_RATIO = 0.92
    FULL_PAGE_SKIP_HEIGHT_RATIO = 0.82

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

    def _find_figure_regions(self, page, page_rect):
        """Use Figure captions to locate corresponding diagram areas.
        Returns a list of fitz.Rect clips ready for rendering.
        """
        text_dict = page.get_text("dict")
        text_blocks = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            txt = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
            if not txt:
                continue
            text_blocks.append((txt, fitz.Rect(block["bbox"])))

        text_blocks.sort(key=lambda x: x[1].y0)

        figure_captions = []
        for txt, rect in text_blocks:
            if re.match(self.FIGURE_CAPTION_RE, txt):
                figure_captions.append((txt, rect))

        if not figure_captions:
            return []

        raw_drawings = [d.get("rect") for d in page.get_drawings() if d.get("rect")]
        drawing_rects = [self._fix_drawing_rect(r) for r in raw_drawings]
        clips = []

        def _is_header_footer_rect(r: fitz.Rect, is_drawing: bool = False) -> bool:
            h_margin = page_rect.y0 + (self.HEADER_MARGIN_DRAWING_PT if is_drawing else self.HEADER_MARGIN_PT)
            f_margin = page_rect.y1 - (self.FOOTER_MARGIN_DRAWING_PT if is_drawing else self.FOOTER_MARGIN_PT)
            return r.y1 < h_margin or r.y0 > f_margin

        for caption, caption_rect in figure_captions:
            # Probe upward for upper boundary.
            upper_boundary = page_rect.y0
            upper_hit_caption = False
            for txt, rect in reversed(text_blocks):
                if rect == caption_rect:
                    continue
                if rect.y1 > caption_rect.y0 - self.CAPTION_OVERLAP_TOLERANCE:
                    continue
                gap = caption_rect.y0 - rect.y1
                if gap < self.CALLOUT_GAP_THRESHOLD and (
                    txt.startswith(self.CALLOUT_PREFIXES)
                    or txt.startswith(self.CALLOUT_NOTE_PREFIX)
                ):
                    continue
                if re.match(self.FIGURE_CAPTION_RE, txt):
                    upper_boundary = rect.y1
                    upper_hit_caption = True
                    break
                if self._is_likely_body_text(txt, rect, page_rect):
                    candidate = rect.y1
                    if self._has_drawing_spanning(candidate, drawing_rects):
                        continue
                    drawings_in_interval = any(
                        dr.y1 > candidate and dr.y0 < caption_rect.y0
                        for dr in drawing_rects
                    )
                    drawings_immediately_above = any(
                        dr.y1 <= candidate and dr.y1 > candidate - 60 and dr.y0 > page_rect.y0 + self.HEADER_MARGIN_PT
                        for dr in drawing_rects
                    )
                    if not drawings_in_interval and drawings_immediately_above:
                        continue
                    upper_boundary = candidate - 0.001
                    break

            # Probe downward for lower boundary.
            lower_boundary = page_rect.y1
            lower_hit_caption = False
            for txt, rect in text_blocks:
                if rect == caption_rect:
                    continue
                if rect.y0 < caption_rect.y1 + self.CAPTION_OVERLAP_TOLERANCE:
                    continue
                gap = rect.y0 - caption_rect.y1
                if gap < self.CALLOUT_GAP_THRESHOLD and (
                    txt.startswith(self.CALLOUT_PREFIXES)
                    or txt.startswith(self.CALLOUT_NOTE_PREFIX)
                ):
                    continue
                if re.match(self.FIGURE_CAPTION_RE, txt):
                    lower_boundary = rect.y0
                    lower_hit_caption = True
                    break
                if self._is_likely_body_text(txt, rect, page_rect):
                    candidate = rect.y0
                    if self._has_drawing_spanning(candidate, drawing_rects):
                        continue
                    drawings_in_interval = any(
                        dr.y0 < candidate and dr.y1 > caption_rect.y1
                        for dr in drawing_rects
                    )
                    drawings_immediately_below = any(
                        dr.y0 >= candidate and dr.y0 < candidate + 60 and dr.y1 < page_rect.y1 - self.FOOTER_MARGIN_PT
                        for dr in drawing_rects
                    )
                    if not drawings_in_interval and drawings_immediately_below:
                        continue
                    lower_boundary = candidate + 0.001
                    break

            # Score drawing density within the two candidate bands.
            upward_area = 0.0
            downward_area = 0.0
            for dr in drawing_rects:
                if _is_header_footer_rect(dr, is_drawing=True):
                    continue
                if dr.y1 > upper_boundary and dr.y0 < caption_rect.y0:
                    upward_area += dr.width * dr.height
                if dr.y0 < lower_boundary and dr.y1 > caption_rect.y1:
                    downward_area += dr.width * dr.height

            if downward_area > upward_area * 1.2:
                region_rect = fitz.Rect(page_rect.x0, caption_rect.y0, page_rect.x1, lower_boundary)
            elif upward_area > downward_area * 1.2:
                region_rect = fitz.Rect(page_rect.x0, upper_boundary, page_rect.x1, caption_rect.y1)
            else:
                # Tie-break: prefer the side without an adjacent caption boundary.
                if upper_hit_caption and not lower_hit_caption:
                    region_rect = fitz.Rect(page_rect.x0, caption_rect.y0, page_rect.x1, lower_boundary)
                elif lower_hit_caption and not upper_hit_caption:
                    region_rect = fitz.Rect(page_rect.x0, upper_boundary, page_rect.x1, caption_rect.y1)
                else:
                    # Both or neither sides hit a caption; default to downward.
                    region_rect = fitz.Rect(page_rect.x0, caption_rect.y0, page_rect.x1, lower_boundary)

            region_items = []
            for txt, rect in text_blocks:
                if (
                    rect.intersects(region_rect)
                    and not _is_header_footer_rect(rect, is_drawing=False)
                    and not (rect != caption_rect and re.match(self.FIGURE_CAPTION_RE, txt))
                ):
                    region_items.append(rect)
            for rect in drawing_rects:
                if rect.intersects(region_rect) and not _is_header_footer_rect(rect, is_drawing=True):
                    region_items.append(rect)

            if not region_items:
                continue

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
                for t, r in text_blocks
                if r.intersects(clip) and not (r != caption_rect and re.match(self.FIGURE_CAPTION_RE, t))
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

    def parse(self, path: str, extract_images: bool = True, output_dir: str | None = None) -> Document:
        doc = fitz.open(path)
        title = Path(path).stem
        total_pages = len(doc)

        toc = doc.get_toc()
        chapters = self._toc_to_chapters(toc, total_pages)

        file_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()

        chunks: List[Chunk] = []
        page_images: dict[int, List[dict]] = {}

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
            if text:
                chunks.append(Chunk(
                    doc_id="",
                    chunk_id=f"page_{page_num + 1}",
                    content=text,
                    chunk_type="text",
                    page_start=page_num + 1,
                    page_end=page_num + 1,
                ))
            if extract_images:
                img_list = page.get_images(full=True)
                large_images = []
                for img_index, img in enumerate(img_list, start=1):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_img = Image.open(io.BytesIO(image_bytes))

                        # Skip tiny decorative images (dots, bullets, etc.)
                        if (
                            pil_img.width < self.MIN_IMAGE_WIDTH
                            or pil_img.height < self.MIN_IMAGE_HEIGHT
                        ):
                            continue

                        # Skip nearly-blank decorative images (lines, separators, empty boxes)
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

                # Also locate diagrams by their Figure captions and render
                # the diagram region. This runs regardless of whether large
                # embedded raster images were found, so mixed pages are handled.
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

                if large_images:
                    page_images[page_num + 1] = large_images

        # Merge image references into same-page text chunks
        for ck in chunks:
            if ck.chunk_type == "text" and ck.page_start in page_images:
                imgs = page_images[ck.page_start]
                img_block = "\n\n[IMAGES ON THIS PAGE]\n" + "\n".join(
                    f"- Image {i + 1} (Path: {img['path']})"
                    for i, img in enumerate(imgs)
                )
                ck.content += img_block
                ck.metadata["images"] = imgs
                del page_images[ck.page_start]

        # Create text chunks for pages that contain only images (no text)
        for page_num in sorted(page_images.keys()):
            imgs = page_images[page_num]
            content = "\n\n[IMAGES ON THIS PAGE]\n" + "\n".join(
                f"- Image {i + 1} (Path: {img['path']})"
                for i, img in enumerate(imgs)
            )
            chunks.append(Chunk(
                doc_id="",
                chunk_id=f"page_{page_num}",
                content=content,
                chunk_type="text",
                page_start=page_num,
                page_end=page_num,
                metadata={"images": imgs},
            ))

        doc.close()
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
