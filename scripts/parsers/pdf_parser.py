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
import pymupdf4llm
from core.config import Config
from PIL import Image

from parsers.gaps_first_scanner import GapsFirstScanner, GapsPageResult

logger = logging.getLogger(__name__)


class PDFParser:
    """PDFParser 类."""

    SUPPORTED = (".pdf",)
    MIN_IMAGE_WIDTH = Config.PARSER_MIN_IMAGE_WIDTH
    MIN_IMAGE_HEIGHT = Config.PARSER_MIN_IMAGE_HEIGHT
    PAGE_RENDER_DPI = Config.PARSER_PAGE_RENDER_DPI

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
    # letter).
    # NOTE: Due to re.match() prefix-matching semantics, explanatory sentences like
    # "Figure B1.1 shows..." are *not* excluded by this regex alone. Use
    # _is_strict_figure_caption for semantic filtering.
    FIGURE_CAPTION_RE = r"^Figure\s+[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
    TABLE_CAPTION_RE = r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"

    # Reference verbs that indicate a body-text reference sentence rather than
    # a true caption (e.g. "Table B1.3 shows the representations...").
    REFERENCE_VERBS = (
        "shows", "describes", "lists", "illustrates", "presents",
        "gives", "provides", "details", "summarizes",
    )

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
    TABLE_MIN_HEIGHT_PT = Config.PARSER_TABLE_MIN_HEIGHT_PT
    TABLE_MIN_WIDTH_RATIO = Config.PARSER_TABLE_MIN_WIDTH_RATIO

    # Table detection enhancement (Milestone 1)
    TABLE_DETECTION_ENABLED = Config.PARSER_TABLE_DETECTION_ENABLED
    TABLE_LIKELY_INDICATORS = [
        r"^\s*[-=]{3,}\s*$",  # separator lines
        r"\|\s*\w",  # markdown-style pipes
        r"\w+\s{3,}\w+\s{3,}\w+\s{3,}\w+",  # at least 4 columns
    ]
    TABLE_OVERLAP_DIAGRAM_THRESHOLD = Config.PARSER_TABLE_OVERLAP_THRESHOLD
    TABLE_MIN_ROWS = Config.PARSER_TABLE_MIN_ROWS
    TABLE_MIN_COLS = Config.PARSER_TABLE_MIN_COLS

    @classmethod
    def _is_strict_table_caption(cls, text: str) -> bool:
        """Strict table-caption detection for performance-sensitive pre-filtering.

        Excludes body-text reference sentences (e.g. "Table B1.3 shows...")
        while keeping true captions (e.g. "Table B1.3: Title").

        Continued tables ("Continued from previous page") are NOT excluded —
        they contain real tabular data and must remain eligible for future
        detection-strategy improvements.
        """
        if not re.match(cls.TABLE_CAPTION_RE, text):
            return False

        # Detect reference sentences directly via a dedicated pattern.
        # re.match() prefix-matching can truncate multi-digit numbers due to
        # backtracking (e.g. "Table B2.12" may match only "Table B2.1"),
        # so we do NOT rely on match.end() to locate the verb.
        ref_pattern = (
            r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?\s+"
            r"(?:shows|describes|lists|illustrates|presents|gives|provides|details|summarizes)\b"
        )
        if re.match(ref_pattern, text, re.IGNORECASE):
            return False

        return True

    @classmethod
    def _is_strict_figure_caption(cls, text: str) -> bool:
        """Strict figure-caption detection, analogous to _is_strict_table_caption."""
        if not re.match(cls.FIGURE_CAPTION_RE, text):
            return False

        ref_pattern = (
            r"^Figure\s*[A-Z]?\d+(?:[-\.]\d+)?\s+"
            r"(?:shows|describes|lists|illustrates|presents|gives|provides|details|summarizes)\b"
        )
        if re.match(ref_pattern, text, re.IGNORECASE):
            return False

        return True

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
            # 使用严格 caption 识别，排除正文引用句（如 "Figure B2.1 shows..."）
            if self._is_strict_figure_caption(txt):
                categories[i] = "figure_caption"
                caption_indices.append(i)
            elif self._is_strict_table_caption(txt):
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

    def parse(
        self,
        file_path: str | Path,
        output_dir: str | Path | None = None,
        tool_type: str = "expert",
    ) -> tuple[str, list[Path]]:
        """解析 PDF，返回与 BigModelParserClient.parse_pdf() 一致的 (markdown_text, image_paths)。.

        markdown_text 包含 Markdown 章节标题（# / ##）和图片引用 ![alt](images/xxx.png)。
        图片保存到 output_dir/images/ 目录下，格式为 PNG（无损高保真）。
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
            output_dir = Config.DB_PATH.parent / "parsed" / file_hash[:16]
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
        problem_pages: dict[int, dict[str, bool]] = {}  # 【Milestone 3】问题页面记录
        scanner = GapsFirstScanner()

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_idx = page_num + 1

            # 4a. 【Gaps-First】线性区域扫描：表格 + 图片
            gaps_result = scanner.process_page(
                page, page.rect, header_margin, footer_margin, img_dir
            )

            # 4b. 获取文本块（排除 diagram 区域内的文本，避免重复输出）
            diagram_regions = [fitz.Rect(img["bbox"]) for img in gaps_result.images]
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

            # 4b3. 从 gaps 构建表格元素
            table_elements = self._build_table_elements_from_gaps(gaps_result)

            # 4c. 提取图片（diagram + raster）
            diagram_images = self._build_images_from_gaps(gaps_result)
            raster_images = self._extract_raster_images(doc, page, page_idx, img_dir)

            all_image_paths.extend([img["path"] for img in raster_images])
            all_image_paths.extend([img["path"] for img in diagram_images])

            # 4e. 合并排序生成 Markdown（含表格元素）
            page_md = self._build_page_markdown(
                text_blocks, raster_images, diagram_images, table_elements
            )
            page_markdowns.append(page_md)

        # 4f. 【Milestone 3】PyMuPDF4LLM fallback：对问题页面补充检测
        if problem_pages:
            logger.info(f"PyMuPDF4LLM fallback 触发 | 问题页面: {list(problem_pages.keys())}")
            p4l_results = self._call_pymupdf4llm_for_pages(doc, list(problem_pages.keys()), img_dir)
            page_markdowns = self._fuse_p4l_tables_into_pages(
                page_markdowns, p4l_results, problem_pages
            )

        doc.close()

        # 5. 根据章节结构组织整体 Markdown
        markdown_text = self._organize_by_chapters(page_markdowns, chapter_map, toc)

        # 6. 【Milestone 2】最终存在性校验：移除断链图片引用
        markdown_text, all_image_paths = self._validate_image_links(
            markdown_text, all_image_paths, img_dir
        )

        logger.info(
            f"PDF 解析完成 | file={file_path.name} | pages={total_pages} | "
            f"images={len(all_image_paths)} | md_chars={len(markdown_text)}"
        )
        return markdown_text, all_image_paths

    # -----------------------------------------------------------------------
    # 新增辅助方法
    # -----------------------------------------------------------------------

    @classmethod
    def _is_page_likely_has_tables(cls, text_blocks: list[dict[str, Any]]) -> bool:
        """判断页面是否可能有表格（预筛选，避免无表格页面调用 find_tables）。"""
        for block in text_blocks:
            text = block.get("text", "")
            for pattern in cls.TABLE_LIKELY_INDICATORS:
                if re.search(pattern, text, re.MULTILINE):
                    return True
        return False

    @classmethod
    def _table_overlaps_diagram(
        cls, table_bbox: fitz.Rect, diagram_regions: list[fitz.Rect]
    ) -> bool:
        """判断表格区域是否与 diagram 区域重叠（保护位域图不被表格化）。"""
        table_area = table_bbox.get_area()
        if table_area <= 0:
            return False
        for dr in diagram_regions:
            # 使用 +table_bbox 创建副本，因为 intersect() 会修改原对象
            intersect = (+table_bbox).intersect(dr)
            if (
                intersect.get_area() > 0
                and intersect.get_area() / table_area > cls.TABLE_OVERLAP_DIAGRAM_THRESHOLD
            ):
                return True
        return False

    @staticmethod
    def _strip_table_text_blocks(
        text_blocks: list[dict[str, Any]], table_elements: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """从文本块列表中移除落在表格区域内的块。"""
        if not table_elements:
            return text_blocks

        kept_blocks = []
        for block in text_blocks:
            block_center_y = (block["y0"] + block["y1"]) / 2
            inside_table = False
            for te in table_elements:
                if te["bbox"].y0 <= block_center_y <= te["bbox"].y1:
                    inside_table = True
                    break
            if not inside_table:
                kept_blocks.append(block)
        return kept_blocks

    @staticmethod
    def _should_trigger_p4l_fallback(
        text_blocks: list[dict[str, Any]],
        table_elements: list[dict[str, Any]],
    ) -> bool:
        """判断当前页面是否需要触发 PyMuPDF4LLM fallback。

        触发条件：页面存在 table caption，但 Layer 1 自研检测未输出任何表格。
        """
        has_table_caption = any(
            PDFParser._is_strict_table_caption(block.get("text", "")) for block in text_blocks
        )
        return has_table_caption and not table_elements

    @staticmethod
    def _build_table_elements_from_gaps(gaps_result: GapsPageResult) -> list[dict[str, Any]]:
        """从 GapsPageResult 构建表格元素列表（兼容 _build_page_markdown 格式）."""
        table_elements: list[dict[str, Any]] = []
        for table in gaps_result.tables:
            table_elements.append(
                {
                    "type": "table",
                    "y0": 0,
                    "y1": 0,
                    "text": table["markdown"],
                }
            )
        return table_elements

    @staticmethod
    def _build_images_from_gaps(gaps_result: GapsPageResult) -> list[dict[str, Any]]:
        """从 GapsPageResult 构建 diagram 图片列表."""
        diagram_images: list[dict[str, Any]] = []
        for img in gaps_result.images:
            bbox = img.get("bbox", [0, 0, 0, 0])
            y0 = bbox[1] if len(bbox) > 1 else 0
            y1 = bbox[3] if len(bbox) > 3 else 0
            img_path = Path(img["path"])
            alt = img.get("caption", "") or f"page_{img['page_idx']}"
            diagram_images.append(
                {
                    "type": "image",
                    "y0": y0,
                    "y1": y1,
                    "path": img_path,
                    "rel_path": f"images/{img_path.name}",
                    "alt": alt,
                }
            )
        return diagram_images

    @staticmethod
    def _build_table_elements_from_uzn(uzn_result) -> list[dict[str, Any]]:
        """从 UZN 结果构建表格元素列表（兼容 _build_page_markdown 格式）."""
        table_elements: list[dict[str, Any]] = []
        for zone in uzn_result.zones:
            if not zone.table_markdown:
                continue
            # 找到 table cluster 的 bbox 作为定位
            table_bbox = zone.bbox
            for cluster in zone.clusters:
                if cluster.cluster_type == "table":
                    table_bbox = cluster.bbox
                    break
            table_elements.append(
                {
                    "type": "table",
                    "y0": table_bbox.y0,
                    "y1": table_bbox.y1,
                    "text": zone.table_markdown,
                    "bbox": table_bbox,
                }
            )
        return table_elements



    def _call_pymupdf4llm_for_pages(
        self,
        doc: fitz.Document,
        problem_pages: list[int],
        img_dir: Path,
    ) -> dict[int, dict[str, Any]]:
        """对问题页面批量调用 PyMuPDF4LLM Legacy Mode，获取补充表格数据。

        使用 use_layout(False) 启用 Legacy Mode，确保 table_strategy 等参数生效。
        hdr_info=False 禁用 heading 检测，规避标题扁平化缺陷。
        margins=0 规避 #251 bbox 包含 bug。

        返回: {page_idx (1-based): {"text": str, "tables": [...], "images": [...]}}
        """
        if not problem_pages:
            return {}

        try:
            pymupdf4llm.use_layout(False)
            results = cast(
                list[dict[str, Any]],
                pymupdf4llm.to_markdown(
                    doc,
                    pages=[p - 1 for p in problem_pages],  # 0-based
                    page_chunks=True,
                    table_strategy="lines",  # 比 lines_strict 更宽松，覆盖边缘场景
                    write_images=False,  # 不生成图片（图片不被引用，且质量差）
                    margins=0,
                    hdr_info=False,  # 禁用 heading 检测
                ),
            )
            return {p: r for p, r in zip(problem_pages, results)}
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM fallback 调用失败: {e}")
            return {}

    @staticmethod
    def _extract_tables_from_p4l_text(text: str) -> list[str]:
        """从 PyMuPDF4LLM 页面 text 中提取 Markdown 表格块。

        匹配标准 Markdown 表格格式：
        | Header1 | Header2 |
        |---------|---------|
        | Cell1   | Cell2   |
        """
        if not text:
            return []

        # 匹配完整的 Markdown 表格块
        pattern = re.compile(
            r"(?:^|\n)(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)*)",
            re.MULTILINE,
        )
        tables = []
        for match in pattern.finditer(text):
            table_md = match.group(1).strip()
            if table_md:
                tables.append(table_md)
        return tables

    @staticmethod
    def _fuse_p4l_tables_into_pages(
        page_markdowns: list[str],
        p4l_results: dict[int, dict[str, Any]],
        problem_pages: dict[int, dict[str, bool]],
    ) -> list[str]:
        """将 PyMuPDF4LLM 提取的表格融合到对应页面的 Markdown 中。

        融合策略：
        1. 从 p4l_results[page_idx]["text"] 中提取 Markdown 表格
        2. 将表格追加到对应页面 Markdown 的末尾（避免破坏现有结构）
        3. 若页面已有 Markdown 表格（Layer 1 已检测到），则不重复添加
        """
        result = list(page_markdowns)
        for page_idx, flags in problem_pages.items():
            if "missing_tables" not in flags:
                continue

            p4l_page = p4l_results.get(page_idx)
            if not p4l_page:
                continue

            text = p4l_page.get("text", "")
            p4l_tables = PDFParser._extract_tables_from_p4l_text(text)
            if not p4l_tables:
                continue

            page_md = result[page_idx - 1]

            # 若页面已有 Markdown 表格结构，跳过融合（Layer 1 已覆盖）
            has_existing_table = re.search(r"\|[^\n]+\|\n\|[-:\s|]+\|", page_md) is not None
            if has_existing_table:
                continue

            # 追加表格到页面末尾
            fused = page_md.rstrip() + "\n\n" + "\n\n".join(p4l_tables) + "\n"
            result[page_idx - 1] = fused

        return result

    @staticmethod
    def _merge_image_lists(
        primary: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
        y_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """合并主路径和 fallback 图片列表，fallback 只补充未覆盖的位置。

        去重策略：若 fallback 图片的 y 中心与 primary 中任一图片的 y 中心
        差距 < y_threshold，视为同一图片，跳过。
        """
        if y_threshold is None:
            y_threshold = Config.PARSER_IMAGE_MERGE_Y_THRESHOLD
        result = list(primary)
        for fb_img in fallback:
            fb_y = (fb_img["y0"] + fb_img["y1"]) / 2
            duplicate = any(
                abs(fb_y - ((p_img["y0"] + p_img["y1"]) / 2)) < y_threshold for p_img in primary
            )
            if not duplicate:
                result.append(fb_img)
        return result

    @staticmethod
    def _validate_image_links(
        markdown_text: str, image_paths: list[Path], img_dir: Path
    ) -> tuple[str, list[Path]]:
        """移除 Markdown 中指向不存在或空文件的图片引用，同步清理路径列表。

        仅校验 images/ 目录下的本地文件引用，外部 URL 保留。
        """
        # 1. 过滤出有效路径
        valid_paths = [p for p in image_paths if p.exists() and p.stat().st_size > 0]
        valid_names = {p.name for p in valid_paths}

        # 2. 扫描 Markdown，替换断链引用
        def _replace_broken(match: re.Match) -> str:
            _alt = match.group(1)
            rel_path = match.group(2)
            filename = Path(rel_path).name
            # 外部 URL 或有效文件保留
            if rel_path.startswith(("http://", "https://", "data:")):
                return match.group(0)
            if filename in valid_names:
                return match.group(0)
            logger.warning(f"移除断链图片引用: {rel_path}")
            return ""

        cleaned_md = re.sub(r"!\[(.*?)\]\((.*?)\)", _replace_broken, markdown_text)
        # 清理因移除引用产生的多余空行
        cleaned_md = re.sub(r"\n{3,}", "\n\n", cleaned_md)

        return cleaned_md, valid_paths

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
    TAB_MERGE_THRESHOLD_PT = Config.PARSER_TAB_MERGE_THRESHOLD_PT

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
            # 使用严格 caption 识别，排除正文引用句
            if self._is_strict_figure_caption(txt):
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

    def _extract_images_via_image_info(
        self,
        page: fitz.Page,
        page_idx: int,
        img_dir: Path,
        diagram_regions: list[fitz.Rect] | None = None,
        image_size_limit: float | None = None,
        max_images_per_page: int | None = None,
    ) -> list[dict[str, Any]]:
        """使用 page.get_image_info() 提取页面图片（替代 get_images + get_image_bbox）。

        复用 PyMuPDF4LLM 的图片检测过滤链：
        1. get_image_info() 获取图片元数据（含 bbox，不依赖定位）
        2. 尺寸过滤：width/height >= image_size_limit * page_size
        3. 去重：删除被大图片完全包含的小图片
        4. diagram 区域过滤：完全位于已识别 diagram 区域内的图片视为填充图案，跳过
        5. 限制：最多 max_images_per_page 张

        优势：get_image_info() 直接返回 bbox，不依赖 get_image_bbox() 定位，
        避免 bbox 定位失败导致的断链问题。
        """
        diagram_regions = diagram_regions or []
        if image_size_limit is None:
            image_size_limit = Config.PARSER_IMAGE_SIZE_LIMIT
        if max_images_per_page is None:
            max_images_per_page = Config.PARSER_MAX_IMAGES_PER_PAGE

        images: list[dict[str, Any]] = []
        try:
            img_info = page.get_image_info()
        except Exception as e:
            logger.warning(f"get_image_info() failed on page {page_idx}: {e}")
            return images

        # 转换 bbox 为 fitz.Rect
        for i in range(len(img_info)):
            img_info[i]["bbox"] = fitz.Rect(img_info[i]["bbox"])

        clip = page.rect

        # 过滤：尺寸、clip 交集、最小 3px
        img_info = [
            item
            for item in img_info
            if item["bbox"].width >= image_size_limit * clip.width
            and item["bbox"].height >= image_size_limit * clip.height
            and item["bbox"].intersects(clip)
            and item["bbox"].width > 3
            and item["bbox"].height > 3
        ]

        # 按面积降序排序
        img_info.sort(key=lambda i: abs(i["bbox"]), reverse=True)

        # 去重：删除被大图片完全包含的小图片
        for i in range(len(img_info) - 1, 0, -1):
            r = img_info[i]["bbox"]
            if r.is_empty:
                del img_info[i]
                continue
            for j in range(i):
                if r in img_info[j]["bbox"]:
                    del img_info[i]
                    break

        # 过滤：完全位于 diagram 区域内的图片视为填充图案/装饰元素，跳过
        if diagram_regions:
            filtered = []
            for item in img_info:
                bbox = item["bbox"]
                inside_diagram = any(bbox in dr for dr in diagram_regions)
                if not inside_diagram:
                    filtered.append(item)
            img_info = filtered

        # 限制数量
        img_info = img_info[:max_images_per_page]

        for img_index, item in enumerate(img_info, start=1):
            bbox = item["bbox"]
            try:
                # 通过 xref 提取原始图片数据
                xref = item.get("xref")
                if xref is None:
                    # 无 xref 时回退到页面区域渲染
                    pix = page.get_pixmap(clip=bbox, dpi=self.PAGE_RENDER_DPI)
                    img_path = img_dir / f"page_{page_idx}_img_{img_index:02d}.png"
                    pix.save(img_path)
                    # 渲染后检查是否为低内容图片
                    pil_check = Image.open(img_path)
                    if self._is_low_content_image(pil_check):
                        img_path.unlink(missing_ok=True)
                        continue
                else:
                    parent = page.parent
                    if parent is None:
                        continue
                    base_image = parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_img = Image.open(io.BytesIO(image_bytes))

                    if (
                        pil_img.width < self.MIN_IMAGE_WIDTH
                        or pil_img.height < self.MIN_IMAGE_HEIGHT
                    ):
                        continue
                    if self._is_low_content_image(pil_img):
                        continue

                    img_path = img_dir / f"page_{page_idx}_img_{img_index:02d}.png"
                    if pil_img.mode == "P":
                        pil_img = pil_img.convert("RGBA")
                    pil_img.save(img_path, format="PNG")

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
                logger.warning(f"Failed to extract image_info item on page {page_idx}: {e}")

        return images

    def _extract_raster_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_idx: int,
        img_dir: Path,
        diagram_regions: list[fitz.Rect] | None = None,
    ) -> list[dict[str, Any]]:
        """提取页面中的嵌入图片（raster），保存为 JPG。

        跳过完全位于已识别 diagram 区域内的图片（视为填充图案/装饰元素）。
        """
        diagram_regions = diagram_regions or []
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

                # 跳过完全位于 diagram 区域内的图片（填充图案/装饰元素）
                if diagram_regions and any(bbox in dr for dr in diagram_regions):
                    continue

                img_path = img_dir / f"page_{page_idx}_img_{img_index}.png"
                if pil_img.mode == "P":
                    pil_img = pil_img.convert("RGBA")
                pil_img.save(img_path, format="PNG")

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
        """将矢量图区域渲染为图片，保存为 PNG（无损高保真）."""
        images = []
        for i, clip_rect in enumerate(diagram_regions, start=1):
            try:
                img_path = img_dir / f"page_{page_idx}_diagram_{i:02d}.png"
                zoom = self.PAGE_RENDER_DPI / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)

                # 直接保存为 PNG（无损，保留原始质量）
                pix.save(img_path)

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
        table_elements: list[dict[str, Any]] | None = None,
    ) -> str:
        r"""将文本块、表格和图片按 y 坐标排序，生成页面 Markdown。.

        非 heading 文本块中的 \n 是 PyMuPDF 的段落内换行，替换为空格以保持同一段落。
        不同文本块之间的 \n\n 是段落分隔。
        """
        table_elements = table_elements or []
        elements = text_blocks + raster_images + diagram_images + table_elements
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
            elif elem["type"] == "table":
                parts.append(f"\n{elem['text']}\n")

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
