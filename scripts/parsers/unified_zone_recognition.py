"""Unified Zone Recognition (UZN) 模块.

将 PDF 页面的表格和图片统一识别为"兴趣区域"，在每个区域内执行局部集群检测，
通过多源融合分类器判定集群类型，最终统一输出为 Markdown 元素和图片。

架构流程：
    硬分隔符构建 → 兴趣区域化 → 局部集群检测 → 多源融合分类 → Caption 配对 → 统一渲染
"""

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import fitz
from core.config import Config
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class ZoneCluster:
    """单个 drawing 集群（在兴趣区域内检测到的图形/表格候选）."""

    bbox: fitz.Rect
    cluster_type: Literal["table", "diagram", "raster_image", "noise", "uncertain"] = "uncertain"
    # 内部 drawing 路径（用于 is_significant 等后续分析）
    paths: list[dict[str, Any]] = field(default_factory=list)
    # 匹配的 caption 文本（如果有）
    caption_text: str = ""
    # 匹配的 caption bbox
    caption_bbox: fitz.Rect | None = None
    # 内部文本块列表（用于表格/图形判别）
    inner_text_blocks: list[tuple[str, fitz.Rect]] = field(default_factory=list)


@dataclass
class Zone:
    """兴趣区域（Zone）— 页面上的一个垂直条带，包含潜在的图形或表格."""

    # 区域垂直范围
    y0: float
    y1: float
    # 区域水平范围（通常为全页宽，但可能因多栏布局而不同）
    x0: float
    x1: float
    # 区域类型：由什么类型的 caption 定义
    zone_type: Literal["figure", "table", "orphan"] = "orphan"
    # 关联的 caption 文本
    caption_text: str = ""
    # 关联的 caption bbox
    caption_bbox: fitz.Rect | None = None
    # 区域内的 clusters
    clusters: list[ZoneCluster] = field(default_factory=list)
    # 区域内检测到的表格元素（Markdown）
    table_markdown: str = ""
    # 区域内渲染的图片路径
    image_path: Path | None = None
    # 区域内提取的 raster 图片信息
    raster_images: list[dict[str, Any]] = field(default_factory=list)

    @property
    def bbox(self) -> fitz.Rect:
        """Return zone bounding box."""
        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)


# 模块级类型别名
HardSep = tuple[str, fitz.Rect]


@dataclass
class UZNPageResult:
    """单页 UZN 处理结果."""

    page_idx: int
    zones: list[Zone] = field(default_factory=list)
    # 页面级 raster 图片（无 caption 关联的嵌入图）
    page_raster_images: list[dict[str, Any]] = field(default_factory=list)
    # 所有图片路径（用于返回给调用方）
    all_image_paths: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ZoneBuilder — 硬分隔符 → 兴趣区域
# ---------------------------------------------------------------------------


class ZoneBuilder:
    """根据页面文本块分类结果，构建垂直兴趣区域（Zone）.

    算法原理：
    1. 收集所有硬分隔符：header/footer/figure captions/table captions/table bodies
    2. 按 y 坐标排序，合并同类相邻分隔符
    3. 为每个 caption 构建兴趣区域，向上下扩展直到遇到其他硬分隔符
    4. 未被 caption 覆盖的区域标记为 orphan zone
    """

    # 区域向 caption 外扩展的最大距离（pt）
    MAX_ZONE_EXTENSION = 600.0
    # body_text 作为硬分隔符的最小宽度比例
    BODY_TEXT_BARRIER_WIDTH_RATIO = 0.5
    # figure caption 向上搜索时，table_caption 作为硬停止点
    TABLE_CAPTION_STOP_DISTANCE = 50.0

    FIGURE_CAPTION_RE = r"^Figure\s+[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
    TABLE_CAPTION_RE = r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"

    REFERENCE_VERBS = (
        "shows",
        "describes",
        "lists",
        "illustrates",
        "presents",
        "gives",
        "provides",
        "details",
        "summarizes",
    )

    @classmethod
    def _is_strict_figure_caption(cls, text: str) -> bool:
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
    def _is_strict_table_caption(cls, text: str) -> bool:
        if not re.match(cls.TABLE_CAPTION_RE, text):
            return False
        ref_pattern = (
            r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?\s+"
            r"(?:shows|describes|lists|illustrates|presents|gives|provides|details|summarizes)\b"
        )
        if re.match(ref_pattern, text, re.IGNORECASE):
            return False
        return True

    @classmethod
    def _is_likely_body_text(cls, txt: str, rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
        """Heuristic to distinguish body paragraphs from diagram labels."""
        if rect.height > page_rect.height * 0.35:
            return False
        if rect.width > page_rect.width * 0.85 and rect.height > 80:
            return False
        width_ratio = rect.width / page_rect.width
        if width_ratio > 0.52 and rect.height > 8 and len(txt) > 45:
            return True
        return False

    def build_zones(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
        header_margin: float = 0.0,
        footer_margin: float = 0.0,
    ) -> list[Zone]:
        """构建页面兴趣区域列表."""
        text_dict = cast(dict[str, Any], page.get_text("dict"))
        blocks: list[tuple[str, fitz.Rect, str]] = []
        raw_blocks = text_dict.get("blocks") or []
        for block in raw_blocks:
            if "lines" not in block:
                continue
            txt = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
            if not txt:
                continue
            blocks.append((txt, fitz.Rect(block["bbox"]), "other"))

        blocks.sort(key=lambda x: x[1].y0)

        # 1. 分类文本块
        figure_captions: list[tuple[str, fitz.Rect]] = []
        table_captions: list[tuple[str, fitz.Rect]] = []
        body_texts: list[fitz.Rect] = []
        for txt, rect, _ in blocks:
            if self._is_strict_figure_caption(txt):
                figure_captions.append((txt, rect))
            elif self._is_strict_table_caption(txt):
                table_captions.append((txt, rect))
            elif self._is_likely_body_text(txt, rect, page_rect):
                body_texts.append(rect)

        # 2. 检测 table bodies（使用 lines_strict 策略，与 _detect_and_convert_tables 一致）
        table_bodies: list[fitz.Rect] = []
        table_caption_rects = [r for _, r in table_captions]
        if table_caption_rects:
            try:
                tabs_result = page.find_tables(strategy="lines_strict")
                if tabs_result is None:
                    raise ValueError("find_tables returned None")
                for tab in tabs_result:
                    bbox = fitz.Rect(tab.bbox)
                    # 只保留有 table caption 在上方 100pt 内的表格
                    has_caption_above = any(
                        cap.y1 <= bbox.y0 + 5 and cap.y1 >= bbox.y0 - 100
                        for cap in table_caption_rects
                    )
                    if has_caption_above:
                        table_bodies.append(bbox)
            except Exception:
                pass

        # 3. 构建硬分隔符列表（携带类型标签）
        separators: list[HardSep] = []
        # header
        if header_margin > 0:
            separators.append(
                ("header", fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, header_margin))
            )
        # footer
        if footer_margin > 0:
            separators.append(
                (
                    "footer",
                    fitz.Rect(
                        page_rect.x0, page_rect.y1 - footer_margin, page_rect.x1, page_rect.y1
                    ),
                )
            )
        # figure captions
        for txt, rect in figure_captions:
            separators.append(("figure_caption", rect))
        # table captions
        for txt, rect in table_captions:
            separators.append(("table_caption", rect))
        # table bodies
        for rect in table_bodies:
            separators.append(("table_body", rect))
        # body texts
        for rect in body_texts:
            separators.append(("body_text", rect))
        # 其他文本块也作为分隔符（防止 zone 无限制扩展覆盖普通文本）
        for txt, rect, _ in blocks:
            is_fig_cap = any(
                abs(rect.y0 - c[1].y0) < 1 and abs(rect.y1 - c[1].y1) < 1 for c in figure_captions
            )
            is_tab_cap = any(
                abs(rect.y0 - c[1].y0) < 1 and abs(rect.y1 - c[1].y1) < 1 for c in table_captions
            )
            is_body = any(
                abs(rect.y0 - b.y0) < 1 and abs(rect.y1 - b.y1) < 1 for b in body_texts
            )
            if not (is_fig_cap or is_tab_cap or is_body):
                separators.append(("text_block", rect))

        # 4. 排序并合并同类相邻分隔符（但 table_caption ↔ figure_caption 不合并）
        separators.sort(key=lambda x: x[1].y0)
        merged: list[HardSep] = []
        for sep_type, rect in separators:
            if not merged:
                merged.append((sep_type, fitz.Rect(rect)))
                continue
            last_type, last_rect = merged[-1]
            # 间距 < 5pt 且类型兼容才合并
            if rect.y0 <= last_rect.y1 + 5:
                # 跨类型不合并：table_caption ↔ figure_caption
                cross_type = (
                    (last_type == "table_caption" and sep_type == "figure_caption")
                    or (last_type == "figure_caption" and sep_type == "table_caption")
                    or (last_type == "table_body" and sep_type == "figure_caption")
                    or (last_type == "figure_caption" and sep_type == "table_body")
                    or (last_type == "figure_caption" and sep_type == "text_block")
                    or (last_type == "text_block" and sep_type == "figure_caption")
                    or (last_type == "table_caption" and sep_type == "text_block")
                    or (last_type == "text_block" and sep_type == "table_caption")
                )
                if not cross_type:
                    last_rect.y1 = max(last_rect.y1, rect.y1)
                    continue
            merged.append((sep_type, fitz.Rect(rect)))

        # 5. 构建 gaps（分隔符之间的区域 + 页面两端）
        gaps: list[tuple[float, float]] = []
        # 页面顶部到第一个 separator
        if merged and merged[0][1].y0 > page_rect.y0:
            gaps.append((page_rect.y0, merged[0][1].y0))
        # separator 之间
        for i in range(len(merged) - 1):
            y0 = merged[i][1].y1
            y1 = merged[i + 1][1].y0
            if y1 > y0:
                gaps.append((y0, y1))
        # 最后一个 separator 到页面底部
        if merged and merged[-1][1].y1 < page_rect.y1:
            gaps.append((merged[-1][1].y1, page_rect.y1))

        # 6. 为每个 figure_caption 和 table_caption 构建 Zone
        zones: list[Zone] = []
        cap_to_sep_idx: dict[str, int] = {}
        for idx, (sep_type, rect) in enumerate(merged):
            if sep_type in ("figure_caption", "table_caption"):
                # 找到对应的原始 caption 文本
                cap_text = ""
                if sep_type == "figure_caption":
                    for txt, crect in figure_captions:
                        if abs(crect.y0 - rect.y0) < 1 and abs(crect.y1 - rect.y1) < 1:
                            cap_text = txt
                            break
                else:
                    for txt, crect in table_captions:
                        if abs(crect.y0 - rect.y0) < 1 and abs(crect.y1 - rect.y1) < 1:
                            cap_text = txt
                            break
                cap_to_sep_idx[cap_text] = idx

        # figure zones：从 caption 向上下扩展
        for txt, crect in figure_captions:
            zone_y0 = crect.y0
            zone_y1 = crect.y1
            sep_idx = cap_to_sep_idx.get(txt)
            if sep_idx is None:
                continue

            # 向上扩展：检查上方的 gaps 和 separators
            for i in range(sep_idx - 1, -1, -1):
                sep_type, sep_rect = merged[i]
                if sep_type in ("header", "table_caption", "table_body"):
                    zone_y0 = sep_rect.y1
                    break
                # figure_caption 也是硬停止点：防止相邻 figure zone 重叠
                if sep_type == "figure_caption":
                    zone_y0 = sep_rect.y1
                    break
                # body_text 软停止：figure 的 diagram 可能在 caption 上方
                # （如 caption 位于 diagram 下方的排版），允许 zone 越过 body_text
                # 向上扩展以包含 diagram，但限制最大扩展距离
                if sep_type == "body_text":
                    if crect.y0 - sep_rect.y1 > self.MAX_ZONE_EXTENSION:
                        zone_y0 = max(sep_rect.y1, crect.y0 - self.MAX_ZONE_EXTENSION)
                        break
                    zone_y0 = sep_rect.y0
                    continue
                # text_block 软停止：限制扩展距离但不完全 break
                if sep_type == "text_block":
                    if crect.y0 - sep_rect.y1 > self.MAX_ZONE_EXTENSION:
                        zone_y0 = max(sep_rect.y1, crect.y0 - self.MAX_ZONE_EXTENSION)
                        break
                    # 不 break，继续向上扩展
                    zone_y0 = sep_rect.y0
                    continue
                if crect.y0 - sep_rect.y1 > self.MAX_ZONE_EXTENSION:
                    zone_y0 = max(sep_rect.y1, crect.y0 - self.MAX_ZONE_EXTENSION)
                    break
                zone_y0 = sep_rect.y0  # 继续向上
            else:
                zone_y0 = max(page_rect.y0, crect.y0 - self.MAX_ZONE_EXTENSION)

            # 向下扩展
            for i in range(sep_idx + 1, len(merged)):
                sep_type, sep_rect = merged[i]
                if sep_type in ("footer", "table_caption", "table_body"):
                    zone_y1 = sep_rect.y0
                    break
                # figure_caption 也是硬停止点
                if sep_type == "figure_caption":
                    zone_y1 = sep_rect.y0
                    break
                # body_text 作为硬停止点
                if sep_type == "body_text":
                    zone_y1 = sep_rect.y0
                    break
                # text_block 软停止：限制扩展距离但不完全 break
                if sep_type == "text_block":
                    if sep_rect.y0 - crect.y1 > self.MAX_ZONE_EXTENSION:
                        zone_y1 = min(sep_rect.y0, crect.y1 + self.MAX_ZONE_EXTENSION)
                        break
                    # 不 break，继续向下扩展
                    zone_y1 = sep_rect.y1
                    continue
                if sep_rect.y0 - crect.y1 > self.MAX_ZONE_EXTENSION:
                    zone_y1 = min(sep_rect.y0, crect.y1 + self.MAX_ZONE_EXTENSION)
                    break
                zone_y1 = sep_rect.y1
            else:
                zone_y1 = min(page_rect.y1, crect.y1 + self.MAX_ZONE_EXTENSION)

            zones.append(
                Zone(
                    y0=zone_y0,
                    y1=zone_y1,
                    x0=page_rect.x0,
                    x1=page_rect.x1,
                    zone_type="figure",
                    caption_text=txt,
                    caption_bbox=crect,
                )
            )

        # table zones：从 caption 向下扩展
        for txt, crect in table_captions:
            zone_y0 = crect.y0
            zone_y1 = crect.y1
            sep_idx = cap_to_sep_idx.get(txt)
            if sep_idx is None:
                continue

            # 向上：到 caption 顶部即可（table caption 上方通常没有表格内容）
            for i in range(sep_idx - 1, -1, -1):
                sep_type, sep_rect = merged[i]
                if sep_type == "header":
                    zone_y0 = sep_rect.y1
                    break
                # body_text 作为硬停止点，text_block 软停止
                if sep_type == "body_text":
                    zone_y0 = sep_rect.y1
                    break
                if sep_type == "text_block":
                    if crect.y0 - sep_rect.y1 > self.MAX_ZONE_EXTENSION:
                        zone_y0 = max(sep_rect.y1, crect.y0 - self.MAX_ZONE_EXTENSION)
                        break
                    zone_y0 = sep_rect.y0
                    continue
            else:
                zone_y0 = max(page_rect.y0, crect.y0 - 50)

            # 向下扩展
            for i in range(sep_idx + 1, len(merged)):
                sep_type, sep_rect = merged[i]
                if sep_type in ("footer", "figure_caption"):
                    zone_y1 = sep_rect.y0
                    break
                # body_text 作为硬停止点，text_block 软停止
                if sep_type == "body_text":
                    zone_y1 = sep_rect.y0
                    break
                if sep_type == "text_block":
                    if sep_rect.y0 - crect.y1 > self.MAX_ZONE_EXTENSION:
                        zone_y1 = min(sep_rect.y0, crect.y1 + self.MAX_ZONE_EXTENSION)
                        break
                    zone_y1 = sep_rect.y1
                    continue
                if sep_rect.y0 - crect.y1 > self.MAX_ZONE_EXTENSION:
                    zone_y1 = min(sep_rect.y0, crect.y1 + self.MAX_ZONE_EXTENSION)
                    break
                zone_y1 = sep_rect.y1
            else:
                zone_y1 = min(page_rect.y1, crect.y1 + self.MAX_ZONE_EXTENSION)

            zones.append(
                Zone(
                    y0=zone_y0,
                    y1=zone_y1,
                    x0=page_rect.x0,
                    x1=page_rect.x1,
                    zone_type="table",
                    caption_text=txt,
                    caption_bbox=crect,
                )
            )

        # 7. 构建 orphan zones：gaps 中未被 caption zone 覆盖的部分
        covered_ranges = sorted([(z.y0, z.y1) for z in zones])
        for g_y0, g_y1 in gaps:
            # 从 gap 中减去所有 covered ranges，得到未覆盖的区间
            uncovered: list[tuple[float, float]] = [(g_y0, g_y1)]
            for c_y0, c_y1 in covered_ranges:
                new_uncovered: list[tuple[float, float]] = []
                for u_y0, u_y1 in uncovered:
                    if c_y1 <= u_y0 or c_y0 >= u_y1:
                        # 覆盖范围与区间不重叠
                        new_uncovered.append((u_y0, u_y1))
                    else:
                        # 覆盖范围与区间重叠，分割区间
                        if c_y0 > u_y0:
                            new_uncovered.append((u_y0, c_y0))
                        if c_y1 < u_y1:
                            new_uncovered.append((c_y1, u_y1))
                uncovered = new_uncovered

            for u_y0, u_y1 in uncovered:
                if u_y1 - u_y0 >= 30:
                    zones.append(
                        Zone(
                            y0=u_y0,
                            y1=u_y1,
                            x0=page_rect.x0,
                            x1=page_rect.x1,
                            zone_type="orphan",
                        )
                    )

        # 8. 去重：合并重叠的 zone
        # 但同类型都有 caption 的重叠 zone 不合并（如 sprui07 p209 两个相邻 figure）
        zones.sort(key=lambda z: z.y0)
        deduped: list[Zone] = []
        for z in zones:
            if deduped and z.y0 < deduped[-1].y1 and z.zone_type == deduped[-1].zone_type:
                # 两个都有 caption 的 zone 不合并，保持独立
                if deduped[-1].caption_text and z.caption_text:
                    deduped.append(z)
                else:
                    # 同类型重叠，合并
                    deduped[-1].y1 = max(deduped[-1].y1, z.y1)
                    if z.caption_text and not deduped[-1].caption_text:
                        deduped[-1].caption_text = z.caption_text
                        deduped[-1].caption_bbox = z.caption_bbox
            else:
                deduped.append(z)

        # 9. 回退：没有任何 zone 时，创建覆盖有效区域的 orphan zone
        if not deduped:
            y0 = page_rect.y0 + (header_margin if header_margin > 0 else 0)
            y1 = page_rect.y1 - (footer_margin if footer_margin > 0 else 0)
            if y1 > y0 + 30:
                deduped.append(
                    Zone(
                        y0=y0,
                        y1=y1,
                        x0=page_rect.x0,
                        x1=page_rect.x1,
                        zone_type="orphan",
                    )
                )

        return deduped


# ---------------------------------------------------------------------------
# ZoneClusterDetector — 局部集群检测
# ---------------------------------------------------------------------------


class ZoneClusterDetector:
    """在每个兴趣区域内执行局部的 drawing 集群检测.

    使用 PyMuPDF 的 page.cluster_drawings()，但只在 zone 的 bbox 范围内检测，
    避免全页扫描的性能开销。
    """

    @staticmethod
    def detect_clusters(page: fitz.Page, zone: Zone) -> list[ZoneCluster]:
        """在 zone 内检测 drawing 集群."""
        zone_rect = zone.bbox
        clusters: list[ZoneCluster] = []

        # 1. 收集 zone 内的 drawings（允许 bbox 部分重叠，不强制中心点在 zone 内）
        raw_drawings = []
        for d in page.get_drawings():
            r = d.get("rect")
            if not r:
                continue
            r = fitz.Rect(r)
            # drawing 必须与 zone 有实质重叠（允许宽度/高度接近 0 的细线）
            y_overlap = min(r.y1, zone_rect.y1) - max(r.y0, zone_rect.y0)
            x_overlap = min(r.x1, zone_rect.x1) - max(r.x0, zone_rect.x0)
            # 对于非零宽度/高度的形状，要求至少 3pt 重叠
            # 对于细竖线（width <= 1pt）或细横线（height <= 1pt），只要另一维度有重叠即可
            has_overlap = y_overlap > 3 and x_overlap > 3
            is_vertical_line = r.width <= 1.0 and y_overlap > 3
            is_horizontal_line = r.height <= 1.0 and x_overlap > 3
            if has_overlap or is_vertical_line or is_horizontal_line:
                raw_drawings.append(d)

        # 2. 检查 raster 图片（无论是否有 drawings，zone 内可能有嵌入位图）
        raster_clusters: list[ZoneCluster] = []
        try:
            for img_info in page.get_image_info():
                bbox = fitz.Rect(img_info.get("bbox", [0, 0, 0, 0]))
                if bbox.is_empty:
                    continue
                # 图片必须与 zone 有实质重叠
                y_overlap = min(bbox.y1, zone_rect.y1) - max(bbox.y0, zone_rect.y0)
                x_overlap = min(bbox.x1, zone_rect.x1) - max(bbox.x0, zone_rect.x0)
                if y_overlap > 3 and x_overlap > 3:
                    # 过滤过小图片
                    min_w = zone_rect.width * 0.05
                    min_h = zone_rect.height * 0.05
                    if bbox.width < min_w and bbox.height < min_h:
                        continue
                    raster_clusters.append(
                        ZoneCluster(
                            bbox=bbox,
                            cluster_type="raster_image",
                            paths=[],
                        )
                    )
        except Exception:
            pass

        # 如果没有 drawings，直接返回 raster clusters
        if not raw_drawings:
            return raster_clusters

        # 3. 局部 cluster_drawings
        try:
            cluster_bboxes = page.cluster_drawings(drawings=raw_drawings)
        except Exception:
            # cluster_drawings 失败时，回退到每个 drawing 的 rect 作为独立 cluster
            cluster_bboxes = [d["rect"] for d in raw_drawings if d.get("rect")]

        # 3. 过滤 tiny clusters
        for bbox in cluster_bboxes:
            bbox = fitz.Rect(bbox)
            if bbox.width < 3 and bbox.height < 3:
                continue
            # 只保留与 zone 有实质重叠的 cluster
            intersect = bbox & zone_rect
            if intersect.get_area() < 100:
                continue
            # cluster 中心点必须在 zone 内（防止大 cluster 的一小部分边缘落入相邻 zone）
            cx = (bbox.x0 + bbox.x1) / 2
            cy = (bbox.y0 + bbox.y1) / 2
            if not (zone_rect.x0 <= cx <= zone_rect.x1 and zone_rect.y0 <= cy <= zone_rect.y1):
                continue

            # 收集 cluster 内的 paths
            inner_paths = [d for d in raw_drawings if fitz.Rect(d["rect"]) in bbox]

            clusters.append(
                ZoneCluster(
                    bbox=bbox,
                    paths=inner_paths,
                )
            )

        # 合并 raster clusters，跳过与已有 diagram cluster 重叠的
        for rc in raster_clusters:
            overlaps = False
            for dc in clusters:
                inter = rc.bbox & dc.bbox
                if inter.get_area() > 100:
                    overlaps = True
                    break
            if not overlaps:
                clusters.append(rc)

        return clusters

    @staticmethod
    def is_significant(box: fitz.Rect, paths: list[dict[str, Any]]) -> bool:
        """检查 cluster 是否包含"有意义"的绘图（非纯线条/矩形）.

        借鉴 pymupdf4llm 的 is_significant，但对位域图更宽松。
        """
        if box.width > box.height:
            d = box.width * 0.025
        else:
            d = box.height * 0.025
        nbox = box + (d, d, -d, -d)

        my_paths = [p for p in paths if p.get("rect") and fitz.Rect(p["rect"]) in box]
        if not my_paths:
            # 无内部 paths，但 bbox 本身有意义（如大面积填充）
            return box.width > 50 and box.height > 50

        widths = {round(fitz.Rect(p["rect"]).width) for p in my_paths} | {round(box.width)}
        heights = {round(fitz.Rect(p["rect"]).height) for p in my_paths} | {round(box.height)}

        # 如果所有 paths 都是同一宽度或同一高度 → 纯线条/矩形
        # 但位域图也是水平线，需要额外判断
        if len(widths) == 1 or len(heights) == 1:
            # 位域图特征：水平线 + 大量文本标签 + 较宽
            # 如果 cluster 内有大量文本块，可能是位域图
            if box.width > 300 and box.height > 30:
                return True
            return False

        # 检查是否有 path 与 nbox 内部相交（非边缘）
        for p in my_paths:
            rect = fitz.Rect(p["rect"])
            if not (rect.is_empty or (rect & nbox).is_empty):
                return True

        return False


# ---------------------------------------------------------------------------
# SimpleClassifier — 多源融合分类
# ---------------------------------------------------------------------------


class SimpleClassifier:
    """基于多源特征对集群进行分类.

    优先级从高到低：
    1. Caption 强信号（table_caption 下方 → TABLE；figure_caption 附近 → DIAGRAM）
    2. 网格度检测（规则行列 → TABLE）
    3. is_significant（True → DIAGRAM）
    4. 默认 DIAGRAM（保守策略）
    """

    # figure caption 附近判定距离
    FIGURE_CAPTION_NEARBY_PT = 200.0
    # table caption 下方判定距离
    TABLE_CAPTION_BELOW_PT = 100.0

    @classmethod
    def classify(
        cls,
        cluster: ZoneCluster,
        zone: Zone,
        page_rect: fitz.Rect,
    ) -> Literal["table", "diagram", "raster_image", "noise"]:
        """对集群进行分类，返回类型."""
        bbox = cluster.bbox

        # ---- 保留 raster_image 分类（来自 ZoneClusterDetector 的明确标记）----
        if cluster.cluster_type == "raster_image":
            return "raster_image"

        # ---- 特征 1：Caption 强信号 ----
        if zone.zone_type == "table" and zone.caption_bbox:
            # cluster 位于 table caption 下方，且在 100pt 范围内 → TABLE
            cap_bottom = zone.caption_bbox.y1
            if bbox.y0 >= cap_bottom - 5 and bbox.y0 <= cap_bottom + cls.TABLE_CAPTION_BELOW_PT:
                return "table"

        if zone.zone_type == "figure" and zone.caption_bbox:
            # cluster 位于 figure caption 附近 → DIAGRAM
            cap_cy = (zone.caption_bbox.y0 + zone.caption_bbox.y1) / 2
            cluster_cy = (bbox.y0 + bbox.y1) / 2
            if abs(cluster_cy - cap_cy) < cls.FIGURE_CAPTION_NEARBY_PT:
                return "diagram"

        # ---- 特征 2：网格度检测 ----
        grid_score = cls._compute_grid_score(cluster.paths)
        if grid_score > 0.7:
            return "table"

        # ---- 特征 3：is_significant ----
        if ZoneClusterDetector.is_significant(bbox, cluster.paths):
            return "diagram"

        # ---- 特征 4：尺寸过滤 ----
        if bbox.width < page_rect.width * 0.1 and bbox.height < page_rect.height * 0.05:
            return "noise"

        # ---- 默认：DIAGRAM（保守策略，避免遗漏图形）----
        return "diagram"

    @classmethod
    def _compute_grid_score(cls, paths: list[dict[str, Any]]) -> float:
        """计算集群的"表格网格度"（0-1）.

        高网格度特征：
        - 多条水平线，y 坐标呈规则间隔
        - 多条垂直线，x 坐标呈规则间隔
        - 水平线和垂直线数量均 >= 2
        """
        horizontal_ys: list[float] = []
        vertical_xs: list[float] = []

        for p in paths:
            items = p.get("items", [])
            for item in items:
                kind = item[0]
                if kind == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 2:  # 水平线
                        horizontal_ys.append((p1.y + p2.y) / 2)
                    elif abs(p1.x - p2.x) < 2:  # 垂直线
                        vertical_xs.append((p1.x + p2.x) / 2)
                elif kind == "re":
                    r = fitz.Rect(item[1])
                    # 矩形边框 → 4 条线
                    if r.height < 3:  # 近似水平
                        horizontal_ys.append(r.y0)
                    if r.width < 3:  # 近似垂直
                        vertical_xs.append(r.x0)

        if len(horizontal_ys) < 2 or len(vertical_xs) < 2:
            return 0.0

        # 检查水平线是否规则间隔
        horizontal_ys = sorted(set(round(y, 1) for y in horizontal_ys))
        vertical_xs = sorted(set(round(x, 1) for x in vertical_xs))

        if len(horizontal_ys) < 2 or len(vertical_xs) < 2:
            return 0.0

        # 计算间隔的变异系数（CV）：CV 越小越规则
        h_gaps = [horizontal_ys[i + 1] - horizontal_ys[i] for i in range(len(horizontal_ys) - 1)]
        v_gaps = [vertical_xs[i + 1] - vertical_xs[i] for i in range(len(vertical_xs) - 1)]

        def _regularity(gaps: list[float]) -> float:
            if not gaps:
                return 0.0
            mean = sum(gaps) / len(gaps)
            if mean == 0:
                return 0.0
            variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
            std = variance**0.5
            cv = std / mean
            # CV < 0.3 → 高度规则 → score ~1.0
            # CV > 1.0 → 不规则 → score ~0.0
            return max(0.0, 1.0 - cv)

        h_reg = _regularity(h_gaps)
        v_reg = _regularity(v_gaps)

        # 综合分数
        return (h_reg + v_reg) / 2


# ---------------------------------------------------------------------------
# ZoneRenderer — 统一渲染输出
# ---------------------------------------------------------------------------


class ZoneRenderer:
    """根据分类结果，统一渲染 zone 内的内容.

    TABLE → find_tables(clip=cluster_bbox) → to_markdown()
    DIAGRAM → get_pixmap(clip=expanded_bbox) + edge-label expansion
    """

    CLIP_PADDING_HORIZONTAL = 15
    CLIP_PADDING_TOP = 15
    CLIP_PADDING_BOTTOM = 15
    EDGE_LABEL_MARGIN = 25.0
    PAGE_RENDER_DPI = 200
    # 同一 zone 内邻近 diagram cluster 的合并距离阈值
    # 过大导致相邻独立 figure 被合并，过小导致同一张图的标签与主体分离
    DIAGRAM_CLUSTER_MERGE_DISTANCE_PT = 40.0
    # 空白图片检测阈值（与 PDFParser._is_low_content_image 一致）
    LOW_CONTENT_THRESHOLD = 0.95

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
        return most_common_count / total > cls.LOW_CONTENT_THRESHOLD

    def render_zone(
        self,
        page: fitz.Page,
        zone: Zone,
        page_rect: fitz.Rect,
        img_dir: Path,
        page_idx: int,
        text_blocks: list[tuple[str, fitz.Rect]],
        header_margin: float = 0.0,
        footer_margin: float = 0.0,
        page_counters: dict[str, int] | None = None,
    ) -> Zone:
        """渲染 zone 内的所有内容，返回更新后的 zone.

        核心原则：
        - TABLE cluster：每个单独渲染为 Markdown（表格）
        - DIAGRAM cluster：Figure zone 内所有 diagram clusters 合并为一个 bbox，
          只渲染 1 张图（借鉴旧算法的 zone-per-image 思想，避免子图重复）
        - RASTER_IMAGE：每个单独处理
        """
        # 1. 渲染 table clusters（每个单独处理）
        for cluster in zone.clusters:
            if cluster.cluster_type == "table":
                self._render_table_cluster(page, zone, cluster)

        # 2. 收集 raster images
        for cluster in zone.clusters:
            if cluster.cluster_type == "raster_image":
                self._render_raster_cluster(page, zone, cluster, img_dir, page_idx, page_counters)

        # 3. Figure/Orphan zone：以 caption 为界，分别合并上下两侧的 diagram clusters
        # 避免 caption 上方的主图与 caption 下方的小装饰/Note 框混为一张图
        # 但 orphan zone 必须已有 caption 配对（被回收）才渲染
        # 同时只渲染"归属"于当前 zone 的 diagram（避免相邻 figure zone 重复）
        diagram_clusters = [
            c for c in zone.clusters
            if c.cluster_type == "diagram"
            and getattr(c, "_assigned_zone", zone) is zone
        ]
        if diagram_clusters:
            if zone.zone_type == "table":
                # table zone 不渲染 diagram（避免与 figure zone 重复）
                pass
            elif zone.zone_type == "orphan" and not zone.caption_text:
                # orphan zone 无 caption 配对 → 不渲染 diagram（避免噪声图片）
                pass
            else:
                # 过滤：caption 上方过远且高度很小的 cluster（通常是上一个 table 的残留边框）
                if zone.caption_bbox:
                    diagram_clusters = [
                        c for c in diagram_clusters
                        if not (
                            c.bbox.y1 < zone.caption_bbox.y0
                            and zone.caption_bbox.y0 - c.bbox.y1 > 50
                            and c.bbox.height < 30
                        )
                    ]
                # 按 caption 分界分组
                groups: list[list[ZoneCluster]] = []
                if zone.caption_bbox:
                    above = [c for c in diagram_clusters if c.bbox.y1 <= zone.caption_bbox.y0]
                    below = [c for c in diagram_clusters if c.bbox.y0 >= zone.caption_bbox.y1]
                    # 跨越 caption 的 cluster（极少见）归入 above
                    crossing = [c for c in diagram_clusters if c not in above and c not in below]
                    above.extend(crossing)
                    merge_dist = ZoneRenderer.DIAGRAM_CLUSTER_MERGE_DISTANCE_PT
                    # above clusters：同样按 gap 拆分，避免 caption 上方不相关内容混入
                    if above:
                        above_sorted = sorted(above, key=lambda c: c.bbox.y0)
                        current_group = [above_sorted[0]]
                        for c in above_sorted[1:]:
                            gap = c.bbox.y0 - current_group[-1].bbox.y1
                            if gap > merge_dist:
                                groups.append(current_group)
                                current_group = [c]
                            else:
                                current_group.append(c)
                        groups.append(current_group)
                    # below clusters：按 gap 拆分为独立 group，避免合并不连续内容
                    if below:
                        below_sorted = sorted(below, key=lambda c: c.bbox.y0)
                        current_group = [below_sorted[0]]
                        for c in below_sorted[1:]:
                            gap = c.bbox.y0 - current_group[-1].bbox.y1
                            if gap > merge_dist:
                                groups.append(current_group)
                                current_group = [c]
                            else:
                                current_group.append(c)
                        groups.append(current_group)
                else:
                    groups.append(diagram_clusters)

                for group in groups:
                    merged_bbox = fitz.Rect(group[0].bbox)
                    for c in group[1:]:
                        merged_bbox |= c.bbox
                    merged_cluster = ZoneCluster(
                        bbox=merged_bbox,
                        cluster_type="diagram",
                        paths=[p for c in group for p in c.paths],
                        caption_text=zone.caption_text,
                        caption_bbox=zone.caption_bbox,
                    )
                    self._render_diagram_cluster(
                        page, zone, merged_cluster, page_rect, img_dir, page_idx,
                        text_blocks, header_margin, footer_margin, page_counters,
                    )

        return zone

    def _render_table_cluster(
        self,
        page: fitz.Page,
        zone: Zone,
        cluster: ZoneCluster,
    ) -> None:
        """在 cluster bbox 内检测表格并转换为 Markdown."""
        bbox = cluster.bbox
        try:
            tabs = page.find_tables(clip=bbox, strategy="lines_strict")
            if tabs and tabs.tables:
                md_parts = []
                for tab in tabs.tables:
                    if tab.row_count < 2 or tab.col_count < 2:
                        continue
                    md = tab.to_markdown(clean=False)
                    if md.strip():
                        md_parts.append(md.strip())
                if md_parts:
                    zone.table_markdown = "\n\n".join(md_parts)
        except Exception as e:
            pnum = page.number + 1 if page.number is not None else 0
            logger.warning(f"Zone table rendering failed on page {pnum}: {e}")

    def _render_diagram_cluster(
        self,
        page: fitz.Page,
        zone: Zone,
        cluster: ZoneCluster,
        page_rect: fitz.Rect,
        img_dir: Path,
        page_idx: int,
        text_blocks: list[tuple[str, fitz.Rect]],
        header_margin: float = 0.0,
        footer_margin: float = 0.0,
        page_counters: dict[str, int] | None = None,
    ) -> None:
        """将 diagram cluster 渲染为图片."""
        bbox = cluster.bbox

        # 过滤：跳过 header/footer 区域的小装饰 cluster
        if header_margin > 0:
            if bbox.y1 < header_margin + 20 and len(cluster.paths) < 5:
                return
        if footer_margin > 0:
            if bbox.y0 > page_rect.y1 - footer_margin - 20 and len(cluster.paths) < 5:
                return
        if bbox.height < 30 and bbox.width < 100 and len(cluster.paths) < 5:
            return

        # 过滤：跳过明显不是 diagram 的 cluster（Note 框、装饰线、caption 下方的元素等）
        # 1. 宽高比极端的条带（但 paths 数量多的是位域图/时序图边框，保留）
        if bbox.height < 35 and bbox.width > page_rect.width * 0.6:
            if len(cluster.paths) < 8:
                return
        if bbox.height < 15 and len(cluster.paths) < 8:
            return
        # 2. 位于 caption 下方的简单框/Note 框（paths 极少）
        if (
            zone.caption_bbox
            and bbox.y0 > zone.caption_bbox.y1 + 10
            and len(cluster.paths) < 3
        ):
            return
        # 3. 位于 caption 下方的条带（paths 多的是时序图/位域图，保留）
        if (
            zone.caption_bbox
            and bbox.y0 > zone.caption_bbox.y1 + 10
            and bbox.height < 45
            and len(cluster.paths) < 15
        ):
            return

        # 用 DiagramClipBuilder 精确计算 clip
        builder = DiagramClipBuilder()
        clip = builder.build_clip(cluster, zone, text_blocks, page_rect)
        if clip is None:
            return

        # 跳过明显是表格/分隔线的 clip（aspect ratio 检查）
        # 但 paths 数量多的是位域图/时序图边框，保留
        aspect = clip.width / clip.height if clip.height > 0 else 999
        if (
            clip.width > page_rect.width * 0.72
            and aspect > 6.5
            and clip.height < page_rect.height * 0.22
            and len(cluster.paths) < 8
        ):
            return
        if clip.width > page_rect.width * 0.92 and clip.height > page_rect.height * 0.82:
            return

        try:
            # 全局序号：使用 page_counters 跨 zone 计数（避免相邻 zone 文件名冲突）
            counters = page_counters if page_counters is not None else {}
            diag_idx = counters.get("diagram", 0) + 1
            counters["diagram"] = diag_idx
            img_path = img_dir / f"page_{page_idx}_diagram_{diag_idx:02d}.jpg"
            zoom = self.PAGE_RENDER_DPI / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip)

            png_buf = io.BytesIO(pix.tobytes("png"))
            pil_img = Image.open(png_buf).convert("RGB")
            pil_img.save(img_path, format="JPEG", quality=Config.PARSER_IMAGE_JPEG_QUALITY)

            rel_path = f"images/{img_path.name}"
            zone.raster_images.append(
                {
                    "type": "image",
                    "y0": clip.y0,
                    "y1": clip.y1,
                    "path": img_path,
                    "rel_path": rel_path,
                    "alt": cluster.caption_text or f"Page {page_idx} Diagram",
                }
            )
        except Exception as e:
            logger.warning(f"Zone diagram rendering failed on page {page_idx}: {e}")

    def _render_raster_cluster(
        self,
        page: fitz.Page,
        zone: Zone,
        cluster: ZoneCluster,
        img_dir: Path,
        page_idx: int,
        page_counters: dict[str, int] | None = None,
    ) -> None:
        """将 raster image cluster 保存为图片文件."""
        bbox = cluster.bbox
        try:
            # 使用 page.get_images + extract_image 获取原始图片数据
            img_list = page.get_images(full=True)
            # 找到与 cluster bbox 最接近的图片
            best_match = None
            best_dist = float("inf")
            for img_info in img_list:
                xref = img_info[0]
                try:
                    bbox_result = page.get_image_bbox(img_info)
                    img_bbox = bbox_result[0] if isinstance(bbox_result, tuple) else bbox_result
                except Exception:
                    continue
                # 计算 bbox 中心点距离
                dist = abs((img_bbox.x0 + img_bbox.x1) / 2 - (bbox.x0 + bbox.x1) / 2)
                dist += abs((img_bbox.y0 + img_bbox.y1) / 2 - (bbox.y0 + bbox.y1) / 2)
                if dist < best_dist:
                    best_dist = dist
                    best_match = (xref, img_bbox)

            counters = page_counters if page_counters is not None else {}
            if best_match is None:
                # 回退：直接渲染 bbox 区域
                img_idx = counters.get("img", 0) + 1
                counters["img"] = img_idx
                img_path = img_dir / f"page_{page_idx}_img_{img_idx}.jpg"
                zoom = self.PAGE_RENDER_DPI / 72
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox)
                png_buf = io.BytesIO(pix.tobytes("png"))
                pil_img = Image.open(png_buf).convert("RGB")
                pil_img.save(img_path, format="JPEG", quality=Config.PARSER_IMAGE_JPEG_QUALITY)
                rel_path = f"images/{img_path.name}"
                zone.raster_images.append(
                    {
                        "type": "image",
                        "y0": bbox.y0,
                        "y1": bbox.y1,
                        "path": img_path,
                        "rel_path": rel_path,
                        "alt": f"Page {page_idx} Image {len(zone.raster_images) + 1}",
                    }
                )
                return

            xref, img_bbox = best_match
            doc = page.parent
            if doc is None:
                return
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))

            # 过滤过小图片
            min_w = 20
            min_h = 20
            if pil_img.width < min_w or pil_img.height < min_h:
                return

            # 过滤空白/低内容图片
            if self._is_low_content_image(pil_img):
                return

            img_idx = counters.get("img", 0) + 1
            counters["img"] = img_idx
            img_path = img_dir / f"page_{page_idx}_img_{img_idx}.jpg"
            if pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")
            pil_img.save(img_path, format="JPEG", quality=Config.PARSER_IMAGE_JPEG_QUALITY)

            rel_path = f"images/{img_path.name}"
            zone.raster_images.append(
                {
                    "type": "image",
                    "y0": img_bbox.y0,
                    "y1": img_bbox.y1,
                    "path": img_path,
                    "rel_path": rel_path,
                    "alt": f"Page {page_idx} Image {img_idx}",
                }
            )
        except Exception as e:
            logger.warning(f"Zone raster rendering failed on page {page_idx}: {e}")


# ---------------------------------------------------------------------------
# DiagramClipBuilder — 精确渲染 clip 边界计算
# ---------------------------------------------------------------------------


class DiagramClipBuilder:
    """从 cluster 的 drawing paths 精确计算渲染 clip 边界.

    6 步线性过滤（prune → barrier → density → edge-label → padding），
    策略参数全部提取为类常量。
    """

    # 步骤 2: Outlier 剪枝
    PRUNE_OUTLIER_THRESHOLD_PT = 250.0

    # 步骤 3: Body-text barrier
    BARRIER_WIDTH_RATIO = 0.5
    BARRIER_CHECK_ABOVE = True
    BARRIER_CHECK_BELOW = True

    # 步骤 4: Text-density guard
    TEXT_DENSITY_THRESHOLD = 0.35
    DENSITY_BAND_PADDING_PT = 20.0

    # 步骤 5: Edge-label expansion
    EDGE_LABEL_MARGIN_PT = 12.0
    EDGE_LABEL_MAX_LENGTH = 60

    # 步骤 6: Padding
    CLIP_PAD_H = 15
    CLIP_PAD_TOP = 0
    CLIP_PAD_BOTTOM = 15

    def build_clip(
        self,
        cluster: ZoneCluster,
        zone: Zone,
        text_blocks: list[tuple[str, fitz.Rect]],
        page_rect: fitz.Rect,
    ) -> fitz.Rect | None:
        """从 cluster paths 计算精确渲染 clip. 返回 None 表示此 cluster 不应渲染."""
        # Step 1: Raw clip from path rects
        path_rects = [
            fitz.Rect(p["rect"]) for p in cluster.paths if p.get("rect")
        ]
        if not path_rects:
            return None
        raw = fitz.Rect(
            min(r.x0 for r in path_rects),
            min(r.y0 for r in path_rects),
            max(r.x1 for r in path_rects),
            max(r.y1 for r in path_rects),
        )

        # Step 2: Prune outlier paths
        center_y = (raw.y0 + raw.y1) / 2
        kept_rects = [
            r for r in path_rects
            if abs((r.y0 + r.y1) / 2 - center_y) <= self.PRUNE_OUTLIER_THRESHOLD_PT
        ]
        if not kept_rects:
            return None
        raw = fitz.Rect(
            min(r.x0 for r in kept_rects),
            min(r.y0 for r in kept_rects),
            max(r.x1 for r in kept_rects),
            max(r.y1 for r in kept_rects),
        )

        # Step 3: Body-text barrier
        if zone.caption_bbox and self._has_body_text_barrier(
            raw, zone.caption_bbox, text_blocks, page_rect
        ):
            return None

        # Step 4: Text-density guard
        if self._is_text_dense(raw, text_blocks, page_rect):
            return None

        # Step 5: Edge-label expansion
        expanded = self._expand_edge_labels(raw, text_blocks, page_rect)

        # Step 6: Padding + clip to page
        clip = fitz.Rect(
            expanded.x0 - self.CLIP_PAD_H,
            expanded.y0 - self.CLIP_PAD_TOP,
            expanded.x1 + self.CLIP_PAD_H,
            expanded.y1 + self.CLIP_PAD_BOTTOM,
        ) & page_rect

        return clip

    # 图注/表格注释等常见非正文格式（允许点后无空格，如 "A.After"、"(1)In"）
    _CALLOUT_PATTERN = re.compile(
        r"^(?:[A-Z]\.\s*|\(\d+\)\s*|Note\s*:\s*|Notes?\s*\(\d+\)\s*)",
        re.IGNORECASE,
    )

    @classmethod
    def _is_body_text(cls, txt: str, rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
        """Heuristic: 判断文本块是否为正文段落（复用 ZoneBuilder 逻辑）."""
        if rect.height > page_rect.height * 0.35:
            return False
        if rect.width > page_rect.width * 0.85 and rect.height > 80:
            return False
        # 排除图注/注释格式（如 "A. After reset...", "(1) In addition..."）
        if cls._CALLOUT_PATTERN.match(txt):
            return False
        width_ratio = rect.width / page_rect.width
        if width_ratio > 0.52 and rect.height > 8 and len(txt) > 45:
            return True
        return False

    def _has_body_text_barrier(
        self,
        raw: fitz.Rect,
        caption_bbox: fitz.Rect,
        text_blocks: list[tuple[str, fitz.Rect]],
        page_rect: fitz.Rect,
    ) -> bool:
        """Caption 与 raw_clip 之间有宽 body_text → True."""
        for txt, rect in text_blocks:
            if not self._is_body_text(txt, rect, page_rect):
                continue
            if rect.width <= page_rect.width * self.BARRIER_WIDTH_RATIO:
                continue
            # Check above caption (raw is above caption)
            if self.BARRIER_CHECK_ABOVE:
                if (
                    raw.y1 < caption_bbox.y0
                    and rect.y0 > raw.y1
                    and rect.y1 < caption_bbox.y0
                ):
                    return True
            # Check below caption (raw is below caption)
            if self.BARRIER_CHECK_BELOW:
                if (
                    raw.y0 > caption_bbox.y1
                    and rect.y0 > caption_bbox.y1
                    and rect.y1 < raw.y0
                ):
                    return True
        return False

    def _is_text_dense(
        self,
        raw: fitz.Rect,
        text_blocks: list[tuple[str, fitz.Rect]],
        page_rect: fitz.Rect,
    ) -> bool:
        """raw_clip 区域内 body_text 面积占比过高 → True."""
        band_y0 = raw.y0 - self.DENSITY_BAND_PADDING_PT
        band_y1 = raw.y1 + self.DENSITY_BAND_PADDING_PT
        band_area = max(1.0, (band_y1 - band_y0) * page_rect.width)
        body_area = sum(
            rect.width * rect.height
            for txt, rect in text_blocks
            if self._is_body_text(txt, rect, page_rect)
            and rect.y0 < band_y1
            and rect.y1 > band_y0
        )
        return body_area / band_area > self.TEXT_DENSITY_THRESHOLD

    def _expand_edge_labels(
        self,
        raw: fitz.Rect,
        text_blocks: list[tuple[str, fitz.Rect]],
        page_rect: fitz.Rect,
    ) -> fitz.Rect:
        """基于 raw_clip 做 edge-label expansion（12pt 范围）."""
        expanded = fitz.Rect(raw)
        for txt, rect in text_blocks:
            # 跳过正文段落和 caption
            if self._is_body_text(txt, rect, page_rect):
                continue
            if len(txt) > self.EDGE_LABEL_MAX_LENGTH:
                continue
            # 跳过覆盖 raw_clip 的大文本块
            intersect = rect & raw
            if (
                intersect
                and not intersect.is_empty
                and intersect.get_area() > raw.get_area() * 0.5
            ):
                continue
            dx = max(raw.x0 - rect.x1, 0, rect.x0 - raw.x1)
            dy = max(raw.y0 - rect.y1, 0, rect.y0 - raw.y1)
            if max(dx, dy) <= self.EDGE_LABEL_MARGIN_PT:
                expanded |= rect
        return expanded


# ---------------------------------------------------------------------------
# OrphanRecycler — 未配对集群回收
# ---------------------------------------------------------------------------


class OrphanRecycler:
    """回收未配对的 orphan zone 内的 significant 集群.

    对于 orphan zone 内的 clusters，尝试在 200pt 范围内找到最近的 figure/table caption，
    将 cluster 回收并重新分类。
    """

    RECYCLE_DISTANCE_PT = 200.0

    @classmethod
    def recycle(
        cls,
        orphan_zones: list[Zone],
        figure_captions: list[tuple[str, fitz.Rect]],
        table_captions: list[tuple[str, fitz.Rect]],
    ) -> list[tuple[int, ZoneCluster]]:
        """回收 orphan clusters，返回需要被追加到对应 zone 的集群列表.

        返回: [(target_zone_index, ZoneCluster), ...]
        """
        recycled: list[tuple[int, ZoneCluster]] = []

        for zone in orphan_zones:
            for cluster in zone.clusters:
                if cluster.cluster_type == "noise":
                    continue

                best_cap: tuple[str, fitz.Rect, str, float] | None = None
                best_dist = float("inf")

                # 搜索 figure captions
                for txt, rect in figure_captions:
                    dist = cls._vertical_distance(cluster.bbox, rect)
                    if dist < best_dist and dist < cls.RECYCLE_DISTANCE_PT:
                        best_dist = dist
                        best_cap = (txt, rect, "figure", dist)

                # 搜索 table captions
                for txt, rect in table_captions:
                    dist = cls._vertical_distance(cluster.bbox, rect)
                    if dist < best_dist and dist < cls.RECYCLE_DISTANCE_PT:
                        best_dist = dist
                        best_cap = (txt, rect, "table", dist)

                if best_cap:
                    cap_text, cap_rect, cap_type, _ = best_cap
                    # 更新 cluster 分类
                    if cap_type == "table":
                        cluster.cluster_type = "table"
                    else:
                        cluster.cluster_type = "diagram"
                    cluster.caption_text = cap_text
                    cluster.caption_bbox = cap_rect

                    # 需要找到这个 caption 对应的 zone（但 zones 尚未构建完毕）
                    # 所以返回一个标记，由调用方处理
                    # 简化：直接返回 cluster，由调用方根据 caption_text 匹配
                    recycled.append((-1, cluster))  # -1 表示需要后续匹配

        return recycled  # type: ignore[return-value]

    @staticmethod
    def _vertical_distance(cluster_bbox: fitz.Rect, caption_rect: fitz.Rect) -> float:
        """计算 cluster 和 caption 的垂直距离."""
        if cluster_bbox.y1 < caption_rect.y0:
            return caption_rect.y0 - cluster_bbox.y1
        elif cluster_bbox.y0 > caption_rect.y1:
            return cluster_bbox.y0 - caption_rect.y1
        else:
            return 0.0


# ---------------------------------------------------------------------------
# UnifiedZoneRecognition — 主入口
# ---------------------------------------------------------------------------


class UnifiedZoneRecognition:
    """UZN 主入口类.

    整合 ZoneBuilder + ZoneClusterDetector + SimpleClassifier + ZoneRenderer + OrphanRecycler，
    提供单页处理接口。
    """

    def __init__(self) -> None:
        """Initialize UZN with all sub-components."""
        self.builder = ZoneBuilder()
        self.detector = ZoneClusterDetector()
        self.classifier = SimpleClassifier()
        self.renderer = ZoneRenderer()
        self.recycler = OrphanRecycler()

    @staticmethod
    def _cluster_has_slant(cluster: ZoneCluster) -> bool:
        """检查 cluster 是否包含斜线（时序图/波形图的上升沿/下降沿特征）."""
        for p in cluster.paths:
            for item in p.get("items", []):
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    if dx > 2 and dy > 2:
                        return True
        return False

    @staticmethod
    def _merge_nearby_clusters(clusters: list[ZoneCluster]) -> list[ZoneCluster]:
        """合并 bbox 邻近的 diagram clusters（距离 < MERGE_DISTANCE_PT）。

        按 y0 排序后，如果当前 cluster 与已合并的最后一个 cluster 在 x 或 y 方向
        上距离很近，则将它们的 bbox 取并集、paths 合并。
        新增：斜线特征差异大时不合并（避免时序图和表格行混为一张图）。
        """
        merge_dist = ZoneRenderer.DIAGRAM_CLUSTER_MERGE_DISTANCE_PT
        sorted_clusters = sorted(clusters, key=lambda c: c.bbox.y0)
        merged: list[ZoneCluster] = []
        for cluster in sorted_clusters:
            if not merged:
                merged.append(cluster)
                continue
            last = merged[-1]
            # 计算两个 bbox 之间的最小距离
            dx = max(last.bbox.x0 - cluster.bbox.x1, 0, cluster.bbox.x0 - last.bbox.x1)
            dy = max(last.bbox.y0 - cluster.bbox.y1, 0, cluster.bbox.y0 - last.bbox.y1)
            # 如果 bbox 有重叠，距离为 0
            if dx < merge_dist and dy < merge_dist:
                # 斜线特征差异大 → 不合并（时序图 vs 表格行）
                has_slant_last = UnifiedZoneRecognition._cluster_has_slant(last)
                has_slant_curr = UnifiedZoneRecognition._cluster_has_slant(cluster)
                if has_slant_last != has_slant_curr:
                    merged.append(cluster)
                    continue
                # 合并：更新 bbox 和 paths
                last.bbox = last.bbox | cluster.bbox
                last.paths = last.paths + cluster.paths
            else:
                merged.append(cluster)
        return merged

    @staticmethod
    def _deduplicate_contained_clusters(clusters: list[ZoneCluster]) -> list[ZoneCluster]:
        """丢弃被其他 diagram cluster 完全包含的小 clusters.

        场景：cluster_drawings 将图的标签/小部件和主图拆成多个 clusters，
        但小 cluster 实际上完全位于大 cluster 的 bbox 范围内。
        保留面积最大的 cluster，丢弃被包含的冗余小 clusters。
        """
        if len(clusters) <= 1:
            return clusters

        # 按面积从大到小排序
        sorted_by_area = sorted(
            clusters, key=lambda c: c.bbox.width * c.bbox.height, reverse=True
        )
        kept: list[ZoneCluster] = []
        for c in sorted_by_area:
            contained = False
            for k in kept:
                # 允许 3pt 容差（浮点精度 + 边框留量）
                if (
                    c.bbox.x0 >= k.bbox.x0 - 3
                    and c.bbox.y0 >= k.bbox.y0 - 3
                    and c.bbox.x1 <= k.bbox.x1 + 3
                    and c.bbox.y1 <= k.bbox.y1 + 3
                ):
                    contained = True
                    break
            if not contained:
                kept.append(c)
        return kept

    def _adjust_figure_zone_by_clusters(
        self, zone: Zone, all_captions: list[tuple[str, fitz.Rect]] | None = None
    ) -> None:
        """根据 diagram clusters 和 caption 的相对位置调整 figure zone 范围.

        核心原则：caption 是硬分割线，figure zone 不应跨越 caption 延伸到其他内容区域。
        - 如果 clusters 主要在 caption 上方 → zone 不向下扩展超过 caption
        - 如果 clusters 主要在 caption 下方 → zone 不向上扩展超过 caption
        - 如果 cluster 跨越 caption → 按 caption 分割为上下两个子 cluster
        """
        if not zone.caption_bbox:
            return

        cap = zone.caption_bbox
        diagram_clusters = [c for c in zone.clusters if c.cluster_type == "diagram"]
        if not diagram_clusters:
            return

        # 1. 分割跨越 caption 的 clusters
        new_diagrams: list[ZoneCluster] = []
        for c in diagram_clusters:
            if c.bbox.y0 < cap.y0 and c.bbox.y1 > cap.y1:
                # cluster 跨越 caption：分割为 above 和 below 两部分
                above_paths = [
                    p for p in c.paths
                    if fitz.Rect(p.get("rect", [0, 0, 0, 0])).y1 <= cap.y0 + 5
                ]
                below_paths = [
                    p for p in c.paths
                    if fitz.Rect(p.get("rect", [0, 0, 0, 0])).y0 >= cap.y1 - 5
                ]
                if above_paths:
                    new_diagrams.append(
                        ZoneCluster(
                            bbox=fitz.Rect(c.bbox.x0, c.bbox.y0, c.bbox.x1, cap.y0),
                            cluster_type="diagram",
                            paths=above_paths,
                        )
                    )
                if below_paths:
                    new_diagrams.append(
                        ZoneCluster(
                            bbox=fitz.Rect(c.bbox.x0, cap.y1, c.bbox.x1, c.bbox.y1),
                            cluster_type="diagram",
                            paths=below_paths,
                        )
                    )
            else:
                new_diagrams.append(c)

        # 替换 zone 中的 diagram clusters
        non_diagram = [c for c in zone.clusters if c.cluster_type != "diagram"]
        zone.clusters = non_diagram + new_diagrams
        diagram_clusters = new_diagrams

        # 2. 按 caption 分界分组
        above = [c for c in diagram_clusters if c.bbox.y1 <= cap.y0 + 5]
        below = [c for c in diagram_clusters if c.bbox.y0 >= cap.y1 - 5]

        if not above and not below:
            return

        # 3. 额外检查：below cluster 是否属于相邻 figure
        # 如果 cluster 位于下一个 figure caption 上方很近的位置，说明它属于下一个 figure
        if below and all_captions:
            my_idx = None
            for idx, (txt, crect) in enumerate(all_captions):
                if abs(crect.y0 - cap.y0) < 1 and abs(crect.y1 - cap.y1) < 1:
                    my_idx = idx
                    break
            if my_idx is not None and my_idx + 1 < len(all_captions):
                _, next_cap_rect = all_captions[my_idx + 1]
                filtered_below = []
                for c in below:
                    # cluster 位于下一个 caption 上方且距离 < 50pt → 可能属于下一个 figure
                    # 但只过滤 paths 极少的小 cluster（避免误杀真正的 diagram）
                    gap = next_cap_rect.y0 - c.bbox.y1
                    if 0 <= gap < 50 and len(c.paths) < 10:
                        continue
                    filtered_below.append(c)
                below = filtered_below

        # 4. 基于 cluster 特征确定保留哪一侧
        # 核心策略：figure caption 应关联 diagram 侧，避开 table-like 侧。
        # table-like = grid_score > 0.7（表格边框/网格结构），即使被分类为 diagram，
        # 也不应影响"选择哪一侧"的决策，避免时序图被表格行挤掉。
        # 但 table-like clusters 仍会被保留渲染（它们可能是 amba_chi 等框图边框）。
        def _is_table_like(c: ZoneCluster) -> bool:
            return SimpleClassifier._compute_grid_score(c.paths) > 0.7

        above_has_real_diagram = any(not _is_table_like(c) for c in above)
        below_has_real_diagram = any(not _is_table_like(c) for c in below)

        below_total = sum(len(c.paths) for c in below)
        above_total = sum(len(c.paths) for c in above)
        keep_above = False
        keep_below = False

        if below and not above:
            keep_below = True
            zone.y0 = max(zone.y0, cap.y0 - 20)
        elif above and not below:
            keep_above = True
            zone.y1 = min(zone.y1, cap.y1 + 20)
        elif above and below:
            if above_has_real_diagram and not below_has_real_diagram:
                keep_above = True
                zone.y1 = min(zone.y1, cap.y1 + 20)
            elif below_has_real_diagram and not above_has_real_diagram:
                keep_below = True
                zone.y0 = max(zone.y0, cap.y0 - 20)
            elif above_has_real_diagram and below_has_real_diagram:
                # 两侧都有 real diagram，选择"纯度"更高的一侧
                above_real = len([c for c in above if not _is_table_like(c)])
                below_real = len([c for c in below if not _is_table_like(c)])
                above_purity = above_real / len(above)
                below_purity = below_real / len(below)
                if above_purity >= below_purity:
                    keep_above = True
                    zone.y1 = min(zone.y1, cap.y1 + 20)
                else:
                    keep_below = True
                    zone.y0 = max(zone.y0, cap.y0 - 20)
            else:
                # 两侧都没有 real diagram，fallback 到 paths 数量
                if above_total > below_total:
                    keep_above = True
                    zone.y1 = min(zone.y1, cap.y1 + 20)
                else:
                    keep_below = True
                    zone.y0 = max(zone.y0, cap.y0 - 20)

        # 5. 只保留选定一侧的 diagram clusters，并标记归属
        kept_diagrams: list[ZoneCluster] = []
        for c in diagram_clusters:
            is_above = c.bbox.y1 <= cap.y0 + 5
            is_below = c.bbox.y0 >= cap.y1 - 5
            if keep_above and is_above:
                c._assigned_zone = zone  # type: ignore[attr-defined]
                kept_diagrams.append(c)
            elif keep_below and is_below:
                c._assigned_zone = zone  # type: ignore[attr-defined]
                kept_diagrams.append(c)
            elif not keep_above and not keep_below:
                # fallback：两侧都不保留时（理论上不会发生），
                # 保留仍在 zone 范围内的 clusters
                if c.bbox.y0 >= zone.y0 - 10 and c.bbox.y1 <= zone.y1 + 10:
                    c._assigned_zone = zone  # type: ignore[attr-defined]
                    kept_diagrams.append(c)

        zone.clusters = non_diagram + kept_diagrams

    def process_page(
        self,
        page: fitz.Page,
        page_idx: int,
        page_rect: fitz.Rect,
        header_margin: float,
        footer_margin: float,
        img_dir: Path,
    ) -> UZNPageResult:
        """处理单页，返回 UZNPageResult."""
        result = UZNPageResult(page_idx=page_idx)

        # 1. 构建兴趣区域
        zones = self.builder.build_zones(page, page_rect, header_margin, footer_margin)

        # 2. 收集页面文本块（用于 edge-label expansion）
        text_dict = cast(dict[str, Any], page.get_text("dict"))
        text_blocks: list[tuple[str, fitz.Rect]] = []
        raw_blocks = text_dict.get("blocks") or []
        for block in raw_blocks:
            if "lines" not in block:
                continue
            txt = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
            if txt:
                text_blocks.append((txt, fitz.Rect(block["bbox"])))

        # 检查页面是否有 caption
        has_captions = any(z.zone_type in ("figure", "table") for z in zones)

        # 只要有 zone（包括 orphan），就执行 cluster detection 和渲染
        if zones:
            orphan_zones: list[Zone] = []
            for zone in zones:
                clusters = self.detector.detect_clusters(page, zone)
                zone.clusters = clusters

                # 4. 分类
                for cluster in zone.clusters:
                    cluster.cluster_type = self.classifier.classify(cluster, zone, page_rect)
                    cluster.caption_text = zone.caption_text
                    cluster.caption_bbox = zone.caption_bbox

                # 4b. 合并同一 zone 内邻近的 diagram clusters（防止同一张图被拆散）
                diagram_clusters = [c for c in zone.clusters if c.cluster_type == "diagram"]
                if len(diagram_clusters) > 1:
                    merged_diagrams = self._merge_nearby_clusters(diagram_clusters)
                    # 4b-ii. 丢弃被大 cluster 完全包含的冗余小 clusters
                    merged_diagrams = self._deduplicate_contained_clusters(merged_diagrams)
                    # 替换 zone 中的 diagram clusters
                    non_diagram = [c for c in zone.clusters if c.cluster_type != "diagram"]
                    zone.clusters = non_diagram + merged_diagrams

                # 4c. 扩展 zone bbox 以包含所有 clusters（确保 _strip_zone_text_blocks 正确过滤）
                if zone.clusters:
                    min_y = min(c.bbox.y0 for c in zone.clusters)
                    max_y = max(c.bbox.y1 for c in zone.clusters)
                    zone.y0 = min(zone.y0, min_y)
                    zone.y1 = max(zone.y1, max_y)

                if zone.zone_type == "orphan":
                    orphan_zones.append(zone)

            # 5. Orphan 回收（仅在 has_captions 时执行）
            if has_captions and orphan_zones:
                figure_captions = [
                    (z.caption_text, z.caption_bbox)
                    for z in zones
                    if z.zone_type == "figure" and z.caption_bbox
                ]
                table_captions = [
                    (z.caption_text, z.caption_bbox)
                    for z in zones
                    if z.zone_type == "table" and z.caption_bbox
                ]

                if figure_captions or table_captions:
                    recycled_clusters = self.recycler.recycle(
                        orphan_zones, figure_captions, table_captions
                    )  # type: ignore[arg-type]
                    for _, cluster in recycled_clusters:
                        # 从 orphan zone 中移除被回收的 cluster
                        for oz in orphan_zones:
                            oz.clusters = [c for c in oz.clusters if c is not cluster]
                        # 根据 caption_text 匹配到对应 zone
                        for zone in zones:
                            if zone.caption_text == cluster.caption_text:
                                # 避免重复添加相同 bbox 的 cluster
                                already_exists = any(
                                    abs(c.bbox.x0 - cluster.bbox.x0) < 1
                                    and abs(c.bbox.y0 - cluster.bbox.y0) < 1
                                    and abs(c.bbox.x1 - cluster.bbox.x1) < 1
                                    and abs(c.bbox.y1 - cluster.bbox.y1) < 1
                                    for c in zone.clusters
                                )
                                if not already_exists:
                                    zone.clusters.append(cluster)
                                    # 扩展 zone bbox 以包含回收的 cluster
                                    zone.y0 = min(zone.y0, cluster.bbox.y0)
                                    zone.y1 = max(zone.y1, cluster.bbox.y1)
                                    # 对受影响 zone 重新 merge/dedup diagram clusters
                                    dc = [c for c in zone.clusters if c.cluster_type == "diagram"]
                                    if len(dc) > 1:
                                        md = self._merge_nearby_clusters(dc)
                                        md = self._deduplicate_contained_clusters(md)
                                        nd = [
                                            c for c in zone.clusters
                                            if c.cluster_type != "diagram"
                                        ]
                                        zone.clusters = nd + md
                                break

            # 5a. figure zone: 根据 diagram clusters 和 caption 的相对位置调整 zone 范围
            # 必须在 orphan 回收之后执行（回收可能扩展 zone 范围）
            # 收集所有 figure captions 用于相邻 caption 判断
            all_figure_captions = sorted(
                [
                    (z.caption_text, z.caption_bbox)
                    for z in zones
                    if z.zone_type == "figure" and z.caption_bbox
                ],
                key=lambda x: x[1].y0,
            )
            for zone in zones:
                if zone.zone_type == "figure" and zone.caption_bbox:
                    self._adjust_figure_zone_by_clusters(zone, all_figure_captions)

            # 5b. 为重叠 zone 中的 diagram cluster 找到"归属" caption
            # 避免相邻 figure zone 重复渲染同一个 diagram（如 sprui07 p209）
            # 注意：_adjust_figure_zone_by_clusters 已设置大部分 cluster 的 _assigned_zone，
            # 此处仅补充处理未被覆盖的 cluster。
            figure_zones = [z for z in zones if z.zone_type == "figure" and z.caption_bbox]
            if len(figure_zones) > 1:
                for zone in zones:
                    for cluster in zone.clusters:
                        if cluster.cluster_type != "diagram":
                            continue
                        # 跳过已由 _adjust_figure_zone_by_clusters 标记的 cluster
                        if getattr(cluster, "_assigned_zone", None) is not None:
                            continue
                        cy = (cluster.bbox.y0 + cluster.bbox.y1) / 2
                        best_zone = zone
                        best_dist = float("inf")
                        for fz in figure_zones:
                            cap_cy = (fz.caption_bbox.y0 + fz.caption_bbox.y1) / 2
                            dist = abs(cy - cap_cy)
                            if dist < best_dist:
                                best_dist = dist
                                best_zone = fz
                        # 标记归属（通过引用比较）
                        cluster._assigned_zone = best_zone  # type: ignore[attr-defined]

            # 6. 渲染（跨 zone 共享计数器，避免文件名冲突）
            page_counters: dict[str, int] = {}
            for zone in zones:
                self.renderer.render_zone(
                    page, zone, page_rect, img_dir, page_idx, text_blocks,
                    header_margin, footer_margin, page_counters,
                )
                result.zones.append(zone)
                result.all_image_paths.extend([r["path"] for r in zone.raster_images])
        else:
            result.zones = zones

        return result
