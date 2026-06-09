# UZN Boundary Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `DiagramClipBuilder` to `unified_zone_recognition.py` to tighten diagram rendering clip boundaries by fusing old `_find_figure_regions` boundary detection strategies.

**Architecture:** A new `DiagramClipBuilder` class performs a 6-step linear filter on cluster paths to compute precise clip rects. `ZoneRenderer._render_diagram_cluster` replaces its existing edge-label expansion + padding logic with a call to `DiagramClipBuilder`. ZoneBuilder is untouched.

**Tech Stack:** Python 3.13, PyMuPDF (fitz), PIL

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/parsers/unified_zone_recognition.py` | Modify | Add `DiagramClipBuilder` class; replace clip logic in `_render_diagram_cluster` |
| `scripts/tests/test_unified_zone_recognition.py` | Create | Unit tests for `DiagramClipBuilder` with mocked fitz objects |

---

## Task 1: Add `DiagramClipBuilder` class

**Files:**
- Modify: `scripts/parsers/unified_zone_recognition.py` (after `ZoneRenderer` class, before `OrphanRecycler`)

- [ ] **Step 1.1: Write the class skeleton with strategy constants**

```python
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
```

- [ ] **Step 1.2: Implement `build_clip` main method**

```python
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
```

- [ ] **Step 1.3: Implement helper methods**

```python
    @staticmethod
    def _is_body_text(txt: str, rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
        """Heuristic: 判断文本块是否为正文段落（复用 ZoneBuilder 逻辑）."""
        if rect.height > page_rect.height * 0.35:
            return False
        if rect.width > page_rect.width * 0.85 and rect.height > 80:
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
        """caption 与 raw_clip 之间有宽 body_text → True."""
        for txt, rect in text_blocks:
            if not self._is_body_text(txt, rect, page_rect):
                continue
            if rect.width <= page_rect.width * self.BARRIER_WIDTH_RATIO:
                continue
            # Check above caption (raw is above caption)
            if self.BARRIER_CHECK_ABOVE:
                if raw.y1 < caption_bbox.y0 and rect.y0 > raw.y1 and rect.y1 < caption_bbox.y0:
                    return True
            # Check below caption (raw is below caption)
            if self.BARRIER_CHECK_BELOW:
                if raw.y0 > caption_bbox.y1 and rect.y0 > caption_bbox.y1 and rect.y1 < raw.y0:
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
            and rect.y0 < band_y1 and rect.y1 > band_y0
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
            if intersect and not intersect.is_empty and intersect.get_area() > raw.get_area() * 0.5:
                continue
            dx = max(raw.x0 - rect.x1, 0, rect.x0 - raw.x1)
            dy = max(raw.y0 - rect.y1, 0, rect.y0 - raw.y1)
            if max(dx, dy) <= self.EDGE_LABEL_MARGIN_PT:
                expanded |= rect
        return expanded
```

- [ ] **Step 1.4: Run ruff/pyright check**

Run: `venv/bin/ruff check scripts/parsers/unified_zone_recognition.py`
Run: `venv/bin/pyright scripts/parsers/unified_zone_recognition.py`
Expected: No errors

---

## Task 2: Integrate into `ZoneRenderer._render_diagram_cluster`

**Files:**
- Modify: `scripts/parsers/unified_zone_recognition.py` (`_render_diagram_cluster` method, ~line 983-1119)

- [ ] **Step 2.1: Remove old edge-label expansion and padding logic**

Delete the block from `# Edge-label expansion` (~line 1032) through the `clip = (...)` block (~line 1078), keeping the aspect-ratio guards after it.

- [ ] **Step 2.2: Insert `DiagramClipBuilder` call**

After the existing pre-filter checks (height, paths count, header/footer, etc.), add:

```python
        # 用 DiagramClipBuilder 精确计算 clip
        builder = DiagramClipBuilder()
        clip = builder.build_clip(cluster, zone, text_blocks, page_rect)
        if clip is None:
            return
```

- [ ] **Step 2.3: Verify aspect-ratio guards still apply to the new clip**

The existing aspect-ratio and full-page guards (lines ~1082-1091) should remain after the new clip computation and continue to operate on `clip`.

- [ ] **Step 2.4: Run ruff/pyright check**

Run: `venv/bin/ruff check scripts/parsers/unified_zone_recognition.py`
Run: `venv/bin/pyright scripts/parsers/unified_zone_recognition.py`
Expected: No errors

---

## Task 3: Regression tests

**Files:**
- Run: `scripts/tests/`

- [ ] **Step 3.1: Run full test suite**

Run: `PYTHONPATH=scripts .venv/bin/python -m pytest scripts/tests/ -v`
Expected: 382/382 pass (2 skipped is normal)

- [ ] **Step 3.2: If any test fails, fix before proceeding**

Most likely failures: tests that assert on exact image dimensions or counts in `test_pdf_parser.py` or UZN-related tests. Update expected values if the new behavior is correct.

---

## Task 4: Validation on degraded pages

- [ ] **Step 4.1: Re-parse 4 documents with the new code**

Run a quick re-parse of tms320f28335, amba_chi, dc_ug, sprui07.

- [ ] **Step 4.2: Manually inspect 6 degraded pages**

Compare extracted images for:
- tms320f28335 p50, p160, p57, p62
- amba_chi p107, p166

Expected improvements:
- p50: no header text in image
- p160: image not full-page-wide
- p57/p62: table rows not extracted as separate images
- p107: sub-diagrams not merged into one
- p166: less body text混入

---

## Task 5: Commit

- [ ] **Step 5.1: Commit the changes**

```bash
git add scripts/parsers/unified_zone_recognition.py
git commit -m "feat(uzn): fuse old boundary detection into DiagramClipBuilder

- Add DiagramClipBuilder with 6-step linear filter
  (prune → barrier → density → edge-label → padding)
- X boundary from path rects (not full page width)
- Y filtering via body-text barrier + density guard + outlier prune
- Edge-label 12pt based on raw_clip (not full-width zone)
- Top padding = 0 to prevent header inclusion
- Strategy params extracted as class constants
- Fixes 6 categories of extraction degradation"
```
