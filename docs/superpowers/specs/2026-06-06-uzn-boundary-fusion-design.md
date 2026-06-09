# UZN 边界融合设计：融合旧算法边界检测策略

> 设计目标：修复图片提取中 6 类退化问题，同时保留 UZN 架构的 cluster detection / 分类 / orphan 回收优势。

---

## 1. 问题诊断

当前 UZN 渲染退化根因：ZoneRenderer 以**全页宽的 zone** 为基础做 edge-label expansion，且缺少旧算法的 4 道防护。

| 退化类型 | 根因 | 旧算法对策 |
|---------|------|-----------|
| Zone x 全页宽 | ZoneBuilder 设 `x0=page_rect.x0, x1=page_rect.x1` | X 边界由 drawings min/max x 决定 |
| Zone y 过度扩展 | body_text/text_block 是**软停止**，zone 向上穿过正文直达 header | body_text barrier + text-density guard + prune outlier |
| 表格误提取为图 | cluster.bbox 包含表格边框 paths | density guard 过滤高密度正文 gap |
| 大图被错误切分 | 多 cluster 被 _merge_nearby_clusters 合并 | prune outlier 移除远离主 cluster 的 decorations |
| 页眉混入 | zone y0 过低，顶部 padding 15pt 把页眉包含进来 | 顶部 padding=0，prune 过滤 header 区域小 decoration |
| 正文混入 | 25pt edge-label + 全宽基础导致连锁卷进 | 12pt edge-label + raw_clip 基础 |

---

## 2. 核心设计原则

1. **搜索范围与渲染边界解耦**：ZoneBuilder 构建的 zone 是"搜索范围"（可以很大），最终渲染 clip 由独立的边界检测机制精确计算。
2. **线性复杂度**：新增机制只遍历 cluster.paths 和 text_blocks 各一次，O(n)。
3. **机制与策略分离**：边界检测的"框架"是固定的 6 步流程，每步的阈值/行为是"策略参数"，全部提取为类常量。
4. **最小侵入**：ZoneBuilder 不改动（保持搜索范围），ZoneRenderer 仅替换 `_render_diagram_cluster` 的 clip 计算逻辑。

---

## 3. 机制架构：DiagramClipBuilder

新增独立类 `DiagramClipBuilder`，职责唯一：从 cluster 的 drawing paths 精确计算渲染 clip。

```
输入: cluster (paths + bbox), zone (caption), text_blocks, page_rect
输出: fitz.Rect | None  (None = 此 cluster 不应渲染为图片)

流程（6 步，线性）：
  1. RawClip — paths rect 的 min/max x/y
  2. PruneOutlier — 移除远离主 cluster 的 paths，重新计算 raw_clip
  3. BodyTextBarrier — caption 与 raw_clip 之间有宽 body_text → return None
  4. TextDensityGuard — raw_clip 区域内 body_text 面积占比过高 → return None
  5. EdgeLabelExpansion — 基于 raw_clip（非全宽），12pt 范围内收集标签文本
  6. Padding — 水平 15pt，顶部 0pt，底部 15pt，clip 到 page_rect
```

### 3.1 与旧算法的策略映射

| 旧算法策略 | 在 DiagramClipBuilder 中的实现 | 策略参数 |
|-----------|------------------------------|---------|
| `_prune_outlier_drawings` | 步骤 2：paths 的 centroid 距主 cluster center_y > 阈值则移除 | `PRUNE_OUTLIER_THRESHOLD_PT = 250.0` |
| `_has_body_text_barrier` | 步骤 3：caption 与 raw_clip 之间有宽度 > 0.5 页宽的 body_text → discard | `BARRIER_WIDTH_RATIO = 0.5` |
| `_gap_is_text_heavy` | 步骤 4：raw_clip 区域内 body_text 面积 / 区域面积 > 阈值 → discard | `TEXT_DENSITY_THRESHOLD = 0.35` |
| Edge-label expansion | 步骤 5：基于 raw_clip，距离 ≤ 12pt 才包含，排除 body_text/caption | `EDGE_LABEL_MARGIN_PT = 12.0` |
| Clip padding | 步骤 6：水平 15pt，底部 15pt，**顶部 0pt** | `CLIP_PAD_H = 15, CLIP_PAD_TOP = 0, CLIP_PAD_BOTTOM = 15` |

### 3.2 关键设计决策

**Q: 为什么不修改 ZoneBuilder 的 Y 扩展策略？**

A: Zone 的 Y 范围是"搜索范围"，需要足够大以捕获 caption 上方/下方的所有 drawings。如果缩小 zone Y，可能导致 cluster detection 遗漏 orphan 或跨页图形。真正的边界收紧应在渲染阶段，由 paths 本身决定（类似旧算法"先找 drawings，再定边界"）。

**Q: 为什么顶部 padding = 0？**

A: 旧算法顶部不 padding，因为 diagram 的顶部通常是线条/边框的上边缘，padding 会把上方的正文/页眉卷进来。tms320f28335 p50 的退化正是顶部 padding 15pt 导致页眉混入。

**Q: 为什么 edge-label margin 从 25pt 改回 12pt？**

A: 25pt 是在全页宽 zone 基础上的"补偿"，实际效果是把大量正文卷进来。改为 12pt + raw_clip 基础后，只有真正紧邻图形的标签会被包含（如 amba_chi p166 的 DataID/CCID 标签仍在 12pt 范围内）。

---

## 4. ZoneRenderer 改动

`_render_diagram_cluster` 的 clip 计算部分完全替换为 `DiagramClipBuilder`：

```python
def _render_diagram_cluster(self, ...):
    # 保留现有的前序过滤（高度、paths 数量、header/footer 等）
    # ... existing filters ...
    
    # 替换：用 DiagramClipBuilder 精确计算 clip
    builder = DiagramClipBuilder()
    clip = builder.build_clip(cluster, zone, text_blocks, page_rect)
    if clip is None:
        return  # 被 barrier/density guard 过滤
    
    # 保留现有的 aspect ratio / full-page 过滤
    # ... existing aspect checks ...
    
    # 渲染 ...
```

原有的 edge-label expansion 代码（~60 行）被移除，替换为 `DiagramClipBuilder.build_clip()` 调用。

---

## 5. 策略参数默认值

所有参数定义为 `DiagramClipBuilder` 的类常量，支持子类覆盖：

```python
class DiagramClipBuilder:
    # 步骤 2: Outlier 剪枝
    PRUNE_OUTLIER_THRESHOLD_PT = 250.0
    
    # 步骤 3: Body-text barrier
    BARRIER_WIDTH_RATIO = 0.5
    BARRIER_CHECK_ABOVE = True   # 检查 caption 上方
    BARRIER_CHECK_BELOW = True   # 检查 caption 下方
    
    # 步骤 4: Text-density guard
    TEXT_DENSITY_THRESHOLD = 0.35
    DENSITY_BAND_PADDING_PT = 20.0  # 检查带在 raw_clip 上下扩展 20pt
    
    # 步骤 5: Edge-label expansion
    EDGE_LABEL_MARGIN_PT = 12.0
    EDGE_LABEL_MAX_LENGTH = 60      # 远距离(>margin)文本长度限制
    
    # 步骤 6: Padding
    CLIP_PAD_H = 15
    CLIP_PAD_TOP = 0
    CLIP_PAD_BOTTOM = 15
```

---

## 6. 预期修复效果

| 退化页面 | 退化类型 | 修复机制 |
|---------|---------|---------|
| tms320f28335 p50 | 页眉混入 | 顶部 padding=0 + prune outlier |
| tms320f28335 p160 | 正文混入（大图） | raw_clip x 边界 + density guard |
| tms320f28335 p57/p62 | 表格误提取 | body-text barrier + density guard |
| amba_chi p107 | 多子图混为一张 | prune outlier（子图距离主图 >250pt） |
| amba_chi p166/p167 | 正文混入 | raw_clip x 边界 + 12pt edge-label |

---

## 7. 测试计划

1. **回归测试**：382 个现有测试必须全部通过。
2. **关键页面验证**：对 6 个退化页面重新提取，人工比对渲染结果。
3. **全量回归**：4 个文档全量解析，对比图片数量和 Markdown 质量。

---

## 8. 文件变更

| 文件 | 变更 | 行数估算 |
|------|------|---------|
| `unified_zone_recognition.py` | 新增 `DiagramClipBuilder` 类；修改 `_render_diagram_cluster` | ~150 新增，~60 删除 |

---

*设计待用户批准后可进入 writing-plans 阶段。*
