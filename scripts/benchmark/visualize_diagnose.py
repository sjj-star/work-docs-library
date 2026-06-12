"""为诊断JSON生成带标注的页面可视化图。

用法：
    python visualize_diagnose.py <diagnose_json> <output_image>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from PIL import Image, ImageDraw


def visualize_diagnose(json_path: str, output_path: str, dpi: int = 150):
    """读取诊断 JSON 并在对应 PDF 页面上绘制标注图."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pdf_path = data["pdf_path"]
    page_1idx = data["page_1idx"]

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_1idx - 1)

    # 渲染页面为 pixmap
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")

    # 坐标转换函数: PDF点 -> 像素
    def pt2px(val):
        return val * zoom

    # 1. 标注 diagram_regions: 红色粗框
    for bbox in data.get("diagram_regions", []):
        x0, y0, x1, y1 = [pt2px(v) for v in bbox]
        draw.rectangle([x0, y0, x1, y1], outline="#FF0000", width=4)
        # 左上角标签
        draw.text((x0 + 2, y0 + 2), "diagram_region", fill="#FF0000")

    # 2. 标注 table_caption: 蓝色高亮底部线
    for block in data.get("classified_blocks", []):
        if block["category"] == "table_caption":
            x0, y0, x1, y1 = [pt2px(v) for v in block["bbox"]]
            draw.line([(x0, y1), (x1, y1)], fill="#0000FF", width=4)
            draw.text((x0, y0 - 12), f"table_caption: {block['text'][:50]}", fill="#0000FF")

    # 3. 标注 figure_caption: 绿色高亮底部线
    for block in data.get("classified_blocks", []):
        if block["category"] == "figure_caption":
            x0, y0, x1, y1 = [pt2px(v) for v in block["bbox"]]
            draw.line([(x0, y1), (x1, y1)], fill="#00AA00", width=4)
            draw.text((x0, y0 - 12), f"figure_caption: {block['text'][:50]}", fill="#00AA00")

    # 4. 标注 find_tables bbox: 黄色虚线框
    for tab in data.get("find_tables_raw", []):
        bbox = tab["bbox"]
        x0, y0, x1, y1 = [pt2px(v) for v in bbox]
        # 绘制虚线矩形
        _draw_dashed_rect(draw, x0, y0, x1, y1, fill="#FFCC00", width=3, dash=8)
        label = f"find_table ({tab.get('row_count', '?')}x{tab.get('col_count', '?')})"
        draw.text((x0 + 2, y0 + 2), label, fill="#CC9900")

    # 5. 标注 detected_tables_after_filter (如果有)
    for tab in data.get("detected_tables_after_filter", []):
        bbox = tab["bbox"]
        x0, y0, x1, y1 = [pt2px(v) for v in bbox]
        draw.rectangle([x0, y0, x1, y1], outline="#FF6600", width=2)
        draw.text((x0 + 2, y1 - 14), "filtered_table", fill="#FF6600")

    # 6. raw drawings: 半透明红色小矩形覆盖
    raw_drawings = data.get("raw_drawings_sample", [])
    for bbox in raw_drawings[:200]:  # 限制数量避免过度覆盖
        x0, y0, x1, y1 = [pt2px(v) for v in bbox]
        # 过滤掉非常小的噪声
        if abs(x1 - x0) < 1 and abs(y1 - y0) < 1:
            continue
        draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 40), outline=None)

    # 添加图例
    legend_y = 10
    legend_items = [
        ("diagram_region (red box)", "#FF0000"),
        ("table_caption (blue underline)", "#0000FF"),
        ("figure_caption (green underline)", "#00AA00"),
        ("find_tables bbox (yellow dashed)", "#FFCC00"),
        ("raw_drawings (transparent red)", "#FF0000"),
    ]
    for text, color in legend_items:
        draw.text((10, legend_y), text, fill=color)
        legend_y += 14

    img.save(output_path, quality=95)
    print(f"Saved annotated image to {output_path} ({img.width}x{img.height})")
    doc.close()


def _draw_dashed_rect(draw, x0, y0, x1, y1, fill, width=2, dash=8):
    """Pillow 绘制虚线矩形。"""

    def _dash_line(start, end, horizontal=True):
        x1c, y1c = start
        x2c, y2c = end
        if horizontal:
            cur = x1c
            while cur < x2c:
                seg_end = min(cur + dash, x2c)
                draw.line([(cur, y1c), (seg_end, y1c)], fill=fill, width=width)
                cur += dash * 2
        else:
            cur = y1c
            while cur < y2c:
                seg_end = min(cur + dash, y2c)
                draw.line([(x1c, cur), (x1c, seg_end)], fill=fill, width=width)
                cur += dash * 2

    _dash_line((x0, y0), (x1, y0), horizontal=True)
    _dash_line((x0, y1), (x1, y1), horizontal=True)
    _dash_line((x0, y0), (x0, y1), horizontal=False)
    _dash_line((x1, y0), (x1, y1), horizontal=False)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_diagnose.py <diagnose_json> <output_image>")
        sys.exit(1)
    visualize_diagnose(sys.argv[1], sys.argv[2])
