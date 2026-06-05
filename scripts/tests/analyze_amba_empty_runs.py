"""AMBA CHI 文档 find_tables 空跑页面详细分析脚本。

按空跑原因分类：
- Type A: find_tables() 完全未检测到任何表格结构
- Type B: 检测到表格但 row_count < 2 或 col_count < 2
- Type C: 通过行列过滤但 to_markdown() 输出为空
- Type D: 通过所有过滤但表格被 diagram 区域保护跳过（本脚本不检测）

输出：JSON 报告 + 分类统计 + 代表性页面截图
"""

import json
import re
import time
from pathlib import Path
from typing import Any, cast

import fitz

AMBA_DOC = Path("/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/IHI0050G_amba_chi_architecture_spec.pdf")
OUTPUT_DIR = Path("/tmp/amba_empty_run_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_CAPTION_RE = r"^(Table|表)\s*[A-Z]?\d+(?:[-\.]\d+)?(?:[:\.]\s*\S|\s+[A-Z]\S*)"
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2


def _get_page_text_blocks(page: fitz.Page) -> list[dict]:
    """提取页面文本块。"""
    text_dict = cast(dict[str, Any], page.get_text("dict"))
    blocks = []
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        text = "".join(s["text"] for line in block["lines"] for s in line["spans"]).strip()
        if text:
            blocks.append({"text": text, "bbox": block["bbox"]})
    return blocks


def analyze_page(doc: fitz.Document, page_num: int) -> dict:
    """分析单个页面的表格检测情况。"""
    page = doc.load_page(page_num)

    text_blocks = _get_page_text_blocks(page)
    captions = [
        (b["text"], fitz.Rect(b["bbox"]))
        for b in text_blocks
        if re.match(TABLE_CAPTION_RE, b["text"])
    ]

    if not captions:
        return {"has_caption": False}

    result = {
        "has_caption": True,
        "captions": [c[0] for c in captions],
        "caption_bboxes": [(c[1].x0, c[1].y0, c[1].x1, c[1].y1) for c in captions],
    }

    t0 = time.perf_counter()
    tabs = page.find_tables(strategy="lines_strict")
    find_tables_ms = (time.perf_counter() - t0) * 1000

    result["find_tables_ms"] = round(find_tables_ms, 2)

    if tabs is None or not tabs.tables:
        result["type"] = "A"
        result["reason"] = "find_tables returned no tables"
        return result

    detected = []
    for tab in tabs.tables:
        detected.append(
            {
                "bbox": tuple(tab.bbox),
                "rows": tab.row_count,
                "cols": tab.col_count,
            }
        )
    result["detected_tables"] = detected

    # 检查是否全部因行列过滤被丢弃
    filtered_out = [t for t in detected if t["rows"] < TABLE_MIN_ROWS or t["cols"] < TABLE_MIN_COLS]
    passed_filter = [
        t for t in detected if t["rows"] >= TABLE_MIN_ROWS and t["cols"] >= TABLE_MIN_COLS
    ]

    if not passed_filter:
        result["type"] = "B"
        result["reason"] = f"all {len(detected)} tables filtered by row/col"
        result["filtered_tables"] = filtered_out
        return result

    # 检查 to_markdown
    empty_md = []
    non_empty_md = []
    for tab in tabs.tables:
        if tab.row_count < TABLE_MIN_ROWS or tab.col_count < TABLE_MIN_COLS:
            continue
        md = tab.to_markdown(clean=False)
        if not md or not md.strip():
            empty_md.append({"bbox": tuple(tab.bbox), "rows": tab.row_count, "cols": tab.col_count})
        else:
            non_empty_md.append(
                {
                    "bbox": tuple(tab.bbox),
                    "rows": tab.row_count,
                    "cols": tab.col_count,
                    "md_len": len(md.strip()),
                }
            )

    if not non_empty_md:
        result["type"] = "C"
        result["reason"] = "all passed-filter tables have empty to_markdown"
        result["empty_md_tables"] = empty_md
        return result

    result["type"] = "OK"
    result["valid_tables"] = non_empty_md
    return result


def render_page_screenshot(
    doc: fitz.Document, page_num: int, out_path: Path, annotations: list | None = None
):
    """渲染页面截图，可选叠加标注框。"""
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=150)

    if annotations:
        # 如果需要在截图上画框，可以用 PIL
        from PIL import Image, ImageDraw

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(img)
        page_rect = page.rect
        scale_x = pix.width / page_rect.width
        scale_y = pix.height / page_rect.height

        for ann in annotations:
            bbox = ann["bbox"]
            x0 = bbox[0] * scale_x
            y0 = bbox[1] * scale_y
            x1 = bbox[2] * scale_x
            y1 = bbox[3] * scale_y
            color = ann.get("color", "red")
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        img.save(str(out_path))
    else:
        pix.save(str(out_path))


def main():
    print(f"分析文档: {AMBA_DOC}")
    print(f"输出目录: {OUTPUT_DIR}")

    doc = fitz.open(str(AMBA_DOC))
    total_pages = len(doc)
    print(f"总页数: {total_pages}")

    results = {}
    type_counts = {"A": 0, "B": 0, "C": 0, "OK": 0, "no_caption": 0}
    type_times = {"A": 0.0, "B": 0.0, "C": 0.0, "OK": 0.0}

    for page_num in range(total_pages):
        page_idx = page_num + 1
        result = analyze_page(doc, page_num)
        results[page_idx] = result

        if not result["has_caption"]:
            type_counts["no_caption"] += 1
        else:
            t = result.get("find_tables_ms", 0)
            typ = result.get("type", "OK")
            type_counts[typ] = type_counts.get(typ, 0) + 1
            type_times[typ] = type_times.get(typ, 0.0) + t

        if page_idx % 50 == 0:
            print(f"  已处理 {page_idx}/{total_pages} 页...")

    doc.close()

    # 保存 JSON 报告
    report_path = OUTPUT_DIR / "amba_analysis_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON 报告已保存: {report_path}")

    # 打印分类统计
    print("\n" + "=" * 60)
    print("【分类统计】")
    print("=" * 60)
    caption_pages = total_pages - type_counts["no_caption"]
    print(f"有 caption 的页面: {caption_pages} / {total_pages}")
    for lbl, key in [
        ("Type A (find_tables 未检测到表格)", "A"),
        ("Type B (行列过滤丢弃)", "B"),
        ("Type C (to_markdown 为空)", "C"),
        ("OK   (产出有效表格)", "OK"),
    ]:
        t = type_times[key]
        print(f"  {lbl}: {type_counts[key]:>3} 页, 累计 {t:.0f}ms")
    print(f"  无 caption:                            {type_counts['no_caption']:>3} 页")

    empty_runs = type_counts["A"] + type_counts["B"] + type_counts["C"]
    empty_time = type_times["A"] + type_times["B"] + type_times["C"]
    total_time = sum(type_times.values())
    pct = empty_time / total_time * 100 if total_time > 0 else 0
    print(f"\n空跑总计: {empty_runs} 页, 累计 {empty_time:.0f}ms ({pct:.1f}%)")

    # 选取代表性页面并渲染
    print("\n渲染代表性页面截图...")
    doc = fitz.open(str(AMBA_DOC))

    for typ in ["A", "B", "C"]:
        pages = [p for p, r in results.items() if r.get("type") == typ]
        if not pages:
            continue
        # 每类型选最多 5 页（首、中、尾）
        selected = [pages[0]]
        if len(pages) > 2:
            selected.append(pages[len(pages) // 2])
        if len(pages) > 1:
            selected.append(pages[-1])
        selected = sorted(set(selected))[:5]

        for p in selected:
            r = results[p]
            out_path = OUTPUT_DIR / f"type{typ}_page_{p:03d}.png"

            # 构建标注框
            annotations = []
            # caption bbox
            for bbox in r.get("caption_bboxes", []):
                annotations.append({"bbox": bbox, "color": "blue"})
            # 检测到的表格 bbox
            for t in r.get("detected_tables", []):
                annotations.append({"bbox": t["bbox"], "color": "red"})
            for t in r.get("filtered_tables", []):
                annotations.append({"bbox": t["bbox"], "color": "orange"})
            for t in r.get("empty_md_tables", []):
                annotations.append({"bbox": t["bbox"], "color": "purple"})
            for t in r.get("valid_tables", []):
                annotations.append({"bbox": t["bbox"], "color": "green"})

            render_page_screenshot(doc, p - 1, out_path, annotations if annotations else None)
            print(f"  已渲染: {out_path.name} (caption: {r['captions']})")

    doc.close()
    print(f"\n全部截图已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
