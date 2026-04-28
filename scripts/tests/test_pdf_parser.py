"""test_pdf_parser 模块."""

from pathlib import Path

import fitz
import pytest
from parsers.pdf_parser import PDFParser


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# parse() 接口测试（新接口：返回 (markdown_text, image_paths)）
# ---------------------------------------------------------------------------


def test_parse_returns_tuple(tmp_path):
    """Test parse returns tuple."""
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Page one content"])
    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert isinstance(md_text, str)
    assert isinstance(img_paths, list)
    assert all(isinstance(p, Path) for p in img_paths)


def test_parse_text_only(tmp_path):
    """Test parse text only."""
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Page one content", "Page two content"])
    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert "Page one content" in md_text
    assert "Page two content" in md_text
    assert len(img_paths) == 0


def test_parse_with_toc(tmp_path):
    """Test parse with toc."""
    pdf_path = tmp_path / "toc.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.set_toc(
        [
            (1, "Chapter 1", 1),
            (1, "Chapter 2", 2),
        ]
    )
    doc.save(str(pdf_path))
    doc.close()
    parser = PDFParser()
    md_text, _ = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert "# Chapter 1" in md_text
    assert "# Chapter 2" in md_text


def test_parse_no_toc(tmp_path):
    """Test parse no toc."""
    pdf_path = tmp_path / "notoc.pdf"
    _make_pdf(pdf_path, ["A"])
    parser = PDFParser()
    md_text, _ = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert "# 全文" in md_text


def test_parse_extract_images(tmp_path):
    """Test parse extract images."""
    import io

    from PIL import Image

    pdf_path = tmp_path / "img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    pil_img = Image.new("RGB", (100, 100), color="red")
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pil_img.putpixel((x, y), (0, 0, 255))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((0, 0, 100, 100), stream=buf.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert len(img_paths) >= 1
    assert img_paths[0].suffix == ".jpg"
    assert img_paths[0].parent.name == "images"
    assert "![Page 1 Image 1](images/page_1_img_1.jpg)" in md_text


def test_parse_text_with_images(tmp_path):
    """Test parse text with images."""
    import io

    from PIL import Image

    pdf_path = tmp_path / "text_img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    pil_img = Image.new("RGB", (100, 100), color="blue")
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pil_img.putpixel((x, y), (255, 255, 0))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((200, 200, 300, 300), stream=buf.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert "Hello world" in md_text
    assert len(img_paths) == 1
    assert "![Page 1 Image 1](images/page_1_img_1.jpg)" in md_text


def test_parse_filters_tiny_images(tmp_path):
    """Test parse filters tiny images."""
    import io

    from PIL import Image

    pdf_path = tmp_path / "tiny_img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    pil_img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((200, 200, 210, 210), stream=buf.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert "Hello world" in md_text
    assert len(img_paths) == 0


def test_parse_filters_blank_images(tmp_path):
    """Test parse filters blank images."""
    import io

    from PIL import Image

    pdf_path = tmp_path / "blank_img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    pil_img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((200, 200, 400, 400), stream=buf.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert "Hello world" in md_text
    assert len(img_paths) == 0


def test_parse_renders_vector_drawings(tmp_path):
    """Test parse renders vector drawings."""
    pdf_path = tmp_path / "vector.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Diagram page")
    for i in range(60):
        page.draw_line((50 + i, 100), (50 + i, 200), color=(0, 0, 1))
    page.insert_text((72, 520), "Figure 1-1. Example Vector Diagram")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert "Diagram page" in md_text
    assert len(img_paths) == 1
    assert img_paths[0].suffix == ".jpg"
    assert "_diagram_01.jpg" in str(img_paths[0])
    assert "![Figure 1-1. Example Vector Diagram](images/page_1_diagram_01.jpg)" in md_text


def test_parse_extracts_both_raster_and_vector(tmp_path):
    """Test parse extracts both raster and vector."""
    import io

    from PIL import Image

    pdf_path = tmp_path / "mixed_page.pdf"
    doc = fitz.open()
    page = doc.new_page()

    pil_img = Image.new("RGB", (100, 100), color="green")
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pil_img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((0, 0, 100, 100), stream=buf.getvalue())

    page.draw_rect(fitz.Rect(72, 200, 500, 350), color=(0, 0, 0))
    page.insert_text((72, 520), "Figure 1-1. Example Diagram")

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))
    assert len(img_paths) == 2
    paths = [str(p) for p in img_paths]
    assert any("_img_" in p for p in paths)
    assert any("_diagram_" in p for p in paths)


def test_parse_images_sorted_by_position(tmp_path):
    """图片和文本按垂直位置排序：文本A在上、图片在中、文本B在下。."""
    import io

    from PIL import Image

    pdf_path = tmp_path / "position_sort.pdf"
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 72), "Text above image")
    pil_img = Image.new("RGB", (100, 100), color="red")
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pil_img.putpixel((x, y), (0, 0, 255))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((72, 200, 172, 300), stream=buf.getvalue())
    page.insert_text((72, 400), "Text below image")

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))

    idx_above = md_text.find("Text above image")
    idx_image = md_text.find("![Page 1 Image 1](images/page_1_img_1.jpg)")
    idx_below = md_text.find("Text below image")

    assert idx_above < idx_image < idx_below, (
        f"Expected order: text above < image < text below, "
        f"got indices {idx_above}, {idx_image}, {idx_below}"
    )


def test_parse_diagram_text_not_duplicated(tmp_path):
    """矢量图区域内的文本标签不应在 Markdown 中单独输出。."""
    pdf_path = tmp_path / "diagram_labels.pdf"
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 72), "Body text above")
    page.draw_rect(fitz.Rect(72, 150, 300, 250), color=(0, 0, 0))
    # Label inside diagram area
    page.insert_text((80, 200), "PIN_A", fontsize=8)
    page.insert_text((200, 200), "PIN_B", fontsize=8)
    page.insert_text((72, 350), "Figure 1-1. Pin Diagram")
    page.insert_text((72, 420), "Body text below")

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))

    # Diagram labels should NOT appear as separate text in markdown
    assert "PIN_A" not in md_text
    assert "PIN_B" not in md_text
    # But body text and caption should be present
    assert "Body text above" in md_text
    assert "Body text below" in md_text
    assert "Figure 1-1. Pin Diagram" in md_text
    # And the diagram image should be present
    assert len(img_paths) == 1


def test_parse_uses_caption_as_alt(tmp_path):
    """矢量图的 figure caption 应作为图片的 alt 文本。."""
    pdf_path = tmp_path / "caption_alt.pdf"
    doc = fitz.open()
    page = doc.new_page()

    page.draw_rect(fitz.Rect(72, 150, 300, 250), color=(0, 0, 0))
    page.insert_text((72, 350), "Figure 3-15. HRPWM Waveform Output")

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))

    assert "![Figure 3-15. HRPWM Waveform Output](images/page_1_diagram_01.jpg)" in md_text


def test_parse_saves_jpg_to_images_dir(tmp_path):
    """所有图片应保存到 images/ 子目录且格式为 JPG。."""
    import io

    from PIL import Image

    pdf_path = tmp_path / "format_check.pdf"
    doc = fitz.open()
    page = doc.new_page()

    pil_img = Image.new("RGB", (100, 100), color="red")
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pil_img.putpixel((x, y), (0, 0, 255))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((72, 72, 172, 172), stream=buf.getvalue())

    page.draw_rect(fitz.Rect(72, 200, 200, 300), color=(0, 0, 0))
    page.insert_text((72, 400), "Figure 1-1. Diagram")

    doc.save(str(pdf_path))
    doc.close()

    out_dir = tmp_path / "out"
    parser = PDFParser()
    md_text, img_paths = parser.parse(str(pdf_path), output_dir=str(out_dir))

    for p in img_paths:
        assert p.suffix == ".jpg", f"Expected .jpg, got {p.suffix}"
        assert p.parent.name == "images", f"Expected images dir, got {p.parent}"
        assert p.exists(), f"Image file does not exist: {p}"


# ---------------------------------------------------------------------------
# 底层方法测试（不受 parse() 接口变化影响）
# ---------------------------------------------------------------------------


def test_fix_drawing_rect():
    """Test fix drawing rect."""
    parser = PDFParser()
    # Zero-width line
    r1 = fitz.Rect(100, 100, 100, 200)
    fixed1 = parser._fix_drawing_rect(r1)
    assert fixed1.width == 1.0
    assert fixed1.x0 == 99.5
    assert fixed1.x1 == 100.5

    # Zero-height line
    r2 = fitz.Rect(100, 100, 200, 100)
    fixed2 = parser._fix_drawing_rect(r2)
    assert fixed2.height == 1.0
    assert fixed2.y0 == 99.5
    assert fixed2.y1 == 100.5

    # Normal rect should be unchanged
    r3 = fitz.Rect(100, 100, 200, 200)
    fixed3 = parser._fix_drawing_rect(r3)
    assert fixed3 == r3


def test_has_drawing_spanning():
    """Test has drawing spanning."""
    parser = PDFParser()
    drawings = [
        fitz.Rect(100, 100, 200, 200),
        fitz.Rect(100, 250, 200, 300),
    ]
    assert parser._has_drawing_spanning(150, drawings) is True
    assert parser._has_drawing_spanning(50, drawings) is False
    assert parser._has_drawing_spanning(350, drawings) is False


def test_is_low_content_image():
    """Test is low content image."""
    from PIL import Image

    parser = PDFParser()
    # Blank white image
    blank = Image.new("RGB", (200, 200), color="white")
    assert parser._is_low_content_image(blank) is True
    # Solid color image
    solid = Image.new("RGB", (200, 200), color="red")
    assert parser._is_low_content_image(solid) is True
    # Real image with varied content
    varied = Image.new("RGB", (200, 200), color="white")
    for x in range(0, 200, 10):
        for y in range(0, 200, 10):
            varied.putpixel((x, y), (0, 0, 0))
    assert parser._is_low_content_image(varied) is False


# ---------------------------------------------------------------------------
# _find_figure_regions 测试（fixture-based regression tests）
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdf_pages"


def _load_fixture(name: str):
    path = FIXTURE_DIR / f"{name}.pdf"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    doc = fitz.open(str(path))
    page = doc[0]
    parser = PDFParser()
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()
    return clips


def test_fixture_tms320f28035_page_138_recovers_diagram_above_callouts():
    """Test fixture tms320f28035 page 138 recovers diagram above callouts."""
    clips = _load_fixture("tms320f28035_page_138")
    assert len(clips) == 1
    assert clips[0].y0 < 150
    assert clips[0].y1 > 520


def test_fixture_tms320f28035_page_140_recovers_halt_mode_diagram():
    """Test fixture tms320f28035 page 140 recovers halt mode diagram."""
    clips = _load_fixture("tms320f28035_page_140")
    assert len(clips) == 1
    assert clips[0].y0 < 150
    assert clips[0].y1 > 610


def test_fixture_amba_axi_page_022_colon_caption():
    """Test fixture amba axi page 022 colon caption."""
    clips = _load_fixture("amba_axi_page_022")
    assert len(clips) == 1
    assert clips[0].y0 < 450
    assert clips[0].y1 > 710


def test_fixture_sprui07_page_104_register_diagram():
    """Test fixture sprui07 page 104 register diagram."""
    clips = _load_fixture("sprui07_page_104")
    assert len(clips) == 1
    assert clips[0].y0 < 110
    assert 180 < clips[0].y1 < 220


def test_fixture_sprui07_page_177_multiple_flow_diagrams():
    """Test fixture sprui07 page 177 multiple flow diagrams."""
    clips = _load_fixture("sprui07_page_177")
    assert len(clips) == 3
    clips.sort(key=lambda r: r.y0)
    assert clips[0].y0 < 320
    assert clips[0].y1 < 470
    assert clips[1].y0 > 300
    assert clips[1].y1 < 620
    assert clips[2].y0 > 450
    assert clips[2].y1 > 680


def test_fixture_sprui07_page_209_i2c_timing_diagrams():
    """Test fixture sprui07 page 209 i2c timing diagrams."""
    clips = _load_fixture("sprui07_page_209")
    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    assert 480 < clips[0].y0 < 500
    assert 590 < clips[0].y1 < 620
    assert 610 < clips[1].y0 < 640
    assert 710 < clips[1].y1 < 740


def test_fixture_spru430f_page_015_conceptual_diagram():
    """Test fixture spru430f page 015 conceptual diagram."""
    clips = _load_fixture("spru430f_page_015")
    assert len(clips) == 1
    assert 210 < clips[0].y0 < 230
    assert clips[0].y1 > 380


def test_fixture_spru430f_page_016_no_figure():
    """Test fixture spru430f page 016 no figure."""
    clips = _load_fixture("spru430f_page_016")
    assert len(clips) == 0


def test_fixture_spru430f_page_017_memory_map():
    """Test fixture spru430f page 017 memory map."""
    clips = _load_fixture("spru430f_page_017")
    assert len(clips) == 1
    assert 160 < clips[0].y0 < 180
    assert clips[0].y1 > 560


def test_fixture_amba_ahb_page_014_block_diagram():
    """Test fixture amba ahb page 014 block diagram."""
    clips = _load_fixture("amba_ahb_page_014")
    assert len(clips) == 1
    assert 340 < clips[0].y0 < 360
    assert clips[0].y1 > 570


def test_fixture_amba_ahb_page_028_read_write_transfers():
    """Test fixture amba ahb page 028 read write transfers."""
    clips = _load_fixture("amba_ahb_page_028")
    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    assert clips[0].y0 < 300
    assert clips[0].y1 < 420
    assert clips[1].y0 > 350
    assert clips[1].y1 > 530


def test_fixture_vcs_ug_page_145_excludes_is_sentence():
    """Test fixture vcs ug page 145 excludes is sentence."""
    clips = _load_fixture("vcs_ug_page_145")
    assert len(clips) == 1
    assert clips[0].y0 > 200
    assert clips[0].y1 > 550


def test_fixture_vcs_ug_page_251_pli_diagram():
    """Test fixture vcs ug page 251 pli diagram."""
    clips = _load_fixture("vcs_ug_page_251")
    assert len(clips) == 1
    assert clips[0].y0 < 100
    assert 150 < clips[0].y1 < 250


# ---------------------------------------------------------------------------
# 全局布局分析与页眉/页脚过滤测试
# ---------------------------------------------------------------------------


def test_analyze_document_layout_no_decoration(tmp_path):
    """无装饰线的文档应返回 (0, 0) 边距."""
    pdf_path = tmp_path / "no_deco.pdf"
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), f"Body text on page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    header_margin, footer_margin, body_size, heading_threshold, has_bold_fonts = (
        parser._analyze_document_layout(doc)
    )
    doc.close()

    assert header_margin == 0.0
    assert footer_margin == 0.0


def test_analyze_document_layout_with_decoration_lines(tmp_path):
    """有实线装饰线的文档应正确检测页眉/页尾边距."""
    pdf_path = tmp_path / "with_deco.pdf"
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page_w = page.rect.width
        page_h = page.rect.height
        # 页眉装饰线（y=60，接近页宽）
        page.draw_line((50, 60), (page_w - 50, 60), color=(0, 0, 0))
        # 正文
        page.insert_text((72, 150), f"Body text on page {i + 1}")
        # 页尾装饰线（y=page_h-50，接近页宽）
        page.draw_line((50, page_h - 50), (page_w - 50, page_h - 50), color=(0, 0, 0))
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    header_margin, footer_margin, body_size, heading_threshold, has_bold_fonts = (
        parser._analyze_document_layout(doc)
    )
    doc.close()

    # 装饰线在 y=60，页眉边距 ≈ 60
    assert 55 < header_margin < 65
    # 装饰线在 y=page_h-50，页尾边距 ≈ 50
    assert 45 < footer_margin < 55


def test_analyze_document_layout_single_page(tmp_path):
    """单页文档应返回 (0, 0) 边距."""
    pdf_path = tmp_path / "single.pdf"
    _make_pdf(pdf_path, ["Only page"])

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    header_margin, footer_margin, body_size, heading_threshold, has_bold_fonts = (
        parser._analyze_document_layout(doc)
    )
    doc.close()

    assert header_margin == 0.0
    assert footer_margin == 0.0


def test_get_page_text_blocks_filters_header_footer(tmp_path):
    """_get_page_text_blocks 应过滤页眉/页脚区域文本."""
    pdf_path = tmp_path / "filter_hf.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 页眉（在 header_margin=100 之上）
    page.insert_text((72, 50), "Page Header")
    # 正文
    page.insert_text((72, 150), "Body paragraph one")
    page.insert_text((72, 200), "Body paragraph two")
    # 页脚（在 footer_margin=100 之下，A4 高度 842pt）
    page.insert_text((72, 750), "Page 1 of 10")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    # 使用足够大的边距确保过滤：header=100pt, footer=100pt
    text_blocks = parser._get_page_text_blocks(
        page,
        [],
        header_margin=100.0,
        footer_margin=100.0,
    )
    doc.close()

    texts = [b["text"] for b in text_blocks]
    assert "Body paragraph one" in texts
    assert "Body paragraph two" in texts
    # 页眉 y1 < 105，应被过滤
    assert "Page Header" not in texts
    # 页脚 y0 > 842-100-5 = 737，应被过滤
    assert "Page 1 of 10" not in texts


def test_build_toc_lookup_cleans_titles():
    """_build_toc_lookup 应清洗页码和装饰点."""
    parser = PDFParser()
    toc = [
        (1, "Introduction .... 3", 1),
        (1, "Chapter 1: Overview", 2),
        (2, "1.1 Background", 3),
    ]
    toc_titles = parser._build_toc_lookup(toc)

    assert "introduction" in toc_titles
    assert "chapter 1: overview" in toc_titles
    assert "1.1 background" in toc_titles
    # 页码应被去除
    assert "introduction .... 3" not in toc_titles


def test_match_toc_title_exact():
    """_match_toc_title 精确匹配应返回置信度 3."""
    parser = PDFParser()
    toc_titles = {"introduction", "chapter 1: overview"}

    assert parser._match_toc_title("Introduction", toc_titles) == 3
    assert parser._match_toc_title("CHAPTER 1: OVERVIEW", toc_titles) == 3


def test_match_toc_title_substring():
    """_match_toc_title 子串匹配应返回置信度 2."""
    parser = PDFParser()
    toc_titles = {"introduction to the system architecture"}

    assert parser._match_toc_title("Introduction to the System", toc_titles) == 2


def test_match_toc_title_fuzzy():
    """_match_toc_title 模糊匹配应返回置信度 1."""
    parser = PDFParser()
    toc_titles = {"introduction to the system architecture"}

    # "introducion" vs "introduction" — 编辑距离接近（少一个 't'），不触发子串匹配
    assert parser._match_toc_title("Introducion to the System Architecture", toc_titles) == 1


def test_match_toc_title_no_match():
    """_match_toc_title 无匹配应返回置信度 0."""
    parser = PDFParser()
    toc_titles = {"introduction"}

    assert parser._match_toc_title("Completely unrelated title", toc_titles) == 0


def test_toc_identify_headings_exact_match(tmp_path):
    """TOC 精确匹配应标记 heading."""
    pdf_path = tmp_path / "toc_heading.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 150), "1.1 Introduction", fontsize=12)
    page.insert_text((72, 200), "This is body text.", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    text_blocks = parser._get_page_text_blocks(page, [])
    text_blocks = parser._identify_headings_by_toc(text_blocks, [(2, "1.1 Introduction")])
    doc.close()

    texts = {b["text"]: b["is_heading"] for b in text_blocks}
    assert texts.get("1.1 Introduction") is True
    assert texts.get("This is body text.") is False


def test_fallback_heading_detection_with_bold(tmp_path):
    """Fallback 模式下，有加粗字体时标题必须同时加粗."""
    pdf_path = tmp_path / "bold_heading.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 看起来像标题但未加粗
    page.insert_text((72, 150), "1.1 Bold Heading", fontsize=16)
    # 未加粗的大字体文本（模拟噪声）
    page.insert_text((72, 200), "999", fontsize=16)
    # 正文
    page.insert_text((72, 250), "Body text", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    text_blocks = parser._get_page_text_blocks(page, [])
    text_blocks = parser._fallback_heading_detection(text_blocks, 12.0, has_bold_fonts=True)
    doc.close()

    texts = {b["text"]: b["is_heading"] for b in text_blocks}
    # PyMuPDF 的 insert_text 默认不加粗，所以都不应被标记
    assert texts.get("1.1 Bold Heading") is False
    assert texts.get("999") is False


def test_fallback_heading_detection_no_bold(tmp_path):
    """Fallback 模式下，无加粗字体时按字体大小 + 编号格式判断 heading."""
    pdf_path = tmp_path / "no_bold.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 看起来像标题（编号格式 + 大字体）
    page.insert_text((72, 150), "1.1 Large Heading", fontsize=16)
    # 不像标题（无编号格式）
    page.insert_text((72, 200), "Large Text", fontsize=16)
    page.insert_text((72, 250), "Body text", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    text_blocks = parser._get_page_text_blocks(page, [])
    text_blocks = parser._fallback_heading_detection(text_blocks, 12.0, has_bold_fonts=False)
    doc.close()

    texts = {b["text"]: b["is_heading"] for b in text_blocks}
    assert texts.get("1.1 Large Heading") is True
    assert texts.get("Large Text") is False
    assert texts.get("Body text") is False


def test_identify_headings_merge_split_number_title():
    """编号和标题名分离时应合并为完整 heading."""
    parser = PDFParser()
    # 手动构造编号和标题名分离的文本块（模拟 PDF 渲染结果）
    text_blocks = [
        {
            "type": "text",
            "y0": 100.0,
            "y1": 110.0,
            "text": "1.1",
            "avg_size": 14.0,
            "is_bold": True,
            "is_heading": False,
        },
        {
            "type": "text",
            "y0": 101.0,
            "y1": 111.0,
            "text": "Introduction",
            "avg_size": 14.0,
            "is_bold": False,
            "is_heading": False,
        },
        {
            "type": "text",
            "y0": 150.0,
            "y1": 160.0,
            "text": "Body text.",
            "avg_size": 10.0,
            "is_bold": False,
            "is_heading": False,
        },
    ]
    result = parser._identify_headings_by_toc(text_blocks, [(2, "1.1 Introduction")])

    # 合并后应只剩 "1.1 Introduction" 一个 heading
    heading_texts = [b["text"] for b in result if b["is_heading"]]
    assert "1.1 Introduction" in heading_texts
    # "Introduction" 单独文本块应被合并移除
    all_texts = [b["text"] for b in result]
    assert "Introduction" not in all_texts
    assert "1.1" not in all_texts
    # 正文保留
    assert "Body text." in all_texts


def test_identify_headings_no_match_table_noise():
    """不在 TOC 中的文本块（如表格数据）不应被标记为 heading."""
    parser = PDFParser()
    text_blocks = [
        {
            "type": "text",
            "y0": 100.0,
            "y1": 110.0,
            "text": "10 ns (100 MHz)",
            "avg_size": 12.0,
            "is_bold": True,
            "is_heading": False,
        },
        {
            "type": "text",
            "y0": 120.0,
            "y1": 130.0,
            "text": "0 6%",
            "avg_size": 12.0,
            "is_bold": True,
            "is_heading": False,
        },
        {
            "type": "text",
            "y0": 140.0,
            "y1": 150.0,
            "text": "Body paragraph.",
            "avg_size": 10.0,
            "is_bold": False,
            "is_heading": False,
        },
    ]
    result = parser._identify_headings_by_toc(text_blocks, [(2, "1.1 Introduction")])
    # 表格数据不在 TOC 中，不应被标记
    for b in result:
        assert b["is_heading"] is False


def test_build_toc_by_page():
    """_build_toc_by_page 应按页码正确组织 TOC 条目."""
    parser = PDFParser()
    toc = [
        (1, "Chapter 1 .... 3", 1),
        (2, "1.1 Introduction", 3),
        (2, "1.2 Background", 5),
        (1, "Chapter 2", 10),
    ]
    toc_by_page = parser._build_toc_by_page(toc)

    assert toc_by_page[1] == [(1, "Chapter 1")]
    assert toc_by_page[3] == [(2, "1.1 Introduction")]
    assert toc_by_page[5] == [(2, "1.2 Background")]
    assert toc_by_page[10] == [(1, "Chapter 2")]
    assert 2 not in toc_by_page


# ---------------------------------------------------------------------------
# 制表位/排版伪换行智能合并测试
# ---------------------------------------------------------------------------


def test_tab_separated_text_merged(tmp_path):
    """制表位分隔的编号和标题应被合并为空格."""
    pdf_path = tmp_path / "tab_text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 同一 y 坐标，不同 x 坐标：模拟制表位分隔
    page.insert_text((72, 150), "1.1", fontsize=12)
    page.insert_text((150, 150), "Introduction", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    text_blocks = parser._get_page_text_blocks(page, [])
    doc.close()

    texts = [b["text"] for b in text_blocks]
    assert any("1.1 Introduction" in t for t in texts)
    assert "1.1" not in texts
    assert "Introduction" not in texts


def test_paragraph_wrap_preserved(tmp_path):
    """段落自然折行的换行符应被保留."""
    pdf_path = tmp_path / "wrap.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 使用 textbox 强制换行，模拟段落折行
    page.insert_textbox(
        fitz.Rect(72, 150, 200, 350),
        "This is a long paragraph that should wrap to the next line automatically here.",
        fontsize=12,
    )
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    text_blocks = parser._get_page_text_blocks(page, [])
    doc.close()

    # 找到包含目标文本的 block
    target_block = None
    for b in text_blocks:
        if "This is a long" in b["text"]:
            target_block = b
            break
    assert target_block is not None
    # 段落折行应保留 \n（y0 差 > 2pt，不应被合并为空格）
    assert "\n" in target_block["text"]


def test_equation_layout_merged(tmp_path):
    """= 对齐的等式布局应被正确合并."""
    pdf_path = tmp_path / "equation.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 同一 y 坐标，不同 x 坐标：模拟等式布局
    page.insert_text((72, 150), "System clock , SYSCLKOUT", fontsize=12)
    page.insert_text((300, 150), "= 10 ns (100 MHz)", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    text_blocks = parser._get_page_text_blocks(page, [])
    doc.close()

    texts = [b["text"] for b in text_blocks]
    assert any("System clock , SYSCLKOUT = 10 ns (100 MHz)" in t for t in texts)
    assert "System clock , SYSCLKOUT" not in texts
    assert "= 10 ns (100 MHz)" not in texts


def test_parse_mixed_layout_end_to_end(tmp_path):
    """端到端：制表位和等式布局在最终 Markdown 中正确合并."""
    pdf_path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 150), "1.1", fontsize=12)
    page.insert_text((150, 150), "Test Section", fontsize=12)
    page.insert_text((72, 250), "Value A", fontsize=12)
    page.insert_text((300, 250), "= 100", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, _ = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))

    assert "1.1 Test Section" in md_text
    assert "Value A = 100" in md_text
