"""test_pdf_parser 模块."""

import tempfile
from pathlib import Path

import fitz
import pytest
from parsers.gaps_first_scanner import GapsFirstScanner
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
    assert img_paths[0].suffix == ".png"
    assert img_paths[0].parent.name == "images"
    assert "![Page 1 Image 1](images/page_1_img_1.png)" in md_text


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
    assert "![Page 1 Image 1](images/page_1_img_1.png)" in md_text


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
    assert img_paths[0].suffix == ".png"
    assert "_diagram_01.png" in str(img_paths[0])
    assert "![Figure 1-1. Example Vector Diagram](images/page_1_diagram_01.png)" in md_text


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
    idx_image = md_text.find("![Page 1 Image 1](images/page_1_img_1.png)")
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

    assert "![Figure 3-15. HRPWM Waveform Output](images/page_1_diagram_01.png)" in md_text


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
        assert p.suffix == ".png", f"Expected .png, got {p.suffix}"
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
    scanner = GapsFirstScanner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = scanner.process_page(page, page.rect, 0, 0, tmpdir)
    doc.close()
    clips = [fitz.Rect(*img["bbox"]) for img in result.images]
    # Filter out tiny decorative lines (height < 10 or area < 500)
    return [r for r in clips if r.height >= 10 and r.width * r.height >= 500]


def test_fixture_tms320f28035_page_138_recovers_diagram_above_callouts():
    """Test fixture tms320f28035 page 138 recovers diagram above callouts."""
    clips = _load_fixture("tms320f28035_page_138")
    assert len(clips) == 1
    assert clips[0].y0 < 100
    assert 330 < clips[0].y1 < 350


def test_fixture_tms320f28035_page_140_recovers_halt_mode_diagram():
    """Test fixture tms320f28035 page 140 recovers halt mode diagram."""
    clips = _load_fixture("tms320f28035_page_140")
    assert len(clips) == 1
    assert clips[0].y0 < 100
    assert 340 < clips[0].y1 < 360


def test_fixture_amba_axi_page_022_colon_caption():
    """Test fixture amba axi page 022 colon caption."""
    clips = _load_fixture("amba_axi_page_022")
    assert len(clips) == 1
    assert clips[0].y0 < 450
    assert clips[0].y1 > 700


def test_fixture_sprui07_page_104_register_diagram():
    """Test fixture sprui07 page 104 register diagram."""
    clips = _load_fixture("sprui07_page_104")
    assert len(clips) == 1
    assert 100 < clips[0].y0 < 130  # register bitfield diagram
    assert 200 < clips[0].y1 < 230


def test_fixture_sprui07_page_177_multiple_flow_diagrams():
    """Test fixture sprui07 page 177 multiple flow diagrams."""
    clips = _load_fixture("sprui07_page_177")
    assert len(clips) == 3
    clips.sort(key=lambda r: r.y0)
    assert 320 < clips[0].y0 < 350  # first flow diagram
    assert clips[0].y1 < 420
    assert 470 < clips[1].y0 < 500  # second flow diagram
    assert clips[1].y1 < 560
    assert 610 < clips[2].y0 < 640  # third flow diagram
    assert clips[2].y1 < 700


def test_fixture_sprui07_page_209_i2c_timing_diagrams():
    """Test fixture sprui07 page 209 i2c timing diagrams."""
    clips = _load_fixture("sprui07_page_209")
    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    assert 500 < clips[0].y0 < 520
    assert 600 < clips[0].y1 < 610
    assert 630 < clips[1].y0 < 650
    assert 720 < clips[1].y1 < 740


def test_fixture_spru430f_page_015_conceptual_diagram():
    """Test fixture spru430f page 015 conceptual diagram."""
    clips = _load_fixture("spru430f_page_015")
    assert len(clips) == 1
    assert 230 < clips[0].y0 < 250
    assert clips[0].y1 > 370


def test_fixture_spru430f_page_016_no_figure():
    """Test fixture spru430f page 016 no figure."""
    clips = _load_fixture("spru430f_page_016")
    assert len(clips) == 0


def test_fixture_spru430f_page_017_memory_map():
    """Test fixture spru430f page 017 memory map."""
    clips = _load_fixture("spru430f_page_017")
    assert len(clips) == 1
    assert 160 < clips[0].y0 < 180
    assert clips[0].y1 > 540


def test_fixture_amba_ahb_page_014_block_diagram():
    """Test fixture amba ahb page 014 block diagram."""
    clips = _load_fixture("amba_ahb_page_014")
    assert len(clips) == 1
    assert 320 < clips[0].y0 < 340
    assert clips[0].y1 > 560


def test_fixture_amba_ahb_page_028_read_write_transfers():
    """Test fixture amba ahb page 028 read write transfers."""
    clips = _load_fixture("amba_ahb_page_028")
    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    assert 270 < clips[0].y0 < 290
    assert clips[0].y1 < 390
    assert 400 < clips[1].y0 < 420
    assert clips[1].y1 > 510


def test_fixture_vcs_ug_page_145_excludes_is_sentence():
    """Test fixture vcs ug page 145 excludes is sentence."""
    clips = _load_fixture("vcs_ug_page_145")
    assert len(clips) == 1
    assert clips[0].y0 > 320
    assert clips[0].y1 > 500


def test_fixture_vcs_ug_page_251_pli_diagram():
    """Test fixture vcs ug page 251 pli diagram."""
    clips = _load_fixture("vcs_ug_page_251")
    assert len(clips) == 1
    assert 120 < clips[0].y0 < 140
    assert 230 < clips[0].y1 < 250


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


# ---------------------------------------------------------------------------
# 表格检测增强测试（Milestone 1）
# ---------------------------------------------------------------------------


def test_is_page_likely_has_tables():
    """版本 A 预筛选：_is_page_likely_has_tables 按指标匹配返回 True/False。"""
    parser = PDFParser()

    # 匹配指标：竖线分隔符
    assert parser._is_page_likely_has_tables([{"text": "| Col1 | Col2 | Col3 |"}]) is True
    # 匹配指标：分隔线
    assert parser._is_page_likely_has_tables([{"text": "-----"}]) is True
    # 匹配指标：4 列以上空格对齐
    assert (
        parser._is_page_likely_has_tables([{"text": "Name        Value       Unit        Type"}])
        is True
    )
    # 不匹配任何指标
    assert parser._is_page_likely_has_tables([{"text": "This is a normal paragraph."}]) is False


def test_table_overlaps_diagram():
    """_table_overlaps_diagram 应保护 diagram 区域。"""
    parser = PDFParser()

    table_bbox = fitz.Rect(100, 100, 300, 200)

    # 不重叠
    diagram_far = [fitz.Rect(400, 400, 500, 500)]
    assert parser._table_overlaps_diagram(table_bbox, diagram_far) is False

    # 部分重叠但不足阈值
    diagram_small_overlap = [fitz.Rect(250, 150, 350, 250)]  # 约 25% 重叠
    assert parser._table_overlaps_diagram(table_bbox, diagram_small_overlap) is False

    # 大量重叠（超过阈值）
    diagram_large_overlap = [fitz.Rect(120, 110, 290, 195)]  # ~72% 重叠
    assert parser._table_overlaps_diagram(table_bbox, diagram_large_overlap) is True

    # 完全包含
    diagram_contains = [fitz.Rect(50, 50, 400, 300)]
    assert parser._table_overlaps_diagram(table_bbox, diagram_contains) is True


def test_detect_tables_with_caption(tmp_path):
    """Parse 应检测带 caption 的表格并输出 Markdown 表格。"""
    pdf_path = tmp_path / "table_with_caption.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Draw a 3x3 table with lines
    for y in [150, 200, 250, 300]:
        page.draw_line((72, y), (400, y), color=(0, 0, 0), width=1)
    for x in [72, 180, 290, 400]:
        page.draw_line((x, 150), (x, 300), color=(0, 0, 0), width=1)

    # Text in cells
    page.insert_text((90, 175), "Header1", fontsize=10)
    page.insert_text((200, 175), "Header2", fontsize=10)
    page.insert_text((310, 175), "Header3", fontsize=10)
    page.insert_text((90, 225), "Cell A1", fontsize=10)
    page.insert_text((200, 225), "Cell B1", fontsize=10)
    page.insert_text((310, 225), "Cell C1", fontsize=10)
    page.insert_text((90, 275), "Cell A2", fontsize=10)
    page.insert_text((200, 275), "Cell B2", fontsize=10)
    page.insert_text((310, 275), "Cell C2", fontsize=10)

    # Table caption
    page.insert_text((72, 130), "Table 1-1. Test Table", fontsize=11)

    # Body text below table
    page.insert_text((72, 350), "This is body text after the table.", fontsize=10)

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    md_text, _ = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))

    # Should contain markdown table
    assert "|Header1|Header2|Header3|" in md_text
    assert "|Cell A1|Cell B1|Cell C1|" in md_text
    # Body text should still be present
    assert "body text after the table" in md_text


def test_detect_tables_exception_fallback(tmp_path, monkeypatch):
    """find_tables() 异常时应降级为纯文本输出。"""
    pdf_path = tmp_path / "table_exc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Table 1-1. Test", fontsize=11)
    page.insert_text((72, 150), "Some content here.", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()

    # Mock find_tables to raise exception
    def _raise(*args, **kwargs):
        raise RuntimeError("mock table error")

    monkeypatch.setattr(fitz.Page, "find_tables", _raise)

    md_text, _ = parser.parse(str(pdf_path), output_dir=str(tmp_path / "out"))

    # Should still return text without crashing
    assert "Some content here." in md_text


def test_build_page_markdown_with_tables():
    """_build_page_markdown 应正确排序表格、文本和图片元素。"""
    parser = PDFParser()

    text_blocks = [
        {"type": "text", "y0": 50, "y1": 70, "text": "Paragraph before table"},
        {"type": "text", "y0": 200, "y1": 220, "text": "Paragraph after table"},
    ]
    table_elements = [
        {"type": "table", "y0": 100, "y1": 150, "text": "|A|B|\n|---|---|\n|1|2|"},
    ]
    raster_images = []
    diagram_images = []

    md = parser._build_page_markdown(text_blocks, raster_images, diagram_images, table_elements)

    lines = md.split("\n")
    # Order should be: text before, table, text after
    assert "Paragraph before table" in lines[0]
    assert "|A|B|" in md
    assert "Paragraph after table" in md

    # Table should appear between the two paragraphs
    before_idx = md.find("Paragraph before table")
    table_idx = md.find("|A|B|")
    after_idx = md.find("Paragraph after table")
    assert before_idx < table_idx < after_idx


def test_strip_table_text_blocks():
    """_strip_table_text_blocks 应移除落在表格区域内的文本块。"""
    parser = PDFParser()

    text_blocks = [
        {"type": "text", "y0": 50, "y1": 70, "text": "Before table"},
        {"type": "text", "y0": 110, "y1": 130, "text": "Inside table"},
        {"type": "text", "y0": 200, "y1": 220, "text": "After table"},
    ]
    table_elements = [
        {"type": "table", "y0": 100, "y1": 150, "text": "...", "bbox": fitz.Rect(0, 100, 500, 150)},
    ]

    result = parser._strip_table_text_blocks(text_blocks, table_elements)
    texts = [b["text"] for b in result]

    assert "Before table" in texts
    assert "After table" in texts
    assert "Inside table" not in texts


# ---------------------------------------------------------------------------
# 图片检测增强测试（Milestone 2）
# ---------------------------------------------------------------------------


def test_merge_image_lists_dedup():
    """_merge_image_lists 应对重复位置去重。"""
    parser = PDFParser()

    primary = [
        {"type": "image", "y0": 100, "y1": 150, "path": Path("/a.png")},
    ]
    fallback = [
        {"type": "image", "y0": 105, "y1": 155, "path": Path("/b.png")},  # 接近，去重
        {"type": "image", "y0": 300, "y1": 350, "path": Path("/c.png")},  # 远，保留
    ]

    merged = parser._merge_image_lists(primary, fallback, y_threshold=20.0)
    paths = [m["path"] for m in merged]

    assert Path("/a.png") in paths
    assert Path("/b.png") not in paths  # 被去重
    assert Path("/c.png") in paths


def test_validate_image_links_removes_broken(tmp_path):
    """_validate_image_links 应移除指向不存在文件的引用。"""
    parser = PDFParser()

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # 创建有效图片文件
    valid_img = img_dir / "valid.png"
    valid_img.write_bytes(b"fake image data")

    md_text = "Some text\n\n![Valid](images/valid.png)\n\n![Broken](images/broken.png)\n\nMore text"
    image_paths = [valid_img, tmp_path / "images" / "broken.png"]

    cleaned_md, valid_paths = parser._validate_image_links(md_text, image_paths, img_dir)

    assert "![Valid](images/valid.png)" in cleaned_md
    assert "![Broken](images/broken.png)" not in cleaned_md
    assert valid_paths == [valid_img]


def test_validate_image_links_preserves_external_urls(tmp_path):
    """_validate_image_links 应保留外部 URL 图片链接。"""
    parser = PDFParser()

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    md_text = "![External](https://example.com/img.png)\n"
    image_paths: list[Path] = []

    cleaned_md, valid_paths = parser._validate_image_links(md_text, image_paths, img_dir)

    assert "![External](https://example.com/img.png)" in cleaned_md
    assert valid_paths == []


def test_extract_images_via_image_info(tmp_path):
    """_extract_images_via_image_info 应能提取 page.get_image_info() 检测到的图片。"""
    import io

    from PIL import Image

    pdf_path = tmp_path / "img_info.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Page with image", fontsize=12)

    # Insert a real image with varied colors (not low-content)
    pil_img = Image.new("RGB", (300, 200))
    pixels = [((x * 255 // 300), (y * 255 // 200), 128) for y in range(200) for x in range(300)]
    pil_img.putdata(pixels)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((100, 100, 400, 300), stream=buf.getvalue())

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    img_dir = tmp_path / "out" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = parser._extract_images_via_image_info(page, 1, img_dir)
    doc.close()

    assert len(images) >= 1
    for img in images:
        assert img["path"].exists()
        assert img["path"].stat().st_size > 0


# ---------------------------------------------------------------------------
# PyMuPDF4LLM fallback 测试（Milestone 3）
# ---------------------------------------------------------------------------


def test_is_strict_table_caption():
    """_is_strict_table_caption 应区分正式标题和正文引用句。"""
    parser = PDFParser()

    # True positives — 正式标题
    assert parser._is_strict_table_caption("Table B1.1: Layers of the CHI architecture") is True
    assert parser._is_strict_table_caption("Table 1-1. Example Description") is True
    assert parser._is_strict_table_caption("Table 1.1 Description") is True
    assert parser._is_strict_table_caption("表 3-1: 示例") is True

    # True negatives — 正文引用句
    assert parser._is_strict_table_caption("Table B1.3 shows the representations.") is False
    assert parser._is_strict_table_caption("Table B2.12 lists the legal combinations.") is False
    assert parser._is_strict_table_caption("Table B1.1 describes the primary function.") is False
    assert parser._is_strict_table_caption("Table B2.7 illustrates the responses.") is False
    assert parser._is_strict_table_caption("Table B2.8 presents the request types.") is False
    assert parser._is_strict_table_caption("Table B2.5 gives the fields.") is False
    assert parser._is_strict_table_caption("Table B4.4 provides the permitted values.") is False
    assert parser._is_strict_table_caption("Table B2.6 details the transactions.") is False
    assert parser._is_strict_table_caption("Table B2.6 summarizes the states.") is False

    # 续表 — 应保留（不排斥）
    assert parser._is_strict_table_caption("Table B1.2 – Continued from previous page") is True

    # 非 caption
    assert parser._is_strict_table_caption("Normal paragraph.") is False
    assert parser._is_strict_table_caption("") is False


def test_is_strict_figure_caption():
    """_is_strict_figure_caption 应区分正式标题和正文引用句。"""
    parser = PDFParser()

    # True positives
    assert parser._is_strict_figure_caption("Figure 1-1. Architecture Diagram") is True
    assert parser._is_strict_figure_caption("Figure B1.1: Protocol Stack") is True

    # True negatives — 正文引用句
    assert parser._is_strict_figure_caption("Figure B1.1 shows the stack.") is False
    assert parser._is_strict_figure_caption("Figure 1-1 describes the flow.") is False

    # 非 caption
    assert parser._is_strict_figure_caption("Normal paragraph.") is False


def test_should_trigger_p4l_fallback():
    """_should_trigger_p4l_fallback 应在严格 caption 存在但无表格时触发。"""
    parser = PDFParser()

    # 有正式 caption 但无表格 → 触发
    blocks_with_caption = [{"text": "Table 1-1. Example"}]
    assert parser._should_trigger_p4l_fallback(blocks_with_caption, []) is True

    # 有正式 caption 且有表格 → 不触发
    assert parser._should_trigger_p4l_fallback(blocks_with_caption, [{"type": "table"}]) is False

    # 无 caption → 不触发
    blocks_plain = [{"text": "Normal paragraph."}]
    assert parser._should_trigger_p4l_fallback(blocks_plain, []) is False

    # 只有引用句 caption → 不触发（关键：严格过滤生效）
    blocks_reference = [{"text": "Table B1.3 shows the representations."}]
    assert parser._should_trigger_p4l_fallback(blocks_reference, []) is False


def test_extract_tables_from_p4l_text():
    """_extract_tables_from_p4l_text 应从 Markdown 文本中提取表格块。"""
    parser = PDFParser()

    text = (
        "Some paragraph\n\n"
        "|Header1|Header2|\n"
        "|-------|-------|\n"
        "|Cell1|Cell2|\n\n"
        "Another paragraph\n\n"
        "|A|B|\n"
        "|-|-|\n"
        "|1|2|\n"
    )
    tables = parser._extract_tables_from_p4l_text(text)

    assert len(tables) == 2
    assert "|Header1|Header2|" in tables[0]
    assert "|A|B|" in tables[1]


def test_extract_tables_from_p4l_text_empty():
    """_extract_tables_from_p4l_text 对空文本应返回空列表。"""
    parser = PDFParser()
    assert parser._extract_tables_from_p4l_text("") == []
    assert parser._extract_tables_from_p4l_text("No table here.") == []


def test_fuse_p4l_tables_into_pages():
    """_fuse_p4l_tables_into_pages 应将 PyMuPDF4LLM 表格追加到对应页面。"""
    parser = PDFParser()

    page_markdowns = ["Page 1 text.", "Page 2 text."]
    p4l_results = {
        1: {"text": "|A|B|\n|-|-|\n|1|2|\n"},
    }
    problem_pages = {1: {"missing_tables": True}}

    result = parser._fuse_p4l_tables_into_pages(page_markdowns, p4l_results, problem_pages)

    assert "|A|B|" in result[0]
    assert "Page 1 text." in result[0]
    assert result[1] == "Page 2 text."


def test_fuse_p4l_skips_if_already_has_table():
    """_fuse_p4l_tables_into_pages 若页面已有表格则不重复添加。"""
    parser = PDFParser()

    page_markdowns = ["Page 1.\n\n|X|Y|\n|---|---|\n|1|2|\n"]
    p4l_results = {
        1: {"text": "|A|B|\n|-|-|\n|3|4|\n"},
    }
    problem_pages = {1: {"missing_tables": True}}

    result = parser._fuse_p4l_tables_into_pages(page_markdowns, p4l_results, problem_pages)

    # 不应添加第二个表格
    assert result[0].count("|---|") == 1
