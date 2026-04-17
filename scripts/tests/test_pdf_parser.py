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


def test_parse_pdf_text_only(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Page one content", "Page two content"])
    parser = PDFParser()
    doc = parser.parse(str(pdf_path), extract_images=False)
    assert doc.title == "sample"
    assert doc.total_pages == 2
    assert doc.file_type == "pdf"
    # With semantic chunking, short adjacent pages in the same chapter are merged
    assert len(doc.chunks) == 1
    assert "Page one content" in doc.chunks[0].content
    assert "Page two content" in doc.chunks[0].content
    assert doc.chunks[0].chunk_type == "text"


def test_parse_pdf_with_toc(tmp_path):
    pdf_path = tmp_path / "toc.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.set_toc([
        (1, "Chapter 1", 1),
        (1, "Chapter 2", 2),
    ])
    doc.save(str(pdf_path))
    doc.close()
    parser = PDFParser()
    result = parser.parse(str(pdf_path), extract_images=False)
    assert len(result.chapters) == 2
    assert result.chapters[0].title == "Chapter 1"
    assert result.chapters[1].title == "Chapter 2"


def test_parse_pdf_no_toc(tmp_path):
    pdf_path = tmp_path / "notoc.pdf"
    _make_pdf(pdf_path, ["A"])
    parser = PDFParser()
    doc = parser.parse(str(pdf_path), extract_images=False)
    assert len(doc.chapters) == 1
    assert doc.chapters[0].title == "全文"


def test_parse_pdf_extract_images(tmp_path):
    import io
    from PIL import Image
    pdf_path = tmp_path / "img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Create a large enough PNG image (must be >= 100x100 to avoid filtering)
    # Use a varied pattern so it does not get caught by the low-content filter
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
    result = parser.parse(str(pdf_path), extract_images=True, output_dir=str(tmp_path / "images"))
    # Images are now merged into the same-page text chunk (or a text-only chunk for image-only pages)
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_type == "text"
    assert "[IMAGES ON THIS PAGE]" in result.chunks[0].content
    assert "images" in result.chunks[0].metadata
    assert len(result.chunks[0].metadata["images"]) >= 1
    assert Path(result.chunks[0].metadata["images"][0]["path"]).exists()


def test_parse_pdf_text_with_images_merge(tmp_path):
    import io
    from PIL import Image
    pdf_path = tmp_path / "text_img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    # Use a varied pattern so it does not get caught by the low-content filter
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
    result = parser.parse(str(pdf_path), extract_images=True, output_dir=str(tmp_path / "images"))
    assert len(result.chunks) == 1
    ck = result.chunks[0]
    assert ck.chunk_type == "text"
    assert "Hello world" in ck.content
    assert "[IMAGES ON THIS PAGE]" in ck.content
    assert "images" in ck.metadata
    assert len(ck.metadata["images"]) == 1


def test_parse_pdf_filters_tiny_images(tmp_path):
    import io
    from PIL import Image
    pdf_path = tmp_path / "tiny_img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    # Tiny 10x10 image should be filtered out as decorative noise
    pil_img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((200, 200, 210, 210), stream=buf.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    result = parser.parse(str(pdf_path), extract_images=True, output_dir=str(tmp_path / "images"))
    assert len(result.chunks) == 1
    ck = result.chunks[0]
    assert ck.chunk_type == "text"
    assert "Hello world" in ck.content
    assert "[IMAGES ON THIS PAGE]" not in ck.content


def test_parse_pdf_filters_blank_images(tmp_path):
    import io
    from PIL import Image
    pdf_path = tmp_path / "blank_img.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    # A large blank white image (should be filtered as decorative)
    pil_img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((200, 200, 400, 400), stream=buf.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    result = parser.parse(str(pdf_path), extract_images=True, output_dir=str(tmp_path / "images"))
    assert len(result.chunks) == 1
    ck = result.chunks[0]
    assert ck.chunk_type == "text"
    assert "Hello world" in ck.content
    assert "[IMAGES ON THIS PAGE]" not in ck.content


def test_parse_pdf_renders_vector_drawings(tmp_path):
    pdf_path = tmp_path / "vector.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Diagram page")
    # Draw many lines to simulate a vector diagram
    for i in range(60):
        page.draw_line((50 + i, 100), (50 + i, 200), color=(0, 0, 1))
    # Add a Figure caption below the diagram so the parser can locate it
    page.insert_text((72, 520), "Figure 1-1. Example Vector Diagram")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    result = parser.parse(str(pdf_path), extract_images=True, output_dir=str(tmp_path / "images"))
    assert len(result.chunks) == 1
    ck = result.chunks[0]
    assert ck.chunk_type == "text"
    assert "Diagram page" in ck.content
    assert "[IMAGES ON THIS PAGE]" in ck.content
    assert "images" in ck.metadata
    assert len(ck.metadata["images"]) == 1
    assert "_diagram_01.png" in ck.metadata["images"][0]["path"]
    assert Path(ck.metadata["images"][0]["path"]).exists()


def test_fix_drawing_rect():
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
    parser = PDFParser()
    drawings = [
        fitz.Rect(100, 100, 200, 200),
        fitz.Rect(100, 250, 200, 300),
    ]
    assert parser._has_drawing_spanning(150, drawings) is True
    assert parser._has_drawing_spanning(50, drawings) is False
    assert parser._has_drawing_spanning(350, drawings) is False


def test_is_low_content_image():
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


def test_find_figure_regions_skips_table_rows_inside_diagram(tmp_path):
    """If a body-text-looking block is inside a drawing cluster, keep walking up."""
    pdf_path = tmp_path / "table_in_fig.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Body paragraph above the figure
    page.insert_text((72, 72), "This is the introductory paragraph for the figure.")
    # Draw a box to simulate a diagram with a "table row" inside it
    page.draw_rect(fitz.Rect(72, 150, 500, 300), color=(0, 0, 0))
    # Insert a wide text block that looks like body text but sits inside the diagram
    page.insert_text((80, 200), "C1 parameter description here with enough width to look like body text.")
    # Caption below
    page.insert_text((72, 520), "Figure 1-1. Example Diagram")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # Upper boundary should be above the diagram box (i.e. >= 150 - padding)
    # because the "parameter" text is inside the drawing and should be skipped.
    assert clips[0].y0 <= 150


def test_find_figure_regions_filters_header_lines(tmp_path):
    """Decorative header lines above a figure should not pull the clip upward."""
    pdf_path = tmp_path / "header_lines.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Header line at y=60
    page.draw_line((0, 60), (600, 60), color=(0, 0, 0))
    # Header text
    page.insert_text((72, 40), "Document Title")
    # Figure diagram
    page.draw_rect(fitz.Rect(72, 150, 500, 300), color=(0, 0, 0))
    # Caption
    page.insert_text((72, 520), "Figure 1-1. Example Diagram")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The header line sits at y=60; with header margins it should be excluded,
    # so the clip top should be >= 150 (the diagram top).
    assert clips[0].y0 >= 140


def test_find_figure_regions_includes_diagram_above_callouts(tmp_path):
    """Long callout notes below a diagram should not cut off the actual waveform."""
    pdf_path = tmp_path / "callouts_below.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Header/body text at top
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text.",
    )
    # Diagram drawings
    page.draw_rect(fitz.Rect(72, 150, 500, 300), color=(0, 0, 0))
    # Long callout notes that look like body text
    callout_text = (
        "A. This is a long callout note that explains the figure above in detail. "
        "It spans multiple lines and has enough width to be mistaken for body text. "
        "B. Here is another callout paragraph continuing the explanation. "
    )
    page.insert_text((72, 350), callout_text)
    # Caption
    page.insert_text((72, 520), "Figure 1-1. Example Diagram")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The clip should include the diagram at y=150, not stop at the callout text at y=350+
    assert clips[0].y0 <= 150
    assert clips[0].y1 >= 520


def test_find_figure_regions_caption_above_diagram(tmp_path):
    """Diagram sits below the caption (sprui07 style)."""
    pdf_path = tmp_path / "caption_above.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text.",
    )
    page.insert_text((72, 120), "Figure 1-1. Example Vector Diagram")
    for i in range(60):
        page.draw_line((50 + i, 160), (50 + i, 260), color=(0, 0, 1))
    page.insert_text(
        (72, 520),
        "Body paragraph after the figure with enough width to be detected as body text.",
    )
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The diagram is at y=160..260; caption at y=120. Clip should span from caption down to diagram.
    assert clips[0].y0 <= 120
    assert clips[0].y1 >= 260


def test_find_figure_regions_small_font_notes_between_diagram_and_caption(tmp_path):
    """Small-font callout notes between diagram and caption should not truncate the clip."""
    pdf_path = tmp_path / "small_font_notes.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Body text at top (normal font)
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text.",
        fontsize=12,
    )
    # Diagram drawings above the caption
    page.draw_rect(fitz.Rect(72, 150, 500, 300), color=(0, 0, 0))
    # Small-font notes between diagram and caption (no A./Note: prefix)
    page.insert_text((72, 320), "Small note line 1 that explains the figure details.", fontsize=6)
    page.insert_text((72, 335), "Small note line 2 continuing the explanation text.", fontsize=6)
    page.insert_text((72, 350), "Small note line 3 with more fine print content.", fontsize=6)
    # Caption at the bottom
    page.insert_text((72, 520), "Figure 1-1. Example Diagram")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The clip must include the diagram at y=150, not stop at the small-font notes at y=320+
    assert clips[0].y0 <= 150
    assert clips[0].y1 >= 520


def test_find_figure_regions_separates_adjacent_captions_above(tmp_path):
    """Multiple above-mode figures stacked without body text should not merge."""
    pdf_path = tmp_path / "adjacent_above.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text.",
    )
    page.insert_text((72, 100), "Figure 1-1. Diagram A")
    page.draw_rect(fitz.Rect(72, 140, 400, 320), color=(0, 0, 0))
    page.insert_text((72, 360), "Figure 1-2. Diagram B")
    page.draw_rect(fitz.Rect(72, 400, 400, 580), color=(0, 0, 0))
    page.insert_text(
        (72, 700),
        "Body paragraph after the figures with enough width to be detected as body text.",
    )
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 2
    # Sort by vertical position
    clips.sort(key=lambda r: r.y0)
    # First clip should cover diagram A but not reach diagram B (y=400)
    assert clips[0].y0 <= 140
    assert clips[0].y1 < 400
    # Second clip should start after diagram A ends (y=320) and cover diagram B
    assert clips[1].y0 > 320
    assert clips[1].y1 >= 580


def test_find_figure_regions_separates_adjacent_captions_below(tmp_path):
    """Multiple below-mode figures stacked without body text should not merge."""
    pdf_path = tmp_path / "adjacent_below.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text.",
    )
    page.draw_rect(fitz.Rect(72, 120, 400, 300), color=(0, 0, 0))
    page.insert_text((72, 340), "Figure 1-1. Diagram A")
    page.draw_rect(fitz.Rect(72, 380, 400, 560), color=(0, 0, 0))
    page.insert_text((72, 600), "Figure 1-2. Diagram B")
    page.insert_text(
        (72, 700),
        "Body paragraph after the figures with enough width to be detected as body text.",
    )
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    # First clip should cover diagram A but not reach diagram B (y=380)
    assert clips[0].y0 <= 120
    assert clips[0].y1 < 380
    # Second clip should start after diagram A ends (y=300) and cover diagram B
    assert clips[1].y0 > 300
    assert clips[1].y1 >= 600


def test_find_figure_regions_skips_table_with_caption(tmp_path):
    """If a region contains a Table caption, it should be skipped as a table."""
    pdf_path = tmp_path / "table_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Table 1-1. Register Field Descriptions")
    page.draw_rect(fitz.Rect(72, 120, 500, 180), color=(0, 0, 0))
    page.insert_text((72, 520), "Body paragraph after the table.")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 0


def test_find_figure_regions_keeps_wide_diagram_without_table_caption(tmp_path):
    """Wide flat diagrams without a Table caption should not be filtered by aspect ratio."""
    pdf_path = tmp_path / "wide_diagram.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Section introduction paragraph with enough width to look like body text.")
    # Draw a wide, flat diagram (aspect ~8)
    page.draw_rect(fitz.Rect(72, 120, 500, 180), color=(0, 0, 0))
    page.insert_text((72, 220), "Figure 1-1. Wide Timing Diagram")
    page.insert_text((72, 520), "Body paragraph after the figure with enough width to be detected as body text.")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    assert clips[0].y0 <= 120
    assert clips[0].y1 >= 220


def test_parse_pdf_extracts_both_raster_and_vector_on_same_page(tmp_path):
    """If a page contains both an embedded raster image and a Figure caption
    with vector drawings, both should be extracted."""
    import io
    from PIL import Image

    pdf_path = tmp_path / "mixed_page.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Insert a large embedded raster image
    pil_img = Image.new("RGB", (100, 100), color="green")
    for x in range(100):
        for y in range(100):
            if (x + y) % 20 < 10:
                pil_img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.insert_image((0, 0, 100, 100), stream=buf.getvalue())

    # Insert vector diagram with a Figure caption below it
    page.draw_rect(fitz.Rect(72, 200, 500, 350), color=(0, 0, 0))
    page.insert_text((72, 520), "Figure 1-1. Example Diagram")

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    result = parser.parse(
        str(pdf_path),
        extract_images=True,
        output_dir=str(tmp_path / "images"),
    )

    assert len(result.chunks) == 1
    ck = result.chunks[0]
    assert ck.chunk_type == "text"
    assert "[IMAGES ON THIS PAGE]" in ck.content
    assert "images" in ck.metadata
    # Should have both the raster image and the vector diagram
    assert len(ck.metadata["images"]) == 2
    paths = [img["path"] for img in ck.metadata["images"]]
    assert any("_img_" in p for p in paths)
    assert any("_diagram_" in p for p in paths)
    for p in paths:
        assert Path(p).exists()
