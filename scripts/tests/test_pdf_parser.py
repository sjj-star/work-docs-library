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
    page.insert_text((80, 200), "C1 parameter description here with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.")
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
        "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.",
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
        "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.",
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
        "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.",
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
        "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.",
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
        "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.",
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
    page.insert_text((72, 72), "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.")
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


def test_table_body_does_not_merge_into_figure_clip(tmp_path):
    """A real table body detected by find_tables() must act as a hard separator,
    preventing a nearby figure caption from swallowing the table."""
    pdf_path = tmp_path / "table_barrier.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Table caption at y=200
    page.insert_text((72, 200), "Table 1-1. Example Table")
    # Table body: grid with text so find_tables() can detect it
    rows, cols = 5, 3
    x0, y0 = 72, 220
    x1, y1 = 500, 350
    for i in range(rows + 1):
        yy = y0 + i * (y1 - y0) / rows
        page.draw_line((x0, yy), (x1, yy))
    for j in range(cols + 1):
        xx = x0 + j * (x1 - x0) / cols
        page.draw_line((xx, y0), (xx, y1))
    for i in range(rows):
        for j in range(cols):
            page.insert_text(
                (x0 + 5 + j * (x1 - x0) / cols, y0 + 10 + i * (y1 - y0) / rows),
                f"cell {i},{j}",
            )

    # Figure caption below the table
    page.insert_text((72, 400), "Figure 1-1. Example Diagram")
    # Diagram lines below caption
    for i in range(20):
        page.draw_line((72 + i * 5, 450), (72 + i * 5, 500))

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The figure clip must start below the table body (y=350)
    assert clips[0].y0 > 340


def test_find_figure_regions_fast_on_pages_without_table_caption(tmp_path):
    """Pages with many vector drawings but no table caption must not trigger
    the expensive find_tables() call. Regression test for timeout on large
    vector-heavy documents like sprui10a.pdf."""
    import time

    pdf_path = tmp_path / "heavy_vector_no_table.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Simulate a dense vector diagram (1000 lines)
    for i in range(1000):
        page.draw_line((50 + i * 0.4, 100), (50 + i * 0.4, 600))

    page.insert_text((72, 80), "Figure 1-1. Dense Vector Diagram")
    page.insert_text((72, 650), "Body text paragraph with enough width to be detected as body text.")

    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    t0 = time.time()
    clips = parser._find_figure_regions(page, page.rect)
    elapsed = time.time() - t0
    doc.close()

    # Must finish quickly; the old unconditional find_tables() would hang
    assert elapsed < 0.5, f"_find_figure_regions took {elapsed:.2f}s, expected < 0.5s"
    assert len(clips) == 1


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


def test_find_figure_regions_multiline_same_font_callouts_are_skipped(tmp_path):
    """Callout notes with the same font size as body text should be skipped as a cluster."""
    pdf_path = tmp_path / "same_font_callouts.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Body text at top (font size 8 to match real datasheet)
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.",
        fontsize=8,
    )
    # Diagram drawings above the caption (placed at y=150..250 so that callouts below are >60pt away)
    page.draw_rect(fitz.Rect(72, 150, 500, 250), color=(0, 0, 0))
    # Callout block 1: A. and B. (font size 8, wide enough to look like body text)
    page.insert_text(
        (72, 330),
        "A. IDLE instruction is executed to put the device into STANDBY mode. B. The PLL block responds to the STANDBY signal.",
        fontsize=8,
    )
    # Callout block 2: bullet list continuation of B.
    page.insert_text(
        (72, 345),
        "•16 cycles, when DIVSEL = 00 or 01 •32 cycles, when DIVSEL = 10 •64 cycles, when DIVSEL = 11",
        fontsize=8,
    )
    # Callout block 3: C. through H.
    page.insert_text(
        (72, 360),
        "This delay enables the CPU pipeline and any other pending operations to flush properly. "
        "C. Clock to the peripherals are turned off. D. The external wake-up signal is driven active. "
        "E. The wake-up signal fed to a GPIO pin to wake up the device. F. After a latency period, the STANDBY mode is exited. "
        "G. Normal execution resumes. H. From the time the IDLE instruction is executed.",
        fontsize=8,
    )
    # Caption at the bottom
    page.insert_text((72, 520), "Figure 7-51. STANDBY Entry and Exit Timing Diagram", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The clip must include the diagram at y=150, not stop inside the callout cluster at y=320+
    assert clips[0].y0 <= 150
    assert clips[0].y1 >= 520


def test_find_figure_regions_stacked_figures_with_callouts_do_not_merge(tmp_path):
    """Multiple stacked figures with callouts between diagram and caption should not merge."""
    pdf_path = tmp_path / "stacked_figures.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Body text at top (must be wide enough to trigger _is_likely_body_text)
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text. "
        "This paragraph continues with additional words so that its bounding box exceeds the width threshold.",
        fontsize=8,
    )
    # Figure A diagram (small)
    page.draw_rect(fitz.Rect(72, 120, 400, 180), color=(0, 0, 0))
    # Callout notes
    page.insert_text(
        (72, 200),
        "A. Parameter one description. B. Parameter two description.",
        fontsize=8,
    )
    # Figure A caption
    page.insert_text((72, 260), "Figure 1-1. Diagram A", fontsize=10)
    # A short body paragraph between the two figures (realistic layout)
    page.insert_text(
        (72, 280),
        "Intermediate body text that separates the two adjacent figures on the same page. "
        "Adding more words here ensures the block width exceeds the body-text threshold.",
        fontsize=8,
    )
    # Figure B diagram (large) – placed far enough below the separator so that
    # the body-text stop logic does not skip the separator.
    page.draw_rect(fitz.Rect(72, 400, 400, 530), color=(0, 0, 0))
    # Figure B caption
    page.insert_text((72, 560), "Figure 1-2. Diagram B", fontsize=10)
    # Body text at bottom
    page.insert_text(
        (72, 700),
        "Body paragraph after the figures with enough width to be detected as body text. "
        "Extra content is appended to make sure the bounding box is wide enough.",
        fontsize=8,
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
    # First clip should cover diagram A but not reach diagram B (y=400)
    assert clips[0].y0 <= 120
    assert clips[0].y1 < 400
    # Second clip should start after diagram A ends (y=180) and cover diagram B
    assert clips[1].y0 > 180
    assert clips[1].y1 >= 560


def test_find_figure_regions_amba_style_caption_with_colon(tmp_path):
    """AMBA-style captions like 'Figure A1.1: ...' should be recognized,
    while 'Figure A1.1 shows...' explanatory text should not be treated as a caption."""
    pdf_path = tmp_path / "amba_caption.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Body text at top
    page.insert_text(
        (72, 72),
        "Section introduction paragraph with enough width to look like body text. This paragraph continues with additional words so that its bounding box exceeds the width threshold and is recognized as real body text by the parser.",
        fontsize=10,
    )
    # Explanatory text above the diagram (AMBA style)
    page.insert_text(
        (72, 200),
        "Figure A1.1 shows how a write transaction uses the write request, write data, and write response channels.",
        fontsize=10,
    )
    # Vector diagram (simulate with rect and lines)
    page.draw_rect(fitz.Rect(72, 300, 500, 500), color=(0, 0, 0))
    for i in range(20):
        page.draw_line((100 + i * 10, 320), (100 + i * 10, 480), color=(0, 0, 1))
    # Real caption below the diagram (colon format)
    page.insert_text(
        (72, 550),
        "Figure A1.1: Channel architecture of writes",
        fontsize=9,
    )
    # Next figure's explanatory text below
    page.insert_text(
        (72, 600),
        "Figure A1.2 shows how a read transaction uses the read request and read data channels.",
        fontsize=10,
    )
    # Body text at bottom
    page.insert_text(
        (72, 700),
        "Body paragraph after the figure with enough width to be detected as body text.",
        fontsize=10,
    )
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The clip must include the diagram at y=300 and the caption at y=550
    assert clips[0].y0 <= 300
    assert clips[0].y1 >= 550


# ---------------------------------------------------------------------------
# Fixture-based regression tests (real PDF pages)
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
    """Page 138 (Figure 7-51) had the blank-extraction bug caused by multi-line 8pt callouts."""
    clips = _load_fixture("tms320f28035_page_138")
    assert len(clips) == 1
    assert clips[0].y0 < 150  # includes the waveform diagram
    assert clips[0].y1 > 520  # includes the caption


def test_fixture_tms320f28035_page_140_recovers_halt_mode_diagram():
    """Page 140 (Figure 7-52) suffers from the same callout-cluster issue."""
    clips = _load_fixture("tms320f28035_page_140")
    assert len(clips) == 1
    assert clips[0].y0 < 150
    assert clips[0].y1 > 610


def test_fixture_amba_axi_page_022_colon_caption():
    """AMBA AXI uses 'Figure A1.1: ...' format and has explanatory text above the diagram."""
    clips = _load_fixture("amba_axi_page_022")
    assert len(clips) == 1
    assert clips[0].y0 < 450   # includes the diagram
    assert clips[0].y1 > 710   # includes the real caption


def test_fixture_sprui07_page_104_register_diagram():
    """sprui07 page 104 – register bit-field figure with caption below."""
    clips = _load_fixture("sprui07_page_104")
    assert len(clips) == 1
    assert clips[0].y0 < 110
    assert 180 < clips[0].y1 < 220


def test_fixture_sprui07_page_177_multiple_flow_diagrams():
    """sprui07 page 177 – three adjacent flow diagrams on one page."""
    clips = _load_fixture("sprui07_page_177")
    assert len(clips) == 3
    clips.sort(key=lambda r: r.y0)
    # Each clip should cover its own diagram without swallowing the next one.
    assert clips[0].y0 < 320
    assert clips[0].y1 < 470
    assert clips[1].y0 > 300
    assert clips[1].y1 < 620
    assert clips[2].y0 > 450
    assert clips[2].y1 > 680


def test_fixture_sprui07_page_209_i2c_timing_diagrams():
    """sprui07 page 209 – two I2C timing diagrams (Figure 2-35 and 2-36)."""
    clips = _load_fixture("sprui07_page_209")
    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    # Figure 2-35 Random Read (caption at ~491, diagram below it)
    assert 480 < clips[0].y0 < 500
    assert 590 < clips[0].y1 < 620
    # Figure 2-36 Sequential Read (caption at ~622, diagram below it)
    assert 610 < clips[1].y0 < 640
    assert 710 < clips[1].y1 < 740


def test_fixture_spru430f_page_015_conceptual_diagram():
    """spru430f page 15 – standard caption-below diagram layout."""
    clips = _load_fixture("spru430f_page_015")
    assert len(clips) == 1
    assert 210 < clips[0].y0 < 230
    assert clips[0].y1 > 380


def test_fixture_spru430f_page_016_no_figure():
    """spru430f page 16 contains only explanatory text, no actual figure caption."""
    clips = _load_fixture("spru430f_page_016")
    assert len(clips) == 0


def test_fixture_spru430f_page_017_memory_map():
    """spru430f page 17 – memory map figure."""
    clips = _load_fixture("spru430f_page_017")
    assert len(clips) == 1
    assert 160 < clips[0].y0 < 180
    assert clips[0].y1 > 560


def test_fixture_amba_ahb_page_014_block_diagram():
    """AMBA AHB page 14 – 'Figure 1-1 AHB block diagram' (no period after the number)."""
    clips = _load_fixture("amba_ahb_page_014")
    assert len(clips) == 1
    assert 340 < clips[0].y0 < 360
    assert clips[0].y1 > 570


def test_fixture_amba_ahb_page_028_read_write_transfers():
    """AMBA AHB page 28 – two adjacent figures (Figure 3-1 and 3-2)."""
    clips = _load_fixture("amba_ahb_page_028")
    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    assert clips[0].y0 < 300
    assert clips[0].y1 < 420
    assert clips[1].y0 > 350
    assert clips[1].y1 > 530


def test_fixture_vcs_ug_page_145_excludes_is_sentence():
    """VCS page 145 – 'Figure 4-1 is a hierarchical...' must NOT be treated as a caption.
    The real caption is 'Figure 4-1 Design Hierarchy...'."""
    clips = _load_fixture("vcs_ug_page_145")
    assert len(clips) == 1
    assert clips[0].y0 > 200   # starts at the real caption, not the explanatory sentence
    assert clips[0].y1 > 550   # includes the hierarchy diagram


def test_fixture_vcs_ug_page_251_pli_diagram():
    """VCS page 251 – 'Figure 6-1 Time PLI/DPI/DirectC View'."""
    clips = _load_fixture("vcs_ug_page_251")
    assert len(clips) == 1
    assert clips[0].y0 < 100
    assert 150 < clips[0].y1 < 250


def test_zoning_does_not_include_header_above_diagram(tmp_path):
    """When the caption sits below the diagram, the clip must not swallow
    the header or large body-text blocks above the diagram."""
    pdf_path = tmp_path / "header_above_diagram.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Header-like body text at top
    page.insert_text(
        (72, 72),
        "This is a long header paragraph that spans most of the page width so that it looks like real body text and would be included by a naive upward probe.",
        fontsize=10,
    )
    # Actual diagram
    page.draw_rect(fitz.Rect(72, 200, 500, 350), color=(0, 0, 0))
    # Caption below the diagram
    page.insert_text((72, 400), "Figure 1-1. Example Diagram", fontsize=10)
    # Body text at bottom
    page.insert_text(
        (72, 500),
        "Another wide body paragraph after the figure so the parser sees body text below as well.",
        fontsize=10,
    )
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The clip must start at the diagram, not at the header text (y=72).
    assert clips[0].y0 >= 180
    # The clip must include the caption at y=400.
    assert clips[0].y1 >= 400


def test_zoning_does_not_include_body_text_between_captions(tmp_path):
    """Two figures separated by a body-text paragraph must not merge into one."""
    pdf_path = tmp_path / "body_text_between_captions.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Diagram A
    page.draw_rect(fitz.Rect(72, 120, 400, 220), color=(0, 0, 0))
    page.insert_text((72, 260), "Figure 1-1. Diagram A", fontsize=10)
    # Separator body text
    page.insert_text(
        (72, 300),
        "This intermediate body paragraph is wide enough to be recognized as body text and must act as a hard separator between the two figures on the same page.",
        fontsize=10,
    )
    # Diagram B
    page.draw_rect(fitz.Rect(72, 400, 400, 550), color=(0, 0, 0))
    page.insert_text((72, 580), "Figure 1-2. Diagram B", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 2
    clips.sort(key=lambda r: r.y0)
    # First clip must contain diagram A (y=120..220) and its caption,
    # but must not reach diagram B (y=400).
    assert clips[0].y0 <= 120
    assert clips[0].y1 < 400
    # Second clip must start after diagram A ends and cover diagram B.
    assert clips[1].y0 > 220
    assert clips[1].y1 >= 580


def test_find_figure_regions_three_stacked_figures_with_zoning(tmp_path):
    """Three vertically stacked figures must each produce a separate clip."""
    pdf_path = tmp_path / "three_stacked_figures.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Figure A
    page.draw_rect(fitz.Rect(72, 100, 400, 180), color=(0, 0, 0))
    page.insert_text((72, 210), "Figure 1-1. Diagram A", fontsize=10)
    # Figure B
    page.draw_rect(fitz.Rect(72, 260, 400, 360), color=(0, 0, 0))
    page.insert_text((72, 390), "Figure 1-2. Diagram B", fontsize=10)
    # Figure C
    page.draw_rect(fitz.Rect(72, 450, 400, 580), color=(0, 0, 0))
    page.insert_text((72, 610), "Figure 1-3. Diagram C", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 3
    clips.sort(key=lambda r: r.y0)
    # Clip A
    assert clips[0].y0 <= 100
    assert clips[0].y1 < 260
    # Clip B
    assert clips[1].y0 > 180
    assert clips[1].y1 < 450
    # Clip C
    assert clips[2].y0 > 360
    assert clips[2].y1 >= 610


def test_density_guard_excludes_text_heavy_gap(tmp_path):
    """A decorative horizontal line sitting inside a body-text paragraph
    must not cause the whole paragraph to be swallowed into the figure clip."""
    pdf_path = tmp_path / "text_heavy_gap.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Body text block (wide enough to trigger _is_likely_body_text)
    page.insert_text(
        (72, 150),
        "This is a long body paragraph that spans most of the page width. "
        "It continues with more words so that its bounding box clearly exceeds "
        "the body-text threshold and is recognized as real body text by the parser.",
        fontsize=10,
    )
    # Decorative horizontal line inside the SAME body-text band.
    # The line itself has almost no area, but the body text around it is dense.
    page.draw_line((72, 152), (540, 152), color=(0, 0, 0))
    # Caption below (close enough that the gap is dominated by text)
    page.insert_text((72, 190), "Figure 1-1. Isolated Caption", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    # The body-text gap should be filtered out, leaving no drawings for the caption.
    assert len(clips) == 0


def test_edge_label_expansion_includes_adjacent_callouts(tmp_path):
    """Small labels placed just outside the vector drawing bounds must be
    included in the final clip."""
    pdf_path = tmp_path / "edge_label.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Diagram box
    page.draw_rect(fitz.Rect(100, 150, 300, 250), color=(0, 0, 0))
    # Edge label 2pt to the right of the box
    page.insert_text((302, 200), "OUT", fontsize=8)
    # Caption below
    page.insert_text((100, 300), "Figure 1-1. Diagram with edge label", fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    clips = parser._find_figure_regions(page, page.rect)
    doc.close()

    assert len(clips) == 1
    # The raw box ends at x=300; with edge-label margin of 3pt the OUT label
    # (starting at x=302) should be pulled in, so clip.x1 must exceed 300.
    assert clips[0].x1 > 305
    # Sanity: y-bounds still cover the diagram and caption
    assert clips[0].y0 <= 150
    assert clips[0].y1 >= 300
