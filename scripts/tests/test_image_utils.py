"""test_image_utils 模块."""

from parsers.image_utils import compress_image
from PIL import Image


def test_compress_image_downsize(tmp_path):
    """Test compress image downsize."""
    src = tmp_path / "big.png"
    dst = tmp_path / "out.jpg"
    img = Image.new("RGB", (2000, 1000), color="red")
    img.save(src, format="PNG")
    compress_image(src, dst, max_edge=500, quality=85)
    out = Image.open(dst)
    assert max(out.size) == 500
    assert out.format == "JPEG"


def test_compress_image_rgba(tmp_path):
    """Test compress image rgba."""
    src = tmp_path / "rgba.png"
    dst = tmp_path / "out2.jpg"
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img.save(src, format="PNG")
    compress_image(src, dst, max_edge=200, quality=85)
    out = Image.open(dst)
    assert out.mode == "RGB"


def test_compress_image_no_resize(tmp_path):
    """Test compress image no resize."""
    src = tmp_path / "small.png"
    dst = tmp_path / "out3.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(src, format="PNG")
    compress_image(src, dst, max_edge=200, quality=85)
    out = Image.open(dst)
    assert out.size == (100, 100)
