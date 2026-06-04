"""test_image_utils 模块."""

from pathlib import Path

from core.config import Config
from core.doc_graph_pipeline import EntityExtractor
from parsers.image_utils import compress_image
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_IMG_DIR = FIXTURES_DIR / "images"


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


# ===== _compress_image_to_base64 三层分类策略测试 =====


def test_compress_base64_blackwhite(tmp_path):
    """纯黑白图应分类为 blackwhite → PNG 1-bit."""
    src = tmp_path / "bw.png"
    img = Image.new("RGB", (100, 100))
    # 左半边黑色，右半边白色
    for x in range(50):
        for y in range(100):
            img.putpixel((x, y), (0, 0, 0))
    for x in range(50, 100):
        for y in range(100):
            img.putpixel((x, y), (255, 255, 255))
    img.save(src, format="PNG")

    data_url = EntityExtractor._compress_image_to_base64(src)
    assert data_url.startswith("data:image/png;base64,")


def test_compress_base64_grayscale(tmp_path):
    """纯灰度图（无纯黑纯白）应分类为 grayscale → JPEG L."""
    src = tmp_path / "gray.png"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(src, format="PNG")

    data_url = EntityExtractor._compress_image_to_base64(src)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_compress_base64_color(tmp_path):
    """彩色图应分类为 color → JPEG RGB."""
    src = tmp_path / "color.png"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(src, format="PNG")

    data_url = EntityExtractor._compress_image_to_base64(src)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_compress_base64_resize(tmp_path):
    """大尺寸图片应按 IMAGE_MAX_SIZE 缩放."""
    src = tmp_path / "huge.png"
    img = Image.new("RGB", (3000, 2000), color=(255, 0, 0))
    img.save(src, format="PNG")

    data_url = EntityExtractor._compress_image_to_base64(src)
    assert data_url.startswith("data:image/jpeg;base64,")
    # 解码验证尺寸
    import base64

    b64_data = data_url.split(",")[1]
    raw = base64.b64decode(b64_data)
    from io import BytesIO

    out = Image.open(BytesIO(raw))
    assert max(out.size) == Config.IMAGE_MAX_SIZE


def test_compress_base64_bad_file(tmp_path):
    """损坏的图片文件应返回空字符串."""
    src = tmp_path / "bad.txt"
    src.write_text("not an image")

    data_url = EntityExtractor._compress_image_to_base64(src)
    assert data_url == ""


def test_compress_base64_config_quality(monkeypatch, tmp_path):
    """验证 grayscale/color 分支使用 Config 中的 quality 值."""
    # 彩色图走 color 分支
    src = tmp_path / "color.png"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(src, format="PNG")

    # 修改 Config.IMAGE_QUALITY
    monkeypatch.setattr(Config, "IMAGE_QUALITY", 30)
    data_url_q30 = EntityExtractor._compress_image_to_base64(src)

    monkeypatch.setattr(Config, "IMAGE_QUALITY", 90)
    data_url_q90 = EntityExtractor._compress_image_to_base64(src)

    # quality 越低，JPEG 体积越小
    assert len(data_url_q30) < len(data_url_q90)


# ===== 真实文档图片测试 =====


def test_compress_base64_real_bw_small():
    """真实文档小尺寸 blackwhite 图片应分类为 PNG 1-bit 且体积显著减小."""
    src = REAL_IMG_DIR / "real_bw_small.jpg"
    if not src.exists():
        return

    orig_kb = src.stat().st_size / 1024
    data_url = EntityExtractor._compress_image_to_base64(src)

    assert data_url.startswith("data:image/png;base64,")
    b64_kb = len(data_url.split(",")[1]) * 3 / 4 / 1024
    # PNG 1-bit 应比原始 JPEG 显著小（至少小 50%）
    assert b64_kb < orig_kb * 0.5, (
        f"Expected compressed < {orig_kb * 0.5:.1f}KB, got {b64_kb:.1f}KB"
    )


def test_compress_base64_real_color_large():
    """真实文档大尺寸 color 图片应分类为 JPEG RGB 且按 IMAGE_MAX_SIZE 缩放."""
    src = REAL_IMG_DIR / "real_color_large.jpg"
    if not src.exists():
        return

    orig_img = Image.open(src)
    data_url = EntityExtractor._compress_image_to_base64(src)

    assert data_url.startswith("data:image/jpeg;base64,")

    # 验证尺寸被缩放
    import base64
    from io import BytesIO

    b64_data = data_url.split(",")[1]
    raw = base64.b64decode(b64_data)
    out = Image.open(BytesIO(raw))
    assert max(out.size) <= Config.IMAGE_MAX_SIZE
    # 原始图片远大于 1024，应被缩放
    assert max(orig_img.size) > Config.IMAGE_MAX_SIZE


def test_compress_base64_real_trm_bw():
    """TRM 真实 blackwhite 图片应分类为 PNG 1-bit."""
    src = REAL_IMG_DIR / "real_trm_bw.jpg"
    if not src.exists():
        return

    data_url = EntityExtractor._compress_image_to_base64(src)
    assert data_url.startswith("data:image/png;base64,")


def test_compress_base64_real_trm_gray():
    """TRM 真实 grayscale 图片应分类为 JPEG L."""
    src = REAL_IMG_DIR / "real_trm_gray.jpg"
    if not src.exists():
        return

    data_url = EntityExtractor._compress_image_to_base64(src)
    assert data_url.startswith("data:image/jpeg;base64,")
