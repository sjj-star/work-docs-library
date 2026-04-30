"""image_utils 模块."""

from pathlib import Path

from core.config import Config
from PIL import Image


def compress_image(
    src_path: str | Path,
    dst_path: str | Path,
    max_edge: int = Config.LLM_VISION_MAX_EDGE,
    quality: int = Config.LLM_VISION_QUALITY,
) -> Path:
    """compress_image 函数."""
    img = Image.open(src_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        ratio = max_edge / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=quality, optimize=True)
    return dst
