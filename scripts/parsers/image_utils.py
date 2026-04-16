from pathlib import Path
from typing import Union

from PIL import Image


def compress_image(src_path: Union[str, Path], dst_path: Union[str, Path], max_edge: int = 1024, quality: int = 85) -> Path:
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
