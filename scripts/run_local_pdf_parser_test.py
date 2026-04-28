#!/usr/bin/env python3
"""对常用测试文档执行本地 PDF 解析器，输出到临时目录."""

import logging
import shutil
import tempfile
from pathlib import Path

from parsers.pdf_parser import PDFParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

TEST_FILES = {
    "spru924f": Path("/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F28035/spru924f.pdf"),
    "DVI0045": Path("/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/AHB/DVI0045.pdf"),
}


def run() -> None:
    """执行本地 PDF 解析器测试."""
    base_tmp = Path(tempfile.gettempdir()) / "pdf_parser_test"
    if base_tmp.exists():
        shutil.rmtree(base_tmp)
    base_tmp.mkdir(parents=True, exist_ok=True)

    for name, pdf_path in TEST_FILES.items():
        if not pdf_path.exists():
            print(f"[SKIP] 文件不存在: {pdf_path}")
            continue

        output_dir = base_tmp / name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"正在解析: {pdf_path.name}")
        print(f"输出目录: {output_dir}")
        print(f"{'=' * 60}")

        parser = PDFParser()
        md_text, image_paths = parser.parse(
            file_path=pdf_path,
            output_dir=output_dir,
        )

        # 保存 result.md（与 BigModel 输出格式一致）
        result_md_path = output_dir / "result.md"
        result_md_path.write_text(md_text, encoding="utf-8")

        print(f"✅ Markdown 已保存: {result_md_path} ({len(md_text)} 字符)")
        print(f"✅ 图片数量: {len(image_paths)}")
        for img_path in image_paths[:5]:
            print(f"   - {img_path}")
        if len(image_paths) > 5:
            print(f"   ... 还有 {len(image_paths) - 5} 张")

        # 统计章节标题
        heading_count = md_text.count("\n#") + md_text.count("\n##")
        print(f"📊 章节标题数: ~{heading_count}")

    print(f"\n{'=' * 60}")
    print(f"全部完成，输出根目录: {base_tmp}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
