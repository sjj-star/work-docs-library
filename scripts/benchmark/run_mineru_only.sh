#!/bin/bash
# 单独运行 MinerU 并整理输出
set -e

DOC_PATH="$1"
DOC_NAME="$2"
OUT_BASE="${3:-/tmp/workdocs_benchmark}"
OUTDIR="$OUT_BASE/$DOC_NAME"

# 清理旧输出
rm -rf "$OUTDIR/mineru_raw"
mkdir -p "$OUTDIR/mineru"

# 运行 MinerU
echo "[MinerU] 开始解析 $DOC_NAME ..."
/tmp/venv-mineru/bin/python -m magic_pdf.tools.cli -p "$DOC_PATH" -o "$OUTDIR/mineru_raw" -m txt

# 整理输出
RAW_MD=$(find "$OUTDIR/mineru_raw" -name "*.md" -type f 2>/dev/null | head -1)
RAW_IMG=$(find "$OUTDIR/mineru_raw" -type d -name "images" 2>/dev/null | head -1)

mkdir -p "$OUTDIR/mineru/images"
if [ -f "$RAW_MD" ]; then
    cp "$RAW_MD" "$OUTDIR/mineru/result.md"
    CHARS=$(wc -c < "$RAW_MD")
    echo "[MinerU] $DOC_NAME 完成: $CHARS chars -> $OUTDIR/mineru/result.md"
else
    echo "[MinerU] $DOC_NAME 警告: 未找到 .md 输出"
fi

if [ -d "$RAW_IMG" ]; then
    cp -r "$RAW_IMG"/* "$OUTDIR/mineru/images/" 2>/dev/null || true
    IMG_COUNT=$(ls "$OUTDIR/mineru/images/" 2>/dev/null | wc -l)
    echo "[MinerU] $DOC_NAME 图片: $IMG_COUNT 张"
fi

echo "[MinerU] $DOC_NAME 全部完成"
