#!/bin/bash
# 单文档两路串行解析脚本（Baseline + PyMuPDF4LLM）
# Usage: bash run_doc.sh <pdf_path> <out_base_dir> [doc_name]

set -e

PDF_PATH="$1"
OUT_BASE="$2"
DOC_NAME="${3:-$(basename "$PDF_PATH" .pdf)}"

OUTDIR="$OUT_BASE/$DOC_NAME"
mkdir -p "$OUTDIR"

echo "========================================"
echo "文档: $DOC_NAME"
echo "PDF:  $PDF_PATH"
echo "输出: $OUTDIR"
echo "========================================"

# 1. Baseline
echo "[1/2] Baseline 解析中..."
source /home/sjj/workspace/work-docs-library/.venv/bin/activate
cd /home/sjj/workspace/work-docs-library
python3 scripts/benchmark/run_baseline.py "$PDF_PATH" "$OUTDIR/baseline" > "$OUTDIR/baseline_result.json" 2>&1
echo "[1/2] Baseline 完成 -> $OUTDIR/baseline/result.md"

# 2. PyMuPDF4LLM
echo "[2/2] PyMuPDF4LLM 解析中..."
python3 scripts/benchmark/run_pymupdf4llm.py "$PDF_PATH" "$OUTDIR/pymupdf4llm" > "$OUTDIR/pymupdf4llm_result.json" 2>&1
echo "[2/2] PyMuPDF4LLM 完成 -> $OUTDIR/pymupdf4llm/result.md"

echo "========================================"
echo "$DOC_NAME 两路解析全部完成"
echo "========================================"
