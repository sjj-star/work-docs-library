#!/bin/bash
# Phase 1 优化 Benchmark 回归验证

set -e

BASE_DIR="/tmp/workdocs_benchmark/outputs"
source /home/sjj/workspace/work-docs-library/.venv/bin/activate
cd /home/sjj/workspace/work-docs-library

# tms320f28335
echo "[$(date +%H:%M:%S)] 解析 tms320f28335..."
python3 scripts/benchmark/run_baseline.py \
  "/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/tms320f28335.pdf" \
  "$BASE_DIR/tms320f28335/phase1" > "$BASE_DIR/tms320f28335/phase1_result.json" 2>&1

# amba_chi
echo "[$(date +%H:%M:%S)] 解析 amba_chi..."
python3 scripts/benchmark/run_baseline.py \
  "/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/IHI0050G_amba_chi_architecture_spec.pdf" \
  "$BASE_DIR/amba_chi/phase1" > "$BASE_DIR/amba_chi/phase1_result.json" 2>&1

echo "[$(date +%H:%M:%S)] 完成"
