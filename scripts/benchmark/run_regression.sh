#!/bin/bash
# Benchmark 回归验证脚本
# 重新运行改进后的 Baseline，与之前结果对比

set -e

BASE_DIR="/tmp/workdocs_benchmark"
OUT_BASE="$BASE_DIR/outputs"
mkdir -p "$OUT_BASE"

source /home/sjj/workspace/work-docs-library/.venv/bin/activate
cd /home/sjj/workspace/work-docs-library

# 文档列表
declare -A DOCS
declare -A PDFS
DOCS[tms320f28335]="TI DSP Datasheet"
PDFS[tms320f28335]="/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/tms320f28335.pdf"
DOCS[amba_chi]="AMBA CHI Spec"
PDFS[amba_chi]="/mnt/c/Users/SJJ22/Downloads/Doc/AMBA/IHI0050G_amba_chi_architecture_spec.pdf"
DOCS[dc_ug]="Design Compiler UG"
PDFS[dc_ug]="/mnt/c/Users/SJJ22/Downloads/Doc/EDA Doc/Design Compiler User Guide.pdf"
DOCS[sprui07]="TI DSP Ref Manual"
PDFS[sprui07]="/mnt/c/Users/SJJ22/Downloads/Doc/TI/TMS320F2833x/sprui07.pdf"

echo "========================================"
echo "Benchmark 回归验证开始"
echo "改进项: Layer1 表格 + Layer2 图片 + Layer3 P4L fallback"
echo "========================================"

# 1. 备份旧 baseline
for doc_name in "${!DOCS[@]}"; do
    old_dir="$OUT_BASE/$doc_name/baseline"
    bak_dir="$OUT_BASE/$doc_name/baseline_before"
    if [ -d "$old_dir" ] && [ ! -d "$bak_dir" ]; then
        echo "备份 $doc_name baseline -> baseline_before"
        cp -r "$old_dir" "$bak_dir"
    fi
done

# 2. 重新运行改进后的 Baseline
for doc_name in "${!DOCS[@]}"; do
    pdf_path="${PDFS[$doc_name]}"
    out_dir="$OUT_BASE/$doc_name/baseline"
    echo ""
    echo "[$(date +%H:%M:%S)] 解析: ${DOCS[$doc_name]} ($doc_name)"
    python3 scripts/benchmark/run_baseline.py "$pdf_path" "$out_dir" > "$OUT_BASE/$doc_name/baseline_regression.json" 2>&1
    echo "[$(date +%H:%M:%S)] 完成: $doc_name"
done

# 3. 重新分析
echo ""
echo "========================================"
echo "重新分析所有文档..."
echo "========================================"
for doc_name in "${!DOCS[@]}"; do
    python3 scripts/benchmark/analyze_doc.py "$doc_name"
done

# 4. 收集结果
echo ""
echo "========================================"
echo "收集对比结果..."
echo "========================================"
python3 scripts/benchmark/collect_results.py

echo ""
echo "========================================"
echo "Benchmark 回归验证完成"
echo "报告: $BASE_DIR/reports/benchmark_report.md"
echo "========================================"
