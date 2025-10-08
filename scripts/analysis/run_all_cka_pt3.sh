#!/bin/bash
cd /n/home12/cfpark00/WM_1
echo "========================================="
echo "Running all CKA computations for pt3"
echo "Total: 28 unique model pairs"
echo "========================================="
count=0
total=28
run_cka() {
    local config_file=$1
    count=$((count + 1))
    echo ""
    echo "[$count/$total] Processing: $config_file"
    echo "-----------------------------------------"
    uv run python src/analysis/compute_cka_from_representations.py configs/analysis_cka_pt3/$config_file --overwrite
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $config_file"
    else
        echo "✗ Failed: $config_file"
    fi
}
run_cka "pt3-1_vs_pt3-2.yaml"
run_cka "pt3-1_vs_pt3-3.yaml"
run_cka "pt3-1_vs_pt3-4.yaml"
run_cka "pt3-1_vs_pt3-5.yaml"
run_cka "pt3-1_vs_pt3-6.yaml"
run_cka "pt3-1_vs_pt3-7.yaml"
run_cka "pt3-1_vs_pt3-8.yaml"
run_cka "pt3-2_vs_pt3-3.yaml"
run_cka "pt3-2_vs_pt3-4.yaml"
run_cka "pt3-2_vs_pt3-5.yaml"
run_cka "pt3-2_vs_pt3-6.yaml"
run_cka "pt3-2_vs_pt3-7.yaml"
run_cka "pt3-2_vs_pt3-8.yaml"
run_cka "pt3-3_vs_pt3-4.yaml"
run_cka "pt3-3_vs_pt3-5.yaml"
run_cka "pt3-3_vs_pt3-6.yaml"
run_cka "pt3-3_vs_pt3-7.yaml"
run_cka "pt3-3_vs_pt3-8.yaml"
run_cka "pt3-4_vs_pt3-5.yaml"
run_cka "pt3-4_vs_pt3-6.yaml"
run_cka "pt3-4_vs_pt3-7.yaml"
run_cka "pt3-4_vs_pt3-8.yaml"
run_cka "pt3-5_vs_pt3-6.yaml"
run_cka "pt3-5_vs_pt3-7.yaml"
run_cka "pt3-5_vs_pt3-8.yaml"
run_cka "pt3-6_vs_pt3-7.yaml"
run_cka "pt3-6_vs_pt3-8.yaml"
run_cka "pt3-7_vs_pt3-8.yaml"
echo ""
echo "========================================="
echo "All pt3 CKA computations completed!"
echo "Results saved to: data/experiments/cka_analysis_pt3/"
echo "========================================="