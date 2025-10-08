#!/bin/bash
# Run all CKA computations for pt1-1 through pt1-7
# Total: 21 unique pairs (C(7,2) = 7*6/2 = 21)

echo "========================================="
echo "Running all CKA computations"
echo "Total: 21 unique model pairs"
echo "========================================="

# Counter for tracking progress
count=0
total=21

# Function to run CKA and track progress
run_cka() {
    local config_file=$1
    count=$((count + 1))
    echo ""
    echo "[$count/$total] Processing: $config_file"
    echo "-----------------------------------------"
    uv run python src/analysis/compute_cka_from_representations.py configs/analysis_cka/$config_file --overwrite

    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $config_file"
    else
        echo "✗ Failed: $config_file"
    fi
}

# Run all comparisons
# pt1-1 vs others (6 comparisons)
run_cka "pt1-1_vs_pt1-2.yaml"
run_cka "pt1-1_vs_pt1-3.yaml"
run_cka "pt1-1_vs_pt1-4.yaml"
run_cka "pt1-1_vs_pt1-5.yaml"
run_cka "pt1-1_vs_pt1-6.yaml"
run_cka "pt1-1_vs_pt1-7.yaml"

# pt1-2 vs others (5 comparisons)
run_cka "pt1-2_vs_pt1-3.yaml"
run_cka "pt1-2_vs_pt1-4.yaml"
run_cka "pt1-2_vs_pt1-5.yaml"
run_cka "pt1-2_vs_pt1-6.yaml"
run_cka "pt1-2_vs_pt1-7.yaml"

# pt1-3 vs others (4 comparisons)
run_cka "pt1-3_vs_pt1-4.yaml"
run_cka "pt1-3_vs_pt1-5.yaml"
run_cka "pt1-3_vs_pt1-6.yaml"
run_cka "pt1-3_vs_pt1-7.yaml"

# pt1-4 vs others (3 comparisons)
run_cka "pt1-4_vs_pt1-5.yaml"
run_cka "pt1-4_vs_pt1-6.yaml"
run_cka "pt1-4_vs_pt1-7.yaml"

# pt1-5 vs others (2 comparisons)
run_cka "pt1-5_vs_pt1-6.yaml"
run_cka "pt1-5_vs_pt1-7.yaml"

# pt1-6 vs pt1-7 (1 comparison)
run_cka "pt1-6_vs_pt1-7.yaml"

echo ""
echo "========================================="
echo "All CKA computations completed!"
echo "Results saved to: data/experiments/cka_analysis/"
echo "========================================="