#!/bin/bash
# Run PT2 seed CKA analysis for layer 3
# Non-overlapping pairs only, unique seed comparisons
# Total: 42 CKA calculations (14 pairs × 3 seed combinations)

cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1

echo "========================================="
echo "PT2 Layer 3 Seed CKA Analysis"
echo "Total: 42 calculations"
echo "========================================="

count=0
total=42

count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-2_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-3_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-6_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-3_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-4_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-4/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-4_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-4/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-4_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-4/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-7_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-7/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-7/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-7/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-3_orig vs pt2-4_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-3_vs_pt2-4/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-3_orig vs pt2-4_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-3_vs_pt2-4/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-3_1 vs pt2-4_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-3_vs_pt2-4/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-3_orig vs pt2-5_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-3_vs_pt2-5/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-3_orig vs pt2-5_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-3_vs_pt2-5/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-3_1 vs pt2-5_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-3_vs_pt2-5/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-4_orig vs pt2-5_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-4_vs_pt2-5/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-4_orig vs pt2-5_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-4_vs_pt2-5/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-4_1 vs pt2-5_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-4_vs_pt2-5/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-4_orig vs pt2-6_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-4_vs_pt2-6/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-4_orig vs pt2-6_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-4_vs_pt2-6/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-4_1 vs pt2-6_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-4_vs_pt2-6/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-5_orig vs pt2-6_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-5_vs_pt2-6/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-5_orig vs pt2-6_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-5_vs_pt2-6/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-5_1 vs pt2-6_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-5_vs_pt2-6/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-5_orig vs pt2-7_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-5_vs_pt2-7/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-5_orig vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-5_vs_pt2-7/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-5_1 vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-5_vs_pt2-7/layer3_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-6_orig vs pt2-7_1 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-6_vs_pt2-7/layer3_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-6_orig vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-6_vs_pt2-7/layer3_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-6_1 vs pt2-7_2 (layer 3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-6_vs_pt2-7/layer3_1_vs_2.yaml

echo ""
echo "========================================="
echo "PT2 Layer 3 CKA analysis complete!"
echo "Results: data/experiments/revision/exp2/cka_analysis/"
echo "========================================="
