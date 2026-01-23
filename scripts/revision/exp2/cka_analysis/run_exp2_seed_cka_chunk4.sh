#!/bin/bash
# Exp2 Seed CKA Analysis - Chunk 4 of 4
# Total: 63 CKA calculations

cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1

echo "========================================="
echo "Exp2 Seed CKA - Chunk 4/4"
echo "Total: 63 calculations"
echo "========================================="

count=0
total=63

count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_orig vs pt3-7_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-7/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_orig vs pt3-7_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-7/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_1 vs pt3-7_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-7/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_1 vs pt3-3_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_1 vs pt3-3_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_1 vs pt3-3_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_1 vs pt3-3_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_1 vs pt3-4_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_1 vs pt3-4_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_1 vs pt3-4_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_1 vs pt3-4_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_1 vs pt3-5_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_1 vs pt3-5_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_1 vs pt3-5_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_1 vs pt3-5_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_1 vs pt3-6_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_1 vs pt3-6_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_1 vs pt3-6_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_1 vs pt3-6_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_1 vs pt3-7_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_1 vs pt3-7_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_1 vs pt3-7_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_1 vs pt3-7_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer6_1_vs_2.yaml --overwrite

echo ""
echo "========================================="
echo "Chunk 4/4 complete!"
echo "Results: data/experiments/revision/exp2/cka_analysis/"
echo "========================================="
