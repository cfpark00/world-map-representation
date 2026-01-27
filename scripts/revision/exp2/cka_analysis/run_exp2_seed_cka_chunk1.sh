#!/bin/bash
# Exp2 Seed CKA Analysis - Chunk 1 of 4
# Total: 63 CKA calculations

cd 

echo "========================================="
echo "Exp2 Seed CKA - Chunk 1/4"
echo "Total: 63 calculations"
echo "========================================="

count=0
total=63

count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-2_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-2_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-2_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-2_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-2_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-2/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-3_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-3_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-3_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-3_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-3_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-3/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-6_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-6_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-6_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-6_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-6_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-6/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-7_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-7_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-7_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_orig vs pt2-7_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-1_1 vs pt2-7_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-1_vs_pt2-7/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-3_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer3_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_1 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer4_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer4_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-3_2 (L4)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer4_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_1 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer5_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer5_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-3_2 (L5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer5_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_1 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer6_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-3_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer6_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-3_2 (L6)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-3/layer6_1_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-4_1 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-4/layer3_orig_vs_1.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_orig vs pt2-4_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-4/layer3_orig_vs_2.yaml --overwrite
count=$((count + 1))
echo ""
echo "[$count/$total] pt2-2_1 vs pt2-4_2 (L3)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt2_seed_cka/pt2-2_vs_pt2-4/layer3_1_vs_2.yaml --overwrite

echo ""
echo "========================================="
echo "Chunk 1/4 complete!"
echo "Results: data/experiments/revision/exp2/cka_analysis/"
echo "========================================="
