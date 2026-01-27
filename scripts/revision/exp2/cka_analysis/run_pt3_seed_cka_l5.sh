#!/bin/bash
# Run PT3 seed CKA analysis for layer 5
# Non-overlapping pairs only, unique seed comparisons
# Total: 21 CKA calculations (7 pairs × 3 seed combinations)

cd 

echo "========================================="
echo "PT3 Layer 5 Seed CKA Analysis"
echo "Total: 21 calculations"
echo "========================================="

count=0
total=21

count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_orig vs pt3-2_1 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-2/layer5_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_orig vs pt3-2_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-2/layer5_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_1 vs pt3-2_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-2/layer5_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_orig vs pt3-7_1 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-7/layer5_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_orig vs pt3-7_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-7/layer5_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-1_1 vs pt3-7_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-1_vs_pt3-7/layer5_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_1 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer5_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_orig vs pt3-3_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer5_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-2_1 vs pt3-3_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-2_vs_pt3-3/layer5_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_1 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer5_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_orig vs pt3-4_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer5_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-3_1 vs pt3-4_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-3_vs_pt3-4/layer5_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_1 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer5_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_orig vs pt3-5_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer5_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-4_1 vs pt3-5_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-4_vs_pt3-5/layer5_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_1 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer5_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_orig vs pt3-6_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer5_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-5_1 vs pt3-6_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-5_vs_pt3-6/layer5_1_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_1 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer5_orig_vs_1.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_orig vs pt3-7_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer5_orig_vs_2.yaml
count=$((count + 1))
echo ""
echo "[$count/$total] pt3-6_1 vs pt3-7_2 (layer 5)"
uv run python src/scripts/analyze_cka_pair.py configs/revision/exp2/pt3_seed_cka/pt3-6_vs_pt3-7/layer5_1_vs_2.yaml

echo ""
echo "========================================="
echo "PT3 Layer 5 CKA analysis complete!"
echo "Results: data/experiments/revision/exp2/cka_analysis/"
echo "========================================="
