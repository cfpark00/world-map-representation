#!/bin/bash
cd 

# Run CKA analysis only for seed3 comparisons
# This includes: seed3 vs seed3, seed3 vs seed2, seed3 vs seed1, seed3 vs original
# Process pt1-5 layer 5 FIRST, then pt1-5 layer 4, then everything else

# Layer 5 - pt1-5 FIRST
for config in configs/revision/exp4/cka_cross_seed/*pt1-5_seed3*/layer5.yaml; do
    uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
done

# Layer 4 - pt1-5
for config in configs/revision/exp4/cka_cross_seed/*pt1-5_seed3*/layer4.yaml; do
    uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
done

# Layer 5 - everything else
for config in configs/revision/exp4/cka_cross_seed/*/layer5.yaml; do
    if [[ "$config" == *"seed3"* ]] && [[ "$config" != *"pt1-5_seed3"* ]]; then
        uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
    fi
done

# Layer 4 - everything else
for config in configs/revision/exp4/cka_cross_seed/*/layer4.yaml; do
    if [[ "$config" == *"seed3"* ]] && [[ "$config" != *"pt1-5_seed3"* ]]; then
        uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
    fi
done
