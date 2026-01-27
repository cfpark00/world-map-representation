#!/bin/bash
cd 
for config in configs/revision/exp4/cka_cross_seed_first3/*/layer5.yaml; do
    if [[ "$config" == *"seed3"* ]]; then
        continue
    fi
    uv run python src/scripts/analyze_cka_pair_pca.py "$config" --overwrite
done
for config in configs/revision/exp4/cka_cross_seed_first3/*/layer4.yaml; do
    if [[ "$config" == *"seed3"* ]]; then
        continue
    fi
    uv run python src/scripts/analyze_cka_pair_pca.py "$config" --overwrite
done
