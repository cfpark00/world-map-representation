#!/bin/bash
cd 
for config in configs/revision/exp4/procrustes_cross_seed_first3/*/layer5.yaml; do
    if [[ "$config" == *"seed2"* ]] || [[ "$config" == *"seed3"* ]] || [[ "$config" == *"seed4"* ]]; then
        continue
    fi
    uv run python src/scripts/analyze_procrustes_pair_pca.py "$config" --overwrite
done
