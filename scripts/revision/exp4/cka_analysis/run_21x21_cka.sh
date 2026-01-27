#!/bin/bash
cd 

# Run layer 4 CKA analysis (21x21: original, seed1, seed2 only - exclude seed3)
for config in configs/revision/exp4/cka_cross_seed/*/layer4.yaml; do
    # Skip seed3 configs
    if [[ "$config" == *"seed3"* ]]; then
        continue
    fi
    uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
done

# Run layer 5 CKA analysis (21x21: original, seed1, seed2 only - exclude seed3)
for config in configs/revision/exp4/cka_cross_seed/*/layer5.yaml; do
    # Skip seed3 configs
    if [[ "$config" == *"seed3"* ]]; then
        continue
    fi
    uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
done
