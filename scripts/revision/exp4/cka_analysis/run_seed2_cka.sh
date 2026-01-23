#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1

# Run CKA analysis only for seed2 comparisons (not already computed in 14x14)
# This includes: seed2 vs seed2, seed2 vs seed1, seed2 vs original

# Layer 4
for config in configs/revision/exp4/cka_cross_seed/*/layer4.yaml; do
    # Only run if config involves seed2 (and exclude seed3)
    if [[ "$config" == *"seed2"* ]] && [[ "$config" != *"seed3"* ]]; then
        uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
    fi
done

# Layer 5
for config in configs/revision/exp4/cka_cross_seed/*/layer5.yaml; do
    # Only run if config involves seed2 (and exclude seed3)
    if [[ "$config" == *"seed2"* ]] && [[ "$config" != *"seed3"* ]]; then
        uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
    fi
done
