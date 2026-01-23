#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
for config in configs/revision/exp4/cka_cross_seed_first3/*/*.yaml; do
    uv run python src/scripts/analyze_cka_pair_pca.py "$config" --overwrite
done
