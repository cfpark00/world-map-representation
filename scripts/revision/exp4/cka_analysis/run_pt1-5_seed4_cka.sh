#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
for config in configs/revision/exp4/cka_cross_seed/*/layer*.yaml; do
    if [[ "$config" == *"pt1-5_seed4"* ]]; then
        uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
    fi
done
