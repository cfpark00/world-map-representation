#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
for config in configs/analysis_v2/cka_cross_seed/*/layer5.yaml; do
    uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
done
