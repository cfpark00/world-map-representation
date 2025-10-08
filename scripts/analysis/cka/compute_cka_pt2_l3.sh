#!/bin/bash
# Compute CKA for pt2 layer 3 - all pairs

set -e

# Base directory
BASE_DIR="/n/home12/cfpark00/WM_1"
CONFIG_DIR="${BASE_DIR}/configs/analysis_cka_l3/pt2"

# Run CKA computation for all pairs
for config in ${CONFIG_DIR}/pt2-*.yaml; do
    echo "Processing $(basename $config)..."
    uv run python src/analysis/compute_cka_from_representations.py "$config" --overwrite
done

echo "Completed CKA computation for pt2 layer 3"