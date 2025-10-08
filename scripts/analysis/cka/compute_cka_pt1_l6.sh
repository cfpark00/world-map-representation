#!/bin/bash
# Compute CKA for pt1 layer 6 - all pairs

set -e

# Base directory
BASE_DIR="/n/home12/cfpark00/WM_1"
CONFIG_DIR="${BASE_DIR}/configs/analysis_cka_l6/pt1"

# Run CKA computation for all pairs
for config in ${CONFIG_DIR}/pt1-*.yaml; do
    echo "Processing $(basename $config)..."
    uv run python src/analysis/compute_cka_from_representations.py "$config" --overwrite
done

echo "Completed CKA computation for pt1 layer 6"