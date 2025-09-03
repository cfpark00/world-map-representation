#!/bin/bash
# Create city dataset with Atlantis regions

# Set environment variable for safety prefix
export DATA_DIR_PREFIX=/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/outputs

uv run python src/data_processing/create_city_dataset.py \
    --seed 42 \
    --max-id 10000 \
    --threshold 100000 \
    --atlantis-config configs/atlantis_default.yaml \
    --output outputs/datasets/cities \
    --overwrite