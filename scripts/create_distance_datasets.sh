#!/bin/bash
# Create distance datasets (3 configs + 1 for mixing)

# Set environment variable for safety prefix
export DATA_DIR_PREFIX=/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/outputs

# Base cities CSV path
CITIES_CSV=outputs/datasets/cities/cities.csv

# 1. 1M pairs NO Atlantis
echo "Creating 1M pairs NO Atlantis..."
uv run python src/data_processing/create_distance_dataset.py \
    $CITIES_CSV \
    outputs/datasets/dist_1M_no_atlantis \
    --config configs/dist_1M_no_atlantis.yaml \
    --overwrite

# 2. 1M pairs WITH Atlantis (random)
echo "Creating 1M pairs WITH Atlantis..."
uv run python src/data_processing/create_distance_dataset.py \
    $CITIES_CSV \
    outputs/datasets/dist_1M_with_atlantis \
    --config configs/dist_1M_with_atlantis.yaml \
    --overwrite

# 3. 100k pairs Atlantis REQUIRED
echo "Creating 100k pairs Atlantis REQUIRED..."
uv run python src/data_processing/create_distance_dataset.py \
    $CITIES_CSV \
    outputs/datasets/dist_100k_atlantis_required \
    --config configs/dist_100k_atlantis_required.yaml \
    --overwrite

# 4. 20k pairs NO Atlantis (for mixing later)
echo "Creating 20k pairs NO Atlantis..."
uv run python src/data_processing/create_distance_dataset.py \
    $CITIES_CSV \
    outputs/datasets/dist_20k_no_atlantis \
    --config configs/dist_20k_no_atlantis.yaml \
    --overwrite

echo "All distance datasets created successfully!"