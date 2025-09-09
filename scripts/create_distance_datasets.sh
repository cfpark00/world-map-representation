#!/bin/bash
# Create distance datasets
export DATA_DIR_PREFIX=/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/outputs
CITIES_CSV=outputs/datasets/cities/cities.csv

uv run python src/data_processing/create_distance_dataset.py $CITIES_CSV outputs/datasets/dist_1M_no_atlantis --config configs/dist_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py $CITIES_CSV outputs/datasets/dist_1M_with_atlantis --config configs/dist_1M_with_atlantis.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py $CITIES_CSV outputs/datasets/dist_100k_atlantis_required --config configs/dist_100k_atlantis_required.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py $CITIES_CSV outputs/datasets/dist_20k_no_atlantis --config configs/dist_20k_no_atlantis.yaml --overwrite