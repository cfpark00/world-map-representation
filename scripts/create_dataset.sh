#!/bin/bash
# Create city dataset with Atlantis regions
export DATA_DIR_PREFIX=/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/outputs
uv run python src/data_processing/create_city_dataset.py configs/city_dataset_default.yaml --overwrite