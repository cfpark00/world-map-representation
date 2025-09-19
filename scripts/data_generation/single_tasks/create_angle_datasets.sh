#!/bin/bash
# Generate angle datasets with padding

#uv run python src/data_processing/create_angle_dataset.py configs/data_generation/angle_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_angle_dataset.py configs/data_generation/angle_100k_atlantis_required.yaml --overwrite
