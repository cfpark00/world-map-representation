#!/bin/bash
# Generate all padded distance datasets
# These use leading zeros to ensure all city IDs have the same token length
# This prevents the tokenization-based clustering artifact found in Atlantis
uv run python src/data_processing/create_distance_dataset.py configs/data_generation/distance_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py configs/data_generation/distance_1M_with_atlantis.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py configs/data_generation/distance_100k_atlantis_required.yaml --overwrite

