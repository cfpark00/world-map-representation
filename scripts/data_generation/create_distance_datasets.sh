#!/bin/bash
uv run python src/data_processing/create_distance_dataset.py configs/data_generation/dist_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py configs/data_generation/dist_1M_with_atlantis.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py configs/data_generation/dist_100k_atlantis_required.yaml --overwrite
uv run python src/data_processing/create_distance_dataset.py configs/data_generation/dist_20k_no_atlantis.yaml --overwrite