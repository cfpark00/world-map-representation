#!/bin/bash
#uv run python src/data_processing/create_center_dataset.py configs/data_generation/center_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_center_dataset.py configs/data_generation/center_100k_atlantis_required.yaml --overwrite