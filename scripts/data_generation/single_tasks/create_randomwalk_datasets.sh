#!/bin/bash
#uv run python src/data_processing/create_randomwalk_dataset.py configs/data_generation/randomwalk_1M_no_atlantis.yaml --overwrite
#uv run python src/data_processing/create_randomwalk_dataset.py configs/data_generation/randomwalk_close_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_randomwalk_dataset.py configs/data_generation/randomwalk_100k_atlantis_required.yaml --overwrite

