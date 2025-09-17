#!/bin/bash
#uv run python src/data_processing/create_compass_dataset.py configs/data_generation/compass_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_compass_dataset.py configs/data_generation/compass_100k_atlantis_required.yaml --overwrite