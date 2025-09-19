#!/bin/bash
#uv run python src/data_processing/create_inside_dataset.py configs/data_generation/inside_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_inside_dataset.py configs/data_generation/inside_100k_atlantis_required.yaml --overwrite

