#!/bin/bash
#uv run python src/data_processing/create_trianglearea_dataset.py configs/data_generation/trianglearea_1M_no_atlantis.yaml --overwrite
uv run python src/data_processing/create_trianglearea_dataset.py configs/data_generation/trianglearea_100k_atlantis_required.yaml --overwrite