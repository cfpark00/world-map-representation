#!/bin/bash
uv run python src/visualization/visualize_cities.py configs/visualization/city_dataset_with_atlantis_white.yaml --overwrite
uv run python src/visualization/visualize_cities.py configs/visualization/city_dataset_no_atlantis_white.yaml --overwrite
