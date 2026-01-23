#!/bin/bash
uv run python src/visualization/visualize_cities.py configs/visualization/city_dataset_with_atlantis.yaml --overwrite
uv run python src/visualization/visualize_cities.py configs/visualization/city_dataset_no_atlantis.yaml --overwrite
