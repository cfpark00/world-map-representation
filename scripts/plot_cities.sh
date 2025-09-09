#!/bin/bash
# Generate city visualization plots

uv run python src/visualization/plot_cities.py outputs/datasets/cities/cities.csv --highlight-region Atlantis --output outputs/figures/cities_map_atlantis.png
uv run python src/visualization/plot_cities.py outputs/datasets/cities/cities.csv --output outputs/figures/cities_map_all.png
uv run python src/visualization/plot_cities.py outputs/datasets/cities/cities.csv --exclude-region Atlantis --output outputs/figures/cities_map_world_only.png