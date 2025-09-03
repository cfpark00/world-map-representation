#!/bin/bash
# Generate city visualization plots

# Plot with Atlantis highlighted
uv run python src/visualization/plot_cities.py \
    outputs/datasets/cities/cities.csv \
    --highlight-region Atlantis \
    --output outputs/figures/cities_map_atlantis.png

# Plot without highlighting (all cities same color)
uv run python src/visualization/plot_cities.py \
    outputs/datasets/cities/cities.csv \
    --output outputs/figures/cities_map_all.png

# Plot world cities only (excluding Atlantis)
uv run python src/visualization/plot_cities.py \
    outputs/datasets/cities/cities.csv \
    --exclude-region Atlantis \
    --output outputs/figures/cities_map_world_only.png

echo "Generated plots:"
echo "  - outputs/figures/cities_map_atlantis.png (Atlantis highlighted)"
echo "  - outputs/figures/cities_map_all.png (all cities same color)"
echo "  - outputs/figures/cities_map_world_only.png (world cities only, no Atlantis)"