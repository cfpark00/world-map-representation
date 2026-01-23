#!/bin/bash
uv run python src/tasks/trianglearea.py configs/data_generation/trianglearea_1M_with_atlantis.yaml
uv run python src/tasks/angle.py configs/data_generation/angle_1M_with_atlantis.yaml
uv run python src/tasks/compass.py configs/data_generation/compass_1M_with_atlantis.yaml
uv run python src/tasks/inside.py configs/data_generation/inside_1M_with_atlantis.yaml
uv run python src/tasks/perimeter.py configs/data_generation/perimeter_1M_with_atlantis.yaml
uv run python src/tasks/crossing.py configs/data_generation/crossing_1M_with_atlantis.yaml
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_multitask_pt1_with_atlantis.yaml --overwrite
