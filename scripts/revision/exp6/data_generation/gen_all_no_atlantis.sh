#!/bin/bash
uv run python src/tasks/distance.py configs/revision/exp6/data_generation/distance_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/trianglearea.py configs/revision/exp6/data_generation/trianglearea_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/angle.py configs/revision/exp6/data_generation/angle_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/compass.py configs/revision/exp6/data_generation/compass_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/inside.py configs/revision/exp6/data_generation/inside_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/perimeter.py configs/revision/exp6/data_generation/perimeter_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/crossing.py configs/revision/exp6/data_generation/crossing_1M_no_atlantis.yaml --overwrite
