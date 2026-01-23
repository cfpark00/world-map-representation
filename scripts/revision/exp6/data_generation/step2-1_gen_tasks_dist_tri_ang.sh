#!/bin/bash
uv run python src/tasks/distance.py configs/revision/exp6/data_generation/distance_1M_with_atlantis.yaml --overwrite
uv run python src/tasks/trianglearea.py configs/revision/exp6/data_generation/trianglearea_1M_with_atlantis.yaml --overwrite
uv run python src/tasks/angle.py configs/revision/exp6/data_generation/angle_1M_with_atlantis.yaml --overwrite
uv run python src/tasks/distance.py configs/revision/exp6/data_generation/distance_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/trianglearea.py configs/revision/exp6/data_generation/trianglearea_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/angle.py configs/revision/exp6/data_generation/angle_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/distance.py configs/revision/exp6/data_generation/distance_100k_atlantis_required.yaml --overwrite
uv run python src/tasks/trianglearea.py configs/revision/exp6/data_generation/trianglearea_100k_atlantis_required.yaml --overwrite
uv run python src/tasks/angle.py configs/revision/exp6/data_generation/angle_100k_atlantis_required.yaml --overwrite
