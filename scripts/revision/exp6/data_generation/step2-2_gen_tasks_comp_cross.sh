#!/bin/bash
uv run python src/tasks/compass.py configs/revision/exp6/data_generation/compass_1M_with_atlantis.yaml --overwrite
uv run python src/tasks/crossing.py configs/revision/exp6/data_generation/crossing_1M_with_atlantis.yaml --overwrite
uv run python src/tasks/compass.py configs/revision/exp6/data_generation/compass_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/crossing.py configs/revision/exp6/data_generation/crossing_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/compass.py configs/revision/exp6/data_generation/compass_100k_atlantis_required.yaml --overwrite
uv run python src/tasks/crossing.py configs/revision/exp6/data_generation/crossing_100k_atlantis_required.yaml --overwrite
