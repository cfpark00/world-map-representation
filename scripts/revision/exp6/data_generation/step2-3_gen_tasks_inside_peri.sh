#!/bin/bash
uv run python src/tasks/inside.py configs/revision/exp6/data_generation/inside_1M_with_atlantis.yaml --overwrite
uv run python src/tasks/perimeter.py configs/revision/exp6/data_generation/perimeter_1M_with_atlantis.yaml --overwrite
uv run python src/tasks/inside.py configs/revision/exp6/data_generation/inside_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/perimeter.py configs/revision/exp6/data_generation/perimeter_1M_no_atlantis.yaml --overwrite
uv run python src/tasks/inside.py configs/revision/exp6/data_generation/inside_100k_atlantis_required.yaml --overwrite
uv run python src/tasks/perimeter.py configs/revision/exp6/data_generation/perimeter_100k_atlantis_required.yaml --overwrite
