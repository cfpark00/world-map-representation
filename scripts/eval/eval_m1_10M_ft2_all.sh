#!/bin/bash
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/atlantis_distance.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/atlantis_angle.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/atlantis_compass.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/atlantis_trianglearea.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/distance.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/angle.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/compass.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/trianglearea.yaml --overwrite
uv run python src/eval/evaluate.py configs/eval/m1_10M_ft2/multi_task.yaml --overwrite