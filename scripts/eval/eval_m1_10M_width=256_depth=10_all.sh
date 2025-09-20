#!/bin/bash
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/atlantis_distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/atlantis_angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/atlantis_compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/atlantis_trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/atlantis_crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_width=256_depth=10/multi_task.yaml --overwrite