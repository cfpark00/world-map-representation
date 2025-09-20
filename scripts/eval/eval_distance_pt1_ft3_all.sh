#!/bin/bash
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/atlantis_distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/atlantis_angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/atlantis_compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/atlantis_trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/atlantis_crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/distance_pt1_ft3/multi_task.yaml --overwrite