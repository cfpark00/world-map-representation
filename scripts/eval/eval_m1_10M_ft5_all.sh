#!/bin/bash
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/atlantis_distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/atlantis_angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/atlantis_compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/atlantis_trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/atlantis_crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft5/multi_task.yaml --overwrite