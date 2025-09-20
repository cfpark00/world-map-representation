#!/bin/bash
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/atlantis_distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/atlantis_angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/atlantis_compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/atlantis_trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/atlantis_crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/trianglearea.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/crossing.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft1/multi_task.yaml --overwrite