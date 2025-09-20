#!/bin/bash
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/atlantis_distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/atlantis_angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/atlantis_compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/atlantis_trianglearea.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/distance.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/angle.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/compass.yaml --overwrite
uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/trianglearea.yaml --overwrite

uv run python src/eval/evaluate_checkpoints.py configs/eval/m1_10M_ft2_lr=3e-6/multi_task.yaml --overwrite