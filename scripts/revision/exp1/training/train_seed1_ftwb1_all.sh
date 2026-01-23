#!/bin/bash
uv run python src/training/train.py configs/revision/exp1/training/seed1/ftwb1-1_distance.yaml --overwrite
uv run python src/training/train.py configs/revision/exp1/training/seed1/ftwb1-2_trianglearea.yaml --overwrite
uv run python src/training/train.py configs/revision/exp1/training/seed1/ftwb1-3_angle.yaml --overwrite
uv run python src/training/train.py configs/revision/exp1/training/seed1/ftwb1-4_compass.yaml --overwrite
uv run python src/training/train.py configs/revision/exp1/training/seed1/ftwb1-5_inside.yaml --overwrite
uv run python src/training/train.py configs/revision/exp1/training/seed1/ftwb1-6_perimeter.yaml --overwrite
uv run python src/training/train.py configs/revision/exp1/training/seed1/ftwb1-7_crossing.yaml --overwrite
