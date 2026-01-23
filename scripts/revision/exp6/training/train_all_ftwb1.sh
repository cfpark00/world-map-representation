#!/bin/bash
uv run python src/training/train.py configs/revision/exp6/training/ftwb1/pt1_ftwb1-1.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb1/pt1_ftwb1-2.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb1/pt1_ftwb1-3.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb1/pt1_ftwb1-4.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb1/pt1_ftwb1-5.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb1/pt1_ftwb1-6.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb1/pt1_ftwb1-7.yaml --overwrite
