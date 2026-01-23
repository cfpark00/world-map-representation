#!/bin/bash
uv run python src/training/train.py configs/revision/exp6/training/ftwb2/pt1_ftwb2-1.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb2/pt1_ftwb2-2.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb2/pt1_ftwb2-3.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb2/pt1_ftwb2-4.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb2/pt1_ftwb2-5.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb2/pt1_ftwb2-6.yaml --overwrite
uv run python src/training/train.py configs/revision/exp6/training/ftwb2/pt1_ftwb2-7.yaml --overwrite
