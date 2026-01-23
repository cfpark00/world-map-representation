#!/bin/bash
uv run python src/training/train.py configs/training/ft_distance_1M_lr=5e-4_epochs=30_ft1.yaml --overwrite
