#!/bin/bash
uv run python src/training/train.py configs/training/ft_m1_10M_width=256_depth=10_ft3.yaml --overwrite
