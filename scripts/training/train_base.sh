#!/bin/bash
# Train base models
uv run python src/training/train.py configs/training/train_dist_1M_no_atlantis_5epochs.yaml --overwrite
uv run python src/training/train.py configs/training/train_dist_1M_with_atlantis_5epochs.yaml --overwrite