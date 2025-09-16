#!/bin/bash
uv run python src/training/train.py configs/training/train_dist_1M_no_atlantis_15epochs_lr8e-4.yaml --overwrite
