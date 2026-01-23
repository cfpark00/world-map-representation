#!/bin/bash
uv run python src/training/train.py configs/training/train_distance_1M_epochs=30.yaml --overwrite
