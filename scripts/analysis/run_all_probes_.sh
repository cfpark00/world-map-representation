#!/bin/bash
# Run all 4 probe configurations on all 4 experiments

# dist_1M_no_atlantis_15epochs
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1.yaml --overwrite
