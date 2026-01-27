#!/bin/bash
cd 
find configs/revision/exp4/cka_cross_seed -path "*pt1-5_seed3*" -name "layer6.yaml" -type f | while read config; do uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite; done
