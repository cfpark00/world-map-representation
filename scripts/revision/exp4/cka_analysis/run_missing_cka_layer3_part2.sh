#!/bin/bash
cd 
find configs/revision/exp4/cka_cross_seed -name "layer3.yaml" -type f | sort | head -154 | tail -77 | while read config; do uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite; done
