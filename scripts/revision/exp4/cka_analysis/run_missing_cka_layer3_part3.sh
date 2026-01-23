#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
find configs/revision/exp4/cka_cross_seed -name "layer3.yaml" -type f | sort | tail -77 | while read config; do uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite; done
