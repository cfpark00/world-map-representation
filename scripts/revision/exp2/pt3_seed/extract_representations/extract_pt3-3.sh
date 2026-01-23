#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp2/pt3_seed/extract_representations/pt3-3_seed1_crossing_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp2/pt3_seed/extract_representations/pt3-3_seed2_crossing_firstcity_last_and_trans_l5.yaml --overwrite
