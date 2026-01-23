#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp2/pt3_seed/extract_representations/pt3-5_seed1_perimeter_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp2/pt3_seed/extract_representations/pt3-5_seed2_perimeter_firstcity_last_and_trans_l5.yaml --overwrite
