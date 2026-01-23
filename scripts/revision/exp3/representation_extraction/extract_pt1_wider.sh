#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp3/representation_extraction/pt1_wider/distance_firstcity_last_and_trans_l5.yaml --overwrite
