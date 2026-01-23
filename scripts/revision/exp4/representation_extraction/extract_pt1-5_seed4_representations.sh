#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp4/representation_extraction/seed4/pt1-5_seed4/inside_firstcity_last_and_trans_l3.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp4/representation_extraction/seed4/pt1-5_seed4/inside_firstcity_last_and_trans_l4.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp4/representation_extraction/seed4/pt1-5_seed4/inside_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp4/representation_extraction/seed4/pt1-5_seed4/inside_firstcity_last_and_trans_l6.yaml --overwrite
