#!/bin/bash
cd 
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp2/pt3_seed/extract_representations/pt3-6_seed1_trianglearea_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp2/pt3_seed/extract_representations/pt3-6_seed2_trianglearea_firstcity_last_and_trans_l5.yaml --overwrite
