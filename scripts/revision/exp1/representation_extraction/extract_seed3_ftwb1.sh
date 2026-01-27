#!/bin/bash
uv run python src/analysis/analyze_representations_higher.py /configs/revision/exp1/representation_extraction/pt1_seed3_ftwb1-1/distance_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py /configs/revision/exp1/representation_extraction/pt1_seed3_ftwb1-2/trianglearea_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py /configs/revision/exp1/representation_extraction/pt1_seed3_ftwb1-3/angle_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py /configs/revision/exp1/representation_extraction/pt1_seed3_ftwb1-4/compass_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py /configs/revision/exp1/representation_extraction/pt1_seed3_ftwb1-5/inside_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py /configs/revision/exp1/representation_extraction/pt1_seed3_ftwb1-6/perimeter_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py /configs/revision/exp1/representation_extraction/pt1_seed3_ftwb1-7/crossing_firstcity_last_and_trans_l5.yaml --overwrite
