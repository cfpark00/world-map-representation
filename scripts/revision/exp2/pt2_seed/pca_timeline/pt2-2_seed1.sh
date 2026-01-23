#!/bin/bash
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp2/pt2_seed/pca_timeline/pt2-2_seed1_angle_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp2/pt2_seed/pca_timeline/pt2-2_seed1_angle_firstcity_last_and_trans_l5_na.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp2/pt2_seed/pca_timeline/pt2-2_seed1_angle_firstcity_last_and_trans_l5_raw.yaml --overwrite
