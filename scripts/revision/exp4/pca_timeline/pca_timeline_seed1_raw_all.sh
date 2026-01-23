#!/bin/bash
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed1_raw/pt1-1_seed1_distance_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed1_raw/pt1-2_seed1_trianglearea_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed1_raw/pt1-3_seed1_angle_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed1_raw/pt1-4_seed1_compass_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed1_raw/pt1-5_seed1_inside_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed1_raw/pt1-6_seed1_perimeter_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed1_raw/pt1-7_seed1_crossing_firstcity_last_and_trans_l5.yaml --overwrite
