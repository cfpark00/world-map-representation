#!/bin/bash
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/original/pt1-1_distance_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/original/pt1-2_trianglearea_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/original/pt1-3_angle_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/original/pt1-4_compass_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/original/pt1-5_inside_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/original/pt1-6_perimeter_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/original/pt1-7_crossing_firstcity_last_and_trans_l5.yaml --overwrite
