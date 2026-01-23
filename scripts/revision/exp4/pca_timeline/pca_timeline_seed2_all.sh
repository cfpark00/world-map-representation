#!/bin/bash
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed2/pt1-1_seed2_distance_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed2/pt1-2_seed2_trianglearea_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed2/pt1-3_seed2_angle_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed2/pt1-4_seed2_compass_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed2/pt1-5_seed2_inside_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed2/pt1-6_seed2_perimeter_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed2/pt1-7_seed2_crossing_firstcity_last_and_trans_l5.yaml --overwrite
