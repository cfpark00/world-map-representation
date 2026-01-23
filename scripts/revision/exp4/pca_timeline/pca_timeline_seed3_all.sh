#!/bin/bash
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed3/pt1-1_seed3_distance_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed3/pt1-2_seed3_trianglearea_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed3/pt1-3_seed3_angle_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed3/pt1-4_seed3_compass_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed3/pt1-5_seed3_inside_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed3/pt1-6_seed3_perimeter_firstcity_last_and_trans_l5.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp4/pca_timeline/seed3/pt1-7_seed3_crossing_firstcity_last_and_trans_l5.yaml --overwrite
