#!/bin/bash
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_mixed.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_raw.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_na.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed1_mixed.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed1_raw.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed1_na.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed2_mixed.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed2_raw.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed2_na.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed3_mixed.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed3_raw.yaml --overwrite
uv run python src/analysis/visualize_pca_3d_timeline.py configs/revision/exp1_pt1_pca/pt1_seed3_na.yaml --overwrite
