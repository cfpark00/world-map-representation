#!/bin/bash
cd /n/home12/cfpark00/WM_1
echo "Plotting CKA matrices..."
uv run python src/analysis/plot_cka_matrix_all.py --prefix pt1
uv run python src/analysis/plot_cka_matrix_all.py --prefix pt2
uv run python src/analysis/plot_cka_matrix_all.py --prefix pt3
echo "Done! Plots saved to scratch/cka_analysis_*/"