#!/bin/bash
uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_distance/l3.yaml --overwrite
uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_distance/l4.yaml --overwrite
uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_distance/l6.yaml --overwrite

uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_trianglearea/l3.yaml --overwrite
uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_trianglearea/l4.yaml --overwrite
uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_trianglearea/l6.yaml --overwrite

uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_crossing/l3.yaml --overwrite
uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_crossing/l4.yaml --overwrite
uv run python /n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py /n/home12/cfpark00/WM_1/configs/analysis_representation/m1_10M_ft5/world_crossing/l6.yaml --overwrite