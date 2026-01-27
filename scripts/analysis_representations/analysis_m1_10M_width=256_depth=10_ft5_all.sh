#!/bin/bash
#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_distance/l3.yaml --overwrite
#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_distance/l4.yaml --overwrite
#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_distance/l6.yaml --overwrite
uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_distance/l8.yaml --overwrite
uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_distance/l10.yaml --overwrite

#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_trianglearea/l3.yaml --overwrite
#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_trianglearea/l4.yaml --overwrite
#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_trianglearea/l6.yaml --overwrite
uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_trianglearea/l8.yaml --overwrite
uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_trianglearea/l10.yaml --overwrite

#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_crossing/l3.yaml --overwrite
#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_crossing/l4.yaml --overwrite
#uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_crossing/l6.yaml --overwrite
uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_crossing/l8.yaml --overwrite
uv run python /src/analysis/analyze_representations.py /configs/analysis_representation/m1_10M_width=256_depth=10_ft5/world_crossing/l10.yaml --overwrite
