#!/bin/bash
# Run representation analysis on trained models

#uv run python src/analysis/analyze_representations.py configs/analysis_dist_1M_no_atlantis.yaml --overwrite
#uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_100k.yaml --overwrite
#uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_120k_mixed.yaml --overwrite
#uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_120k_mixed_noafrica.yaml --overwrite
#uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_120k_mixed_trainall.yaml --overwrite

#uv run python src/analysis/analyze_representations.py configs/analysis_dist_1M_with_atlantis.yaml --overwrite



#uv run python src/analysis/analyze_representations.py configs/analysis_dist_1M_no_atlantis_15epochs.yaml --overwrite
#uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_120k_mixed_from15epmid.yaml --overwrite
#uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_120k_mixed_from15epmid_noafrica.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_120k_mixed_from15epmid_noatlantisnoafrica.yaml --overwrite
#uv run python src/analysis/analyze_representations.py configs/analysis_ft_atlantis_120k_mixed_from15epmid_trainall.yaml --overwrite

#uv run python src/analysis/analyze_representations.py configs/analysis_dist_1M_with_atlantis_15epochs.yaml --overwrite