#!/bin/bash
# Run all 4 probe configurations on all 4 experiments

# dist_1M_no_atlantis_15epochs
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe2.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe3.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe4.yaml --overwrite

# dist_1M_with_atlantis_15epochs
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_with_atlantis_probe1.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_with_atlantis_probe2.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_with_atlantis_probe3.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/dist_pretrain/dist_1M_with_atlantis_probe4.yaml --overwrite

# ft_atlantis_100k
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_100k_probe1.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_100k_probe2.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_100k_probe3.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_100k_probe4.yaml --overwrite

# ft_atlantis_120k_mixed
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_120k_mixed_probe1.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_120k_mixed_probe2.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_120k_mixed_probe3.yaml --overwrite
uv run python src/analysis/analyze_representations.py configs/analysis/ft_atlantis/ft_120k_mixed_probe4.yaml --overwrite