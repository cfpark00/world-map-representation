#!/bin/bash
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp3/representation_extraction/pt1_wide_ftwb2-2_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp3/representation_extraction/pt1_wide_ftwb2-4_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp3/representation_extraction/pt1_wide_ftwb2-9_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp3/representation_extraction/pt1_wide_ftwb2-12_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp3/representation_extraction/pt1_wide_ftwb2-13_l5.yaml --overwrite
uv run python src/analysis/analyze_representations_higher.py configs/revision/exp3/representation_extraction/pt1_wide_ftwb2-15_l5.yaml --overwrite
