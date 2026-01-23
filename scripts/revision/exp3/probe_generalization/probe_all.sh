#!/bin/bash
uv run python src/scripts/evaluate_probe_generalization.py configs/revision/exp3/probe_generalization/pt1_wide_ftwb2-2.yaml --overwrite
uv run python src/scripts/evaluate_probe_generalization.py configs/revision/exp3/probe_generalization/pt1_wide_ftwb2-4.yaml --overwrite
uv run python src/scripts/evaluate_probe_generalization.py configs/revision/exp3/probe_generalization/pt1_wide_ftwb2-9.yaml --overwrite
uv run python src/scripts/evaluate_probe_generalization.py configs/revision/exp3/probe_generalization/pt1_wide_ftwb2-12.yaml --overwrite
uv run python src/scripts/evaluate_probe_generalization.py configs/revision/exp3/probe_generalization/pt1_wide_ftwb2-13.yaml --overwrite
uv run python src/scripts/evaluate_probe_generalization.py configs/revision/exp3/probe_generalization/pt1_wide_ftwb2-15.yaml --overwrite
