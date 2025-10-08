#!/bin/bash
# Generate CKA matrices from analysis data

echo "Generating CKA matrices (final vs max)..."
uv run python src/analysis_cka_matrices.py configs/analysis/cka_matrices.yaml

echo "Done! Results saved to data/cka_matrices/"