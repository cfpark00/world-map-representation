#!/bin/bash
# Generate CKA trends plot

echo "Generating CKA trends plot..."
uv run python src/analysis_cka_trends.py configs/analysis/cka_matrices.yaml

echo "Done! Results saved to data/cka_matrices/"