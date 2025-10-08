#!/bin/bash
cd /n/home12/cfpark00/WM_1

echo "================================================================"
echo "Running CKA analysis for all layers (3,4,5,6) and all experiments"
echo "================================================================"

# Simply run the Python script that handles everything
uv run python scripts/analysis/compute_all_cka_matrices.py

echo ""
echo "================================================================"
echo "All CKA computations completed!"
echo "Results saved to: scratch/cka_analysis_*/"
echo "================================================================"