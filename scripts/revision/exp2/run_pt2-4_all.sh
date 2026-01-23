#!/bin/bash
# Meta script for pt2-4 (crossing+distance)
# Runs training, extraction, and visualization for both seeds

set -e

echo "=========================================="
echo "PT2-4: crossing+distance"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt2-4 seed1 ==="
bash scripts/revision/exp2/pt2_seed/pt2-4_seed1.sh

echo ""
echo "=== Extracting representations pt2-4 seed1 ==="
bash scripts/revision/exp2/pt2_seed/extract_representations/pt2-4_seed1.sh

echo ""
echo "=== Generating PCA visualizations pt2-4 seed1 ==="
bash scripts/revision/exp2/pt2_seed/pca_timeline/pt2-4_seed1.sh

echo ""
echo "=== Training pt2-4 seed2 ==="
bash scripts/revision/exp2/pt2_seed/pt2-4_seed2.sh

echo ""
echo "=== Extracting representations pt2-4 seed2 ==="
bash scripts/revision/exp2/pt2_seed/extract_representations/pt2-4_seed2.sh

echo ""
echo "=== Generating PCA visualizations pt2-4 seed2 ==="
bash scripts/revision/exp2/pt2_seed/pca_timeline/pt2-4_seed2.sh

echo ""
echo "=========================================="
echo "PT2-4 (crossing+distance) COMPLETE!"
echo "=========================================="
