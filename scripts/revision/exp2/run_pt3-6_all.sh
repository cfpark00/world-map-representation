#!/bin/bash
# Meta script for pt3-6 (trianglearea+angle+compass)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-6: trianglearea+angle+compass"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-6 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-6/pt3-6_seed1.sh

echo ""
echo "=== Training pt3-6 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-6/pt3-6_seed2.sh

echo ""
echo "=========================================="
echo "PT3-6 (trianglearea+angle+compass) COMPLETE!"
echo "=========================================="
