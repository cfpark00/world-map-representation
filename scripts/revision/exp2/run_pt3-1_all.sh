#!/bin/bash
# Meta script for pt3-1 (distance+trianglearea+angle)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-1: distance+trianglearea+angle"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-1 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-1/pt3-1_seed1.sh

echo ""
echo "=== Training pt3-1 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-1/pt3-1_seed2.sh

echo ""
echo "=========================================="
echo "PT3-1 (distance+trianglearea+angle) COMPLETE!"
echo "=========================================="
