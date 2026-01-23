#!/bin/bash
# Meta script for pt3-8 (distance+trianglearea+compass)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-8: distance+trianglearea+compass"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-8 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-8/pt3-8_seed1.sh

echo ""
echo "=== Training pt3-8 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-8/pt3-8_seed2.sh

echo ""
echo "=========================================="
echo "PT3-8 (distance+trianglearea+compass) COMPLETE!"
echo "=========================================="
