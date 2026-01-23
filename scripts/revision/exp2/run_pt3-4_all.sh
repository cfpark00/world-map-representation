#!/bin/bash
# Meta script for pt3-4 (angle+compass+inside)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-4: angle+compass+inside"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-4 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-4/pt3-4_seed1.sh

echo ""
echo "=== Training pt3-4 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-4/pt3-4_seed2.sh

echo ""
echo "=========================================="
echo "PT3-4 (angle+compass+inside) COMPLETE!"
echo "=========================================="
