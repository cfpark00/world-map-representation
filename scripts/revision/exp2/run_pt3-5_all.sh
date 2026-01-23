#!/bin/bash
# Meta script for pt3-5 (perimeter+crossing+distance)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-5: perimeter+crossing+distance"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-5 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-5/pt3-5_seed1.sh

echo ""
echo "=== Training pt3-5 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-5/pt3-5_seed2.sh

echo ""
echo "=========================================="
echo "PT3-5 (perimeter+crossing+distance) COMPLETE!"
echo "=========================================="
