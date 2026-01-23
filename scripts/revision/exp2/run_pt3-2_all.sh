#!/bin/bash
# Meta script for pt3-2 (compass+inside+perimeter)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-2: compass+inside+perimeter"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-2 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-2/pt3-2_seed1.sh

echo ""
echo "=== Training pt3-2 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-2/pt3-2_seed2.sh

echo ""
echo "=========================================="
echo "PT3-2 (compass+inside+perimeter) COMPLETE!"
echo "=========================================="
