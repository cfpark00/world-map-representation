#!/bin/bash
# Meta script for pt3-7 (inside+perimeter+crossing)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-7: inside+perimeter+crossing"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-7 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-7/pt3-7_seed1.sh

echo ""
echo "=== Training pt3-7 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-7/pt3-7_seed2.sh

echo ""
echo "=========================================="
echo "PT3-7 (inside+perimeter+crossing) COMPLETE!"
echo "=========================================="
