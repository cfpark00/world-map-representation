#!/bin/bash
# Meta script for pt3-3 (crossing+distance+trianglearea)
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-3: crossing+distance+trianglearea"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-3 seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-3/pt3-3_seed1.sh

echo ""
echo "=== Training pt3-3 seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-3/pt3-3_seed2.sh

echo ""
echo "=========================================="
echo "PT3-3 (crossing+distance+trianglearea) COMPLETE!"
echo "=========================================="
