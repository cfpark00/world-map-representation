#!/bin/bash
# Meta script for pt1-5 (inside) - both seeds

set -e

echo "=========================================="
echo "PT1-5: inside"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt1-5 seed1 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-5/pt1-5_seed1.sh

echo ""
echo "=== Training pt1-5 seed2 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-5/pt1-5_seed2.sh

echo ""
echo "=========================================="
echo "PT1-5 (inside) COMPLETE!"
echo "=========================================="
