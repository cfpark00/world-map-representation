#!/bin/bash
# Meta script for pt1-2 (trianglearea) - both seeds

set -e

echo "=========================================="
echo "PT1-2: trianglearea"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt1-2 seed1 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-2/pt1-2_seed1.sh

echo ""
echo "=== Training pt1-2 seed2 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-2/pt1-2_seed2.sh

echo ""
echo "=========================================="
echo "PT1-2 (trianglearea) COMPLETE!"
echo "=========================================="
