#!/bin/bash
# Meta script for pt1-6 (perimeter) - both seeds

set -e

echo "=========================================="
echo "PT1-6: perimeter"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt1-6 seed1 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-6/pt1-6_seed1.sh

echo ""
echo "=== Training pt1-6 seed2 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-6/pt1-6_seed2.sh

echo ""
echo "=========================================="
echo "PT1-6 (perimeter) COMPLETE!"
echo "=========================================="
