#!/bin/bash
# Meta script for pt1-3 (angle) - both seeds

set -e

echo "=========================================="
echo "PT1-3: angle"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt1-3 seed1 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-3/pt1-3_seed1.sh

echo ""
echo "=== Training pt1-3 seed2 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-3/pt1-3_seed2.sh

echo ""
echo "=========================================="
echo "PT1-3 (angle) COMPLETE!"
echo "=========================================="
