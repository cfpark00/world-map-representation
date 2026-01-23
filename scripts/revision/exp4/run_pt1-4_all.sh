#!/bin/bash
# Meta script for pt1-4 (compass) - both seeds

set -e

echo "=========================================="
echo "PT1-4: compass"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt1-4 seed1 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-4/pt1-4_seed1.sh

echo ""
echo "=== Training pt1-4 seed2 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-4/pt1-4_seed2.sh

echo ""
echo "=========================================="
echo "PT1-4 (compass) COMPLETE!"
echo "=========================================="
