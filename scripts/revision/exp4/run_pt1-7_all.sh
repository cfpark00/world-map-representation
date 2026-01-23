#!/bin/bash
# Meta script for pt1-7 (crossing) - both seeds

set -e

echo "=========================================="
echo "PT1-7: crossing"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt1-7 seed1 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-7/pt1-7_seed1.sh

echo ""
echo "=== Training pt1-7 seed2 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-7/pt1-7_seed2.sh

echo ""
echo "=========================================="
echo "PT1-7 (crossing) COMPLETE!"
echo "=========================================="
