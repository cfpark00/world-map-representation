#!/bin/bash
# Meta script for pt1-1 (distance) - both seeds

set -e

echo "=========================================="
echo "PT1-1: distance"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt1-1 seed1 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-1/pt1-1_seed1.sh

echo ""
echo "=== Training pt1-1 seed2 ==="
bash scripts/revision/exp4/pt1_single_task_seed/pt1-1/pt1-1_seed2.sh

echo ""
echo "=========================================="
echo "PT1-1 (distance) COMPLETE!"
echo "=========================================="
