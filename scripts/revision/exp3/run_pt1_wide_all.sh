#!/bin/bash
# Run wide model: pretrain + all 7 ftwb fine-tunings

set -e

echo "=========================================="
echo "PT1 WIDE MODEL (2x width, 1/2 epochs)"
echo "=========================================="
echo ""

# Pretrain
echo "=== Pretraining pt1_wide ==="
bash scripts/revision/exp3/pt1_wide/pt1_wide.sh

# Fine-tune on all 7 tasks using ftwb
for task_id in 1 2 3 4 5 6 7; do
  echo ""
  echo "=== Fine-tuning pt1_wide on task ${task_id} (ftwb) ==="
  bash scripts/revision/exp3/pt1_wide/pt1_wide_ftwb${task_id}.sh
done

echo ""
echo "=========================================="
echo "PT1 WIDE MODEL COMPLETE!"
echo "=========================================="
