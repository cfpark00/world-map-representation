#!/bin/bash
# Run narrow model: pretrain + all 7 ftwb fine-tunings

set -e

echo "=========================================="
echo "PT1 NARROW MODEL (1/2 width, 2x epochs)"
echo "=========================================="
echo ""

# Pretrain
echo "=== Pretraining pt1_narrow ==="
bash scripts/revision/exp3/pt1_narrow/pt1_narrow.sh

# Fine-tune on all 7 tasks using ftwb
for task_id in 1 2 3 4 5 6 7; do
  echo ""
  echo "=== Fine-tuning pt1_narrow on task ${task_id} (ftwb) ==="
  bash scripts/revision/exp3/pt1_narrow/pt1_narrow_ftwb${task_id}.sh
done

echo ""
echo "=========================================="
echo "PT1 NARROW MODEL COMPLETE!"
echo "=========================================="
