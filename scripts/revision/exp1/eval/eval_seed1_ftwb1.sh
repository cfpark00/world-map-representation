#!/bin/bash
cd 
for exp in 1 2 3 4 5 6 7; do
  for task in distance trianglearea angle compass inside perimeter crossing; do
    uv run python src/eval/evaluate_checkpoints.py configs/revision/exp1/eval/seed1/ftwb1-${exp}/atlantis_${task}.yaml --overwrite
    uv run python src/eval/evaluate_checkpoints.py configs/revision/exp1/eval/seed1/ftwb1-${exp}/${task}.yaml --overwrite
  done
  uv run python src/eval/evaluate_checkpoints.py configs/revision/exp1/eval/seed1/ftwb1-${exp}/multi_task.yaml --overwrite
done
