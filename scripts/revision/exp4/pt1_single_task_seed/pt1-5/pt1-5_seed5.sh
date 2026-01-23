#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
uv run python src/training/train.py configs/revision/exp4/pt1_single_task_seed/pt1-5/pt1-5_seed5.yaml --overwrite
