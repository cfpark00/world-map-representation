#!/bin/bash
cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1
uv run python src/training/train.py configs/revision/exp2/pt3_seed/pt3-8/pt3-8_seed2.yaml --overwrite
