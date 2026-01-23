#!/bin/bash
# Run all probe generalization evaluations for ftwb2 models

for i in $(seq 1 21); do
    echo "========================================"
    echo "Running ftwb2-$i..."
    echo "========================================"
    bash scripts/revision/exp1/probe_generalization/eval_pt1_seed1_ftwb2-$i.sh
    echo ""
done

echo "All done!"
