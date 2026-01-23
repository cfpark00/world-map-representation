#!/bin/bash
# Run all probe generalization evaluations for seed3 ftwb2 models

for i in $(seq 1 21); do
    echo "========================================"
    echo "Running seed3 ftwb2-$i..."
    echo "========================================"
    bash scripts/revision/exp1/probe_generalization/seed3/eval_pt1_seed3_ftwb2-$i.sh
    echo ""
done

echo "All done!"
