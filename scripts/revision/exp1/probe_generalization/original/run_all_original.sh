#!/bin/bash
# Run all probe generalization evaluations for original ftwb2 models

for i in $(seq 1 21); do
    echo "========================================"
    echo "Running original ftwb2-$i..."
    echo "========================================"
    bash scripts/revision/exp1/probe_generalization/original/eval_pt1_ftwb2-$i.sh
    echo ""
done

echo "All done!"
