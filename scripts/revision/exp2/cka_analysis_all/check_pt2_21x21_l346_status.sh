#!/bin/bash
# Chunk 4: Verification and summary
# Check completion status and summarize results

echo "Checking completion status for PT2 21×21 CKA analysis (layers 3, 4, 6)..."
echo ""

for layer in 3 4 6; do
    echo "=== Layer $layer ==="
    total=$(ls configs/revision/exp2/cka_analysis_all/*_l${layer}.yaml 2>/dev/null | wc -l)
    completed=$(find data/experiments/revision/exp2/cka_analysis_all/ -name "layer${layer}" -type d | wc -l)
    echo "Total pairs: $total"
    echo "Completed: $completed"
    echo "Remaining: $((total - completed))"
    echo ""
done

echo "Summary:"
echo "--------"
echo "Layer 3: 210 pairs (all PT2 21×21 combinations)"
echo "Layer 4: 210 pairs (all PT2 21×21 combinations)"
echo "Layer 6: 210 pairs (all PT2 21×21 combinations)"
echo ""
echo "Total: 630 CKA computations"
echo ""
echo "To run in parallel, execute:"
echo "  bash scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk1.sh &"
echo "  bash scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk2.sh &"
echo "  bash scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk3.sh &"
