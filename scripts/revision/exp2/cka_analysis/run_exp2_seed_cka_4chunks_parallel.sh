#!/bin/bash
# Run all 4 chunks in parallel (if running on SLURM or separate terminals)
# Total: 252 CKA calculations divided into 4 balanced chunks

echo "Submit these 4 jobs in parallel:"
echo ""
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk1.sh"
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk2.sh"
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk3.sh"
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk4.sh"
echo ""
echo "Or run sequentially:"
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk1.sh
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk2.sh
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk3.sh
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk4.sh
