PT2 All Pairs CKA Analysis
==========================

Generated: 2025-11-21

Overview:
- Computes CKA for ALL PT2 pairs (including overlapping tasks)
- 21 total PT2 experiments: 7 models × 3 seeds (orig, seed1, seed2)
- 210 unique pairs per layer
- 840 total CKA calculations (210 pairs × 4 layers)

Structure:
- Configs: configs/revision/exp2/cka_analysis_all/
- Output: data/experiments/revision/exp2/cka_analysis_all/
- Scripts: scripts/revision/exp2/cka_analysis_all/

Execution Options:

1. Run all 4 chunks sequentially:
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_all.sh

2. Run individual chunks (210 configs each):
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_chunk1.sh
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_chunk2.sh
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_chunk3.sh
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_chunk4.sh

3. Run by layer (210 configs per layer):
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_l3.sh
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_l4.sh
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_l5.sh
   bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_l6.sh

Differences from previous exp2 CKA:
- Previous: Only non-overlapping task pairs (14 model pairs × 3 seed combos = 42 directories)
- This: ALL pairs including overlapping tasks (210 pairs total)
- Previous output: data/experiments/revision/exp2/cka_analysis/
- This output: data/experiments/revision/exp2/cka_analysis_all/
