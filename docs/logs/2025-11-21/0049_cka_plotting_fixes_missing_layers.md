# 00:49 - CKA Plotting Fixes and Missing Layer Calculations

## Summary
Fixed CKA 21×21 matrix plotting scripts and identified/resolved missing layer 3 and layer 6 CKA calculations.

## Tasks Completed

### 1. CKA 7×7 Matrix Plotting Updates
- **Removed axis labels and titles** from 7×7 averaged matrix plots (layer 4 and 5)
- **Increased cell font size** from 9pt to 24pt for better readability
- **Added SEM display** in each cell showing mean ± SEM
- **Created two versions** of 7×7 plots:
  - With SEM: `cka_matrix_7x7_averaged_layer{4,5}_with_sem.png` (fontsize 18, shows mean ± SEM)
  - Without SEM: `cka_matrix_7x7_averaged_layer{4,5}.png` (fontsize 24, shows mean only)
- **Fixed SEM calculation**: Corrected diagonal cells to use n=3 (unique pairs) instead of n=6 (was double-counting symmetric pairs)
  - Diagonal: orig vs seed1, orig vs seed2, seed1 vs seed2 = 3 unique pairs
  - Off-diagonal: 3×3 = 9 pairs

### 2. Bar Plot Updates
- **Changed inter-task definition**: Now compares different tasks with SAME seed (not all seed combinations)
  - Intra-task: Same task, different seeds (seed robustness)
  - Inter-task: Different tasks, same seed (task differentiation)
- **Removed all labels and grid** from bar plots
- **Changed error bars from STD to SEM** for both bars
- **Moved numbers to middle of bars** with white text for visibility
- **Removed title and axis labels** for cleaner appearance

### 3. Missing CKA Calculations Investigation
**Discovered missing calculations for 21×21 matrix:**
- **Layer 3**: ALL 231 pairs missing (0% complete)
- **Layer 4**: ✅ Complete (231/231)
- **Layer 5**: ✅ Complete (231/231)
- **Layer 6**: 21 pairs missing (90.9% complete) - all involving pt1-5_seed3

**Root cause analysis:**
- All representation extractions ARE complete for layers 3, 4, 5, 6 (21 models × 4 layers)
- CKA calculations just weren't run for layer 3 and partially missing for layer 6

### 4. Config Generation for Missing CKA Calculations
Created Python scripts to generate missing configs:
- `src/scripts/generate_missing_cka_configs_21x21.py` - Generated 231 layer 3 configs
- `src/scripts/generate_missing_l6_configs.py` - Generated 21 layer 6 configs for pt1-5_seed3

Total configs generated: 252 (231 layer 3 + 21 layer 6)

### 5. Execution Scripts Creation
Created 4 bash scripts to run missing CKA calculations:
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part1.sh` - First ~87 layer 3 configs
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part2.sh` - Middle ~86 layer 3 configs
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part3.sh` - Last ~86 layer 3 configs
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer6.sh` - All 21 layer 6 configs

**Bug fixes during script creation:**
1. Initially called wrong script: `src/analysis/cka_v2/compute_cka.py` (just a library, no main)
2. Then tried: `src/analysis/compute_cka_from_representations.py` (uses old config format)
3. **Final correct script**: `src/scripts/analyze_cka_pair.py` (matches layer 4/5 implementation)
4. Added `--overwrite` flag to all scripts for automatic recomputation

### 6. Documentation
Created `MISSING_CKA_SUMMARY.md` documenting:
- Missing calculation summary by layer
- Generated config locations
- Execution script usage
- Technical notes about config counts

## Files Modified

### Scripts (4 new bash scripts):
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part1.sh`
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part2.sh`
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part3.sh`
- `scripts/revision/exp4/cka_analysis/run_missing_cka_layer6.sh`

### Python Scripts (2 new generators):
- `src/scripts/generate_missing_cka_configs_21x21.py`
- `src/scripts/generate_missing_l6_configs.py`

### Plotting Scripts (2 modified):
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix.py` (layer 5)
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l4.py` (layer 4)

### Documentation:
- `MISSING_CKA_SUMMARY.md` (new)

### Configs Generated:
- 231 layer 3 configs in `configs/revision/exp4/cka_cross_seed/*/layer3.yaml`
- 21 layer 6 configs in `configs/revision/exp4/cka_cross_seed/*pt1-5_seed3*/layer6.yaml`

## Status

**Ready to Execute:**
All 4 scripts are ready to run. User can execute them in parallel or sequentially:
```bash
bash scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part1.sh
bash scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part2.sh
bash scripts/revision/exp4/cka_analysis/run_missing_cka_layer3_part3.sh
bash scripts/revision/exp4/cka_analysis/run_missing_cka_layer6.sh
```

**Expected Runtime:** ~several hours for all layer 3 calculations, ~30 min for layer 6

**Next Steps After Execution:**
1. Generate layer 3 21×21 and 7×7 visualization matrices
2. Generate layer 6 21×21 and 7×7 visualization matrices
3. Compare layer 3/4/5/6 patterns for paper figures

## Technical Notes

- CKA configs use `exp1`/`exp2` format with `repr_dir` paths
- Execution uses `src/scripts/analyze_cka_pair.py` with `--overwrite` flag
- All configs target checkpoint 328146 (final checkpoint)
- GPU acceleration enabled (`use_gpu: true`)
- City filter excludes Atlantis for CKA computation
- Results saved to `data/experiments/revision/exp4/cka_analysis/{pair}/layer{X}/`
