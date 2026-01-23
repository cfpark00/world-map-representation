# Exp4 CKA Analysis Completion and Plotting Fixes

**Date:** 2025-11-21
**Time:** 01:40
**Session Duration:** ~1.5 hours

## Context

User asked about the status of exp4 (revision experiment 4) CKA analysis for layers 3 and 6. Previous developer had indicated certain scripts needed to be run, but the status was unclear.

## Tasks Completed

### 1. Status Assessment of Exp4 CKA Analysis

**Initial Investigation:**
- Explored revision exp4 structure to understand the 21×21 CKA matrix setup
- Exp4 tests PT1 single-task seed robustness: 7 tasks × 3 seeds = 21 models
- Special case: pt1-5 (inside task) uses seed3 instead of seed2 (seed2 training failed)

**First Status Check:**
- Layer 3: 206/231 complete (89%)
- Layer 6: 231/231 complete (100%)

**Discovered Issue:**
- User had already run layer3 analyses, reducing missing count from 55 to 25

**Final Status After User Runs:**
- Layer 3: 231/231 complete (100%) ✅
- Layer 6: 231/231 complete (100%) ✅

### 2. Created Missing Layer3 Completion Script

Generated `scripts/revision/exp4/cka_analysis/FINAL_run_missing_layer3.sh`:
- Contains 25 remaining layer3 CKA pair analyses
- User confirmed this completed the layer3 analysis

### 3. Created Plotting Scripts for Layers 3 and 6

**Scripts Created:**
- `scripts/revision/exp4/cka_analysis/plot_21x21_cka_l3.sh`
- `scripts/revision/exp4/cka_analysis/plot_21x21_cka_l6.sh`

**Python Plotting Scripts:**
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l3.py`
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l6.py`

**Plots Generated for Each Layer:**
- Full 21×21 CKA matrix with all seed comparisons
- 7×7 task-averaged matrix (without SEM)
- 7×7 task-averaged matrix (with SEM)
- Bar plot comparing intra-task vs inter-task CKA

### 4. Fixed Bar Plot Issues Across All Layers

**Issues Identified:**
- Bars were overlapping due to empty category labels `['', '']`
- Missing significance testing
- Layer 6 missing 7×7 plots with SEM
- Used std instead of SEM for error bars in some layers

**Fixes Applied to Layers 3, 4, 5, 6:**

1. **Fixed Bar Overlap:**
   - Changed from empty labels to descriptive multi-line labels
   - Added sample sizes to labels

2. **Added Significance Testing:**
   - Implemented Welch's t-test (unequal variances)
   - Added significance brackets above bars
   - Labels: ns (p≥0.05), * (p<0.05), ** (p<0.01), *** (p<0.001)
   - Added p-values to plot titles

3. **Standardized Error Bars:**
   - Changed all plots to use SEM (standard error of mean) instead of SD
   - Consistent error bar representation across layers

4. **Enhanced Visualizations:**
   - Added grid lines for better readability
   - Proper axis labels and titles
   - Values displayed on top of bars

5. **Fixed Layer 6 Missing Plots:**
   - Added SEM calculation to 7×7 averaging
   - Created 7×7 with SEM plot (was missing)
   - Fixed diagonal cross-seed counting (changed from 6 to 3 unique pairs)

**Scripts Updated:**
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l3.py`
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l4.py`
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix.py` (layer 5)
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l6.py`

### 5. Regenerated All Plots

**All layers successfully regenerated with fixes:**

**Layer 3:**
- Intra-task mean: 0.338, Inter-task mean: 0.259
- Welch's t-test: p=0.313 (ns)
- All 6 plot types generated

**Layer 4:**
- Intra-task mean: 0.672, Inter-task mean: 0.589
- Welch's t-test: p=0.288 (ns)
- All 6 plot types generated

**Layer 5:**
- Intra-task mean: 0.796, Inter-task mean: 0.698
- Welch's t-test: p=0.122 (ns)
- All 6 plot types generated

**Layer 6:**
- Intra-task mean: 0.797, Inter-task mean: 0.714
- Welch's t-test: p=0.146 (ns)
- All 6 plot types generated (now includes with_sem)

## Key Findings

### CKA Analysis Results Summary

**Seed Robustness Across Layers:**
- All layers show non-significant differences between intra-task and inter-task CKA
- Higher layers (4-6) show stronger overall similarity than lower layer (3)
- Layer 3: Lower CKA values overall (intra: 0.34, inter: 0.26)
- Layers 5-6: High CKA values (intra/inter both ~0.7-0.8)

**Interpretation:**
- Representations become more similar across tasks in higher layers
- Seed variation within tasks comparable to task variation across seeds
- Suggests learned representations are relatively robust to random initialization

## Files Created/Modified

### Created:
- `scripts/revision/exp4/cka_analysis/FINAL_run_missing_layer3.sh`
- `scripts/revision/exp4/cka_analysis/plot_21x21_cka_l3.sh`
- `scripts/revision/exp4/cka_analysis/plot_21x21_cka_l6.sh`
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l3.py`
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l6.py`

### Modified:
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l4.py`
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix.py` (layer 5)
- All plotting scripts updated with significance testing and proper bar spacing

### Outputs Generated:
For each layer (3, 4, 5, 6):
- `cka_matrix_21x21_layer[N].png`
- `cka_matrix_21x21_layer[N].csv`
- `cka_matrix_7x7_averaged_layer[N].png`
- `cka_matrix_7x7_averaged_layer[N]_with_sem.png`
- `cka_matrix_7x7_averaged_layer[N].csv`
- `cka_barplot_intra_vs_inter_layer[N].png`

All saved to: `data/experiments/revision/exp4/cka_analysis/`

## Technical Details

### 21×21 Matrix Setup
- 7 tasks: distance, trianglearea, angle, compass, inside, perimeter, crossing
- 3 seeds per task: original (42), seed1, seed2 (except pt1-5 uses seed3)
- Total: 231 unique comparison pairs (21×21 symmetric matrix)

### Statistical Testing
- Method: Welch's t-test (appropriate for unequal variances)
- Comparison: Intra-task (same task, different seeds) vs Inter-task (different tasks, same seed)
- All layers showed non-significant differences (p > 0.05)

## Next Steps

Research continues with completed CKA analysis infrastructure for all layers (3, 4, 5, 6) of the 21×21 seed robustness experiment.
