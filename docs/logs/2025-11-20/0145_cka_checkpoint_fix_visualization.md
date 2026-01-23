# CKA Checkpoint Fix and Visualization Improvements

**Date:** 2025-11-20
**Time:** 01:45
**Session Duration:** ~2 hours

## Summary
Fixed critical bug in exp4 CKA analysis where checkpoint 98448 was being used instead of the final checkpoint 328146. Regenerated all 105 CKA configs and improved visualization with custom three-slope colormap normalization.

## Main Tasks Completed

### 1. CKA Matrix Visualization Script Debugging
- **Issue Found:** Missing CKA entries in 14x14 matrix visualization
- **Root Cause:** Config directories were named alphabetically (pt1-1_vs_pt1-2_seed1) but plot script looked for task-grouped names (pt1-1_seed1_vs_pt1-2)
- **Fix:** Updated `scratch/plot_14x14_cka_matrix.py` to try both orderings when loading CKA results (since CKA is symmetric)
- **Result:** Successfully loaded all 105/105 CKA pairs

### 2. Inside Task CKA Investigation
- **Observation:** Inside task (pt1-5) showed low CKA (~0.25) with other tasks in current analysis
- **Historical Data:** Old analysis showed pt1-5 vs pt1-2 at CKA=0.378 (final checkpoint)
- **Discovery:** Old CKA results at `/n/home12/cfpark00/datadir/WM_1/data/experiments/cka_analysis/pt1-2_vs_pt1-5_l5/cka_results.json` showed:
  - Checkpoint 98448: CKA = 0.254
  - Checkpoint 328146: CKA = 0.713
- **Root Cause:** exp4 configs used checkpoint 98448 instead of final checkpoint 328146

### 3. Critical Bug Fix: Wrong Checkpoint in CKA Configs
- **Problem:** All 105 exp4 CKA configs in `configs/revision/exp4/cka_cross_seed/` used `checkpoint_steps: [98448]`
- **Correct Value:** Final checkpoint is 328146 (verified by checking `data/experiments/pt1-*/checkpoints/`)
- **Fix Applied:**
  1. Updated `scratch/generate_exp4_cka_configs.py` to use checkpoint 328146
  2. Regenerated all 105 configs
  3. User will re-run: `bash scripts/revision/exp4/cka_analysis/run_14x14_cka.sh`

### 4. 7x7 Averaged Matrix Enhancement
- **Request:** Create 7x7 matrix where each cell averages across seed combinations
- **Implementation:** For each task pair (i,j), average CKA across all 4 seed combinations:
  - (orig, orig)
  - (orig, seed1)
  - (seed1, orig)
  - (seed1, seed1)
- **Diagonal Fix:** For diagonal entries (same task), only average cross-seed comparisons:
  - Include: (orig vs seed1) and (seed1 vs orig)
  - Exclude: (orig vs orig) and (seed1 vs seed1) - these are self-comparisons (CKA=1.0)
- **Output:** Both 14x14 and 7x7 matrices saved as PNG and CSV

### 5. Custom Colormap Normalization
- **Motivation:** Need better visual separation in mid-to-high CKA range (0.4-1.0)
- **Implementation:** Created custom three-slope piecewise linear normalization using `matplotlib.colors.FuncNorm`
- **Final Configuration:**
  - **0.0-0.4**: Very compressed (20% of colormap) - low CKA values look similar
  - **0.4-0.6**: Medium (30% of colormap) - moderate variation
  - **0.6-1.0**: Expanded (50% of colormap) - most color variation for high CKA
- **Benefit:** Better distinction between different levels of cross-task similarity while keeping crossing task (~0.004) visually compressed

### 6. Raw Representation Extraction Setup (Explored but Not Used)
- **Request:** Create "raw" prompt format that feeds just `c_XXXX` and extracts last token
- **Implementation:**
  - Added "raw" format to `src/analysis/analyze_representations_higher.py`
  - Created config generator: `scratch/generate_raw_repr_configs.py`
  - Generated 14 configs (7 original + 7 seed1) in `configs/revision/exp4/representation_extraction_raw/`
  - Created extraction scripts in `scripts/revision/exp4/representation_extraction_raw/`
- **Status:** User decided "nvm this" - deprioritized after setting up

## Key Technical Details

### CKA Computation Consistency
Verified that exp4 CKA v2 and old CKA scripts use identical formulas:
- **Both use:** `CKA = sum(K*L) / (sqrt(sum(K*K)) * sqrt(sum(L*L)))`
- **Both use:** Centered kernels (mean-subtracted)
- **Difference was only:** Checkpoint selection (98448 vs 328146)

### Three-Slope Normalization Function
```python
def three_slope_mapping(x, breakpoint1=0.4, breakpoint2=0.6):
    # Segment 1: 0.0 to 0.4 -> 0.0 to 0.2 (compressed)
    # Segment 2: 0.4 to 0.6 -> 0.2 to 0.5 (medium)
    # Segment 3: 0.6 to 1.0 -> 0.5 to 1.0 (expanded)
```

### Files Modified
- `scratch/generate_exp4_cka_configs.py` - Changed checkpoint from 98448 to 328146
- `scratch/plot_14x14_cka_matrix.py` - Added:
  - Symmetric CKA loading (tries both orderings)
  - 7x7 averaged matrix generation with diagonal cross-seed only
  - Custom three-slope colormap normalization
- `src/analysis/analyze_representations_higher.py` - Added "raw" prompt format
- Generated 105 new configs in `configs/revision/exp4/cka_cross_seed/`

### Scripts Created
- `scratch/generate_raw_repr_configs.py`
- `scripts/revision/exp4/representation_extraction_raw/extract_original_representations.sh`
- `scripts/revision/exp4/representation_extraction_raw/extract_seed1_representations.sh`

## Next Steps
1. User will re-run CKA computation with correct checkpoint: `bash scripts/revision/exp4/cka_analysis/run_14x14_cka.sh`
2. After CKA completes, re-run plotting: `uv run python scratch/plot_14x14_cka_matrix.py`
3. Expected outcome: Inside task should show higher CKA (~0.7-0.8) with other tasks at checkpoint 328146

## Important Notes
- **Checkpoint 98448 vs 328146:** Representations change significantly during continued training
- Inside task CKA went from ~0.25 (step 98448) to ~0.71 (step 328146) in old analysis
- This explains the discrepancy between current and historical CKA values
- The 98448 checkpoint was incorrectly hardcoded in the config generator
