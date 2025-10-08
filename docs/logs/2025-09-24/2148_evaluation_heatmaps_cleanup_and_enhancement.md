# Development Log - 2025-09-24 21:48 - Evaluation Heatmaps Cleanup and Enhancement

## Session Overview
Enhanced and cleaned up evaluation heatmap visualization scripts in `/scratch/plots/evaluation/`, adding prediction analysis capabilities and fixing normalization display conventions.

## Main Tasks Completed

### 1. Project Structure Review
- Read and reviewed `CLAUDE.md` and `docs/repo_usage.md`
- Examined evaluation plots directory structure
- Reviewed all heatmap visualizations (FT1, FT2, FT3, FTWB1, FTWB2)

### 2. Cleanup of Evaluation Scripts
- **Removed redundant evaluation scripts**:
  - Deleted `eval_checkpoints.py` (ran transformer forward passes, duplicated proper eval in src/)
  - Deleted `eval_rw_checkpoints.py` (similar redundant functionality)
- **Removed non-heatmap visualization files**:
  - Deleted various plot scripts and PNGs not related to heatmaps
  - Kept only heatmap-related files (5 scripts + 5 PNG outputs)
- **Fixed duplicate script**: Removed `plot_ftwb2_heatmap_21.py` (duplicate of `plot_ftwb2_heatmap.py`)

### 3. Added Error Handling to All Heatmap Scripts
- Modified all 5 heatmap scripts to check for missing evaluation data files
- Added `FileNotFoundError` exceptions with descriptive messages
- Ensures scripts fail immediately with clear errors if data is missing
- Verified all scripts run successfully with current data

### 4. Understanding Normalization Method
- **Baseline source**: All scripts use PT1 (pre-training 1) experiment evaluations as baseline
  - Location: `/n/home12/cfpark00/WM_1/data/experiments/pt1/evals/`
  - Contains both standard and Atlantis task evaluations
- **Normalization formula**:
  - For accuracy tasks (crossing, inside, compass): Linear interpolation
    - `score = (value_atlantis - baseline_atlantis) / (baseline_standard - baseline_atlantis)`
  - For error tasks (distance, trianglearea, angle, perimeter): Log-ratio normalization
    - `score = log(baseline_atlantis/value_atlantis) / log(baseline_atlantis/baseline_standard)`
  - Score interpretation: 0 = no improvement, 1 = reached standard performance, >1 = super-generalization
- **Consistency**: All 5 scripts use identical normalization method

### 5. Enhanced FT3 Script with Prediction Analysis
- Added prediction functionality to `plot_ft3_heatmap.py` matching FT2/FTWB2 pattern
- Now generates 3 heatmaps:
  1. **Actual**: Real FT3 performance on Atlantis tasks
  2. **Predicted**: MAX of 3 FT1 single-task models (since FT3 trains on 3 tasks)
  3. **Performance vs Prediction**: Shows where multi-task training helped or hurt
- Prediction hypothesis: Tests if multi-task fine-tuning equals best of single-task models

### 6. Fixed Prediction Error Display Convention
- **Issue**: Prediction error display was counterintuitive
- **Changed calculation** from `predicted - actual` to `actual - predicted` in:
  - `plot_ft2_heatmap.py`
  - `plot_ft3_heatmap.py`
  - `plot_ftwb2_heatmap.py`
- **New convention**:
  - Red/negative = model underperformed prediction
  - Blue/positive = model exceeded prediction
- Updated titles and labels to reflect new interpretation

## Technical Details

### Normalization Rationale
- **Log-ratio for errors**: Handles multiplicative nature of error reduction
  - Reducing error 1000→100 is same significance as 10→1 (both 10x improvement)
  - Prevents large absolute changes from dominating
- **Why not simple percentage**: `(val-base)/base` would overweight large absolute changes

### Prediction Method
- Takes MAX of normalized scores (not raw metrics)
- Mathematically equivalent to taking MIN of error metrics then normalizing
- Chosen approach keeps code uniform for both error and accuracy metrics

## Files Modified
- `/scratch/plots/evaluation/plot_ft1_heatmap.py` - Added error handling
- `/scratch/plots/evaluation/plot_ft2_heatmap.py` - Added error handling, fixed error display
- `/scratch/plots/evaluation/plot_ft3_heatmap.py` - Added prediction analysis, error handling, fixed display
- `/scratch/plots/evaluation/plot_ftwb1_heatmap.py` - Added error handling
- `/scratch/plots/evaluation/plot_ftwb2_heatmap.py` - Added error handling, fixed error display

## Files Removed
- 2 evaluation scripts (eval_checkpoints.py, eval_rw_checkpoints.py)
- 7 non-heatmap plotting scripts
- 3 non-heatmap PNG files
- 1 duplicate script (plot_ftwb2_heatmap_21.py)

## Current State
- All 5 heatmap scripts functioning properly with consistent normalization
- FT2, FT3, and FTWB2 now have prediction analysis capabilities
- Error handling ensures data integrity
- Display convention is intuitive (red/negative = underperformance)

## Next Steps Potential
- Could add prediction analysis to FT1 and FTWB1 (though less meaningful for single-task)
- Could analyze prediction error patterns across experiments
- Could create summary visualization comparing all fine-tuning strategies