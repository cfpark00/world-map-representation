# Exp1 Heatmap Styling Updates

**Date:** 2025-11-21
**Time:** 04:36
**Session:** Continued from previous context

## Overview
Updated all FTWB1, FTWB2, and FTWB2 vs FTWB1 heatmaps to have consistent, publication-ready styling with transposed matrices (rows=experiments), removed labels, and added training task markers.

## Tasks Completed

### 1. FTWB2 Heatmap Styling (`plot_revision_exp1_ftwb2_heatmaps.py`)

**Changes:**
- **Transposed matrix**: 21 experiments × 7 tasks (rows = trained on, cols = evaluated on)
- Removed colorbar (`cbar=False`)
- Removed all axis labels, tick labels, and titles
- Removed cell separator lines (`linewidths=0`)
- Increased annotation font to 13.2 (1.2x bigger)
- Adjusted figure size to (10, 16) for vertical layout
- Added aggregated heatmap averaging across all 4 seeds

**Files generated:**
- `original_ftwb2_evaluation_heatmap.png`
- `seed1_ftwb2_evaluation_heatmap.png`
- `seed2_ftwb2_evaluation_heatmap.png`
- `seed3_ftwb2_evaluation_heatmap.png`
- `aggregated_ftwb2_evaluation_heatmap.png`

**Results (Aggregated):**
- Trained tasks: 0.897 ± 0.142
- Transfer tasks: 0.601 ± 0.278

### 2. FTWB1 Heatmap Styling (`plot_revision_exp1_ftwb1_heatmaps.py`)

**Changes:**
- Added 'T' markers on diagonal to indicate which task each model was trained on
- Markers positioned at (i + 0.15, i + 0.15) for each diagonal cell
- Applied to both individual seed and aggregated plots

**Files updated:**
- `original_ftwb1_evaluation_heatmap.png`
- `seed1_ftwb1_evaluation_heatmap.png`
- `seed2_ftwb1_evaluation_heatmap.png`
- `seed3_ftwb1_evaluation_heatmap.png`
- `aggregated_ftwb1_evaluation_heatmap.png`

**Results (Aggregated):**
- Trained task (diagonal): 0.838 ± 0.136
- Transfer (off-diagonal): 0.474 ± 0.292

### 3. FTWB2 vs FTWB1 Comparison Styling (`plot_revision_exp1_ftwb2_vs_ftwb1.py`)

**Changes:**
- **Transposed matrices**: Both actual and prediction matrices now 21 × 7
- Removed colorbar
- Removed all axis labels, tick labels, and titles
- Removed cell separator lines
- Increased annotation font to 13.2
- Adjusted figure size to (10, 16)
- Added aggregated plot averaging across all 4 seeds
- Updated statistics calculation to use transposed indices

**Files generated:**
- `original_ftwb2_vs_ftwb1.png`
- `seed1_ftwb2_vs_ftwb1.png`
- `seed2_ftwb2_vs_ftwb1.png`
- `seed3_ftwb2_vs_ftwb1.png`
- `aggregated_ftwb2_vs_ftwb1.png`

**Results (Aggregated):**
- Trained tasks actual: 0.897 ± 0.142
- Trained tasks diff: +0.054 ± 0.095 (FTWB2 > FTWB1 prediction)
- Transfer tasks actual: 0.601 ± 0.278
- Transfer tasks diff: -0.019 ± 0.162 (FTWB2 < FTWB1 prediction)
- Overall diff: +0.002 ± 0.149 (essentially matches prediction)

## Key Insights

### Performance Comparison
1. **FTWB1 (single task)**:
   - Trained task: 0.838 (strong learning on trained task)
   - Transfer: 0.474 (moderate transfer to untrained tasks)

2. **FTWB2 (two tasks)**:
   - Trained tasks: 0.897 (excellent learning on both trained tasks)
   - Transfer: 0.601 (good transfer to untrained tasks)

3. **FTWB2 vs FTWB1 Prediction**:
   - Trained tasks: +0.054 (slight synergy beyond max of single tasks)
   - Transfer: -0.019 (slight interference on transfer tasks)
   - Overall: +0.002 (matches prediction on average)

### Interpretation
- Training on 2 tasks together achieves approximately the max performance of training on each task separately
- Slight positive synergy for trained tasks (better than best single-task model)
- Slight negative interference for transfer tasks (worse than best single-task model)
- Overall, multi-task learning neither significantly helps nor hurts compared to single-task baselines

## Technical Details

### Matrix Orientation
- **Old**: 7 tasks × 21 experiments (row = task, col = experiment)
- **New**: 21 experiments × 7 tasks (row = experiment, col = task)
- **Rationale**: More conventional to have rows represent experimental conditions

### Styling Consistency
All heatmaps now share:
- No colorbar
- No labels or titles
- No cell separators
- Font size 13.2 for annotations
- Square cells
- 'T' markers for trained tasks
- Clean, publication-ready appearance

### Color Schemes
- FTWB1/FTWB2 actual performance: RdYlGn (red=poor, yellow=medium, green=good)
- FTWB2 vs FTWB1 difference: RdBu (red=worse than predicted, blue=better than predicted)

## Files Modified

1. `/src/scripts/plot_revision_exp1_ftwb2_heatmaps.py`
   - Transposed matrix creation
   - Removed colorbar
   - Updated figure size and styling
   - Added aggregated plot function

2. `/src/scripts/plot_revision_exp1_ftwb1_heatmaps.py`
   - Added 'T' markers on diagonal

3. `/src/scripts/plot_revision_exp1_ftwb2_vs_ftwb1.py`
   - Transposed both actual and prediction matrices
   - Removed colorbar
   - Updated figure size and styling
   - Added aggregated plot function
   - Fixed statistics calculation for transposed matrices

## Output Location
All plots saved to: `/n/home12/cfpark00/datadir/WM_1/data/experiments/revision/exp1/plots/`

## Notes
- All changes maintain backward compatibility with existing data loading
- Statistics calculations updated to handle transposed matrices correctly
- Aggregated plots provide robust summary across all seeds
- Clean styling ready for paper figures
