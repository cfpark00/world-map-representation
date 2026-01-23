# 17:06 - Exp3 Plot Styling Update

## Summary
Updated exp3 plotting scripts to match exp1 publication-ready styling, including removal of labels, reordering of experiments, and consistent formatting.

## Tasks Completed

### 1. Reviewed Exp1 Styling
Read exp1 plotting scripts to understand the exact styling:
- `plot_revision_exp1_ftwb1_heatmaps.py`
- `plot_revision_exp1_ftwb2_heatmaps.py`
- `plot_revision_exp1_ftwb2_vs_ftwb1.py`

### 2. Updated Exp3 FTWB1 Heatmap Script
**File**: `src/scripts/plot_revision_exp3_ftwb1_heatmaps.py`
- Changed `linewidths=0, linecolor='none'` (no cell borders)
- Changed `cbar_kws={"shrink": 1.0, "aspect": 20}` (tall colorbar)
- Changed `annot_kws={"fontsize": 13.2, "fontweight": "bold"}`
- Removed all tick labels: `ax.set_xticklabels([])`, `ax.set_yticklabels([])`
- Removed axis labels: `ax.set_xlabel("")`, `ax.set_ylabel("")`
- Added `ax.tick_params(left=False, top=False)`
- Added 'T' markers on diagonal for trained tasks

### 3. Updated Exp3 FTWB2 Heatmap Script
**File**: `src/scripts/plot_revision_exp3_ftwb2_heatmaps.py`
- Transposed matrix: rows=experiments, cols=tasks (matching exp1)
- Changed `cbar=False` (no colorbar)
- Changed `linewidths=0, linecolor='none'`
- Changed `annot_kws={"fontsize": 13.2, "fontweight": "bold"}`
- Removed all tick labels and axis labels
- Fixed 'T' marker positions for transposed matrix

### 4. Updated Exp3 FTWB2 vs FTWB1 Script
**File**: `src/scripts/plot_revision_exp3_ftwb2_vs_ftwb1.py`
- Same changes as FTWB2 heatmap script
- Transposed difference matrix
- Fixed statistics calculation for transposed indexing

### 5. Reordered FTWB2 Experiments
Per user request, reordered `FTWB2_EXPS` to put distance-containing experiments on top 3 rows:
- **Before**: `[2, 4, 9, 12, 13, 15]`
- **After**: `[4, 13, 15, 2, 9, 12]`

New row order:
- Row 1: exp 4 (crossing, distance)
- Row 2: exp 13 (distance, perimeter)
- Row 3: exp 15 (compass, distance)
- Row 4: exp 2 (angle, compass)
- Row 5: exp 9 (compass, trianglearea)
- Row 6: exp 12 (crossing, inside)

### 6. Regenerated Plots
All 3 plots regenerated with new styling:
- `wide_ftwb1_evaluation_heatmap.png`
- `wide_ftwb2_evaluation_heatmap.png`
- `wide_ftwb2_vs_ftwb1.png`

## Results
**Wide model statistics (unchanged):**
- FTWB1: Trained=0.803±0.146, Transfer=0.491±0.267
- FTWB2: Trained=0.856±0.186, Transfer=0.496±0.274
- FTWB2-FTWB1 diff: Trained=+0.042, Transfer=-0.056, Overall=-0.028

## Files Modified
- `src/scripts/plot_revision_exp3_ftwb1_heatmaps.py`
- `src/scripts/plot_revision_exp3_ftwb2_heatmaps.py`
- `src/scripts/plot_revision_exp3_ftwb2_vs_ftwb1.py`

## Output
- `data/experiments/revision/exp3/plots/wide_ftwb1_evaluation_heatmap.png`
- `data/experiments/revision/exp3/plots/wide_ftwb2_evaluation_heatmap.png`
- `data/experiments/revision/exp3/plots/wide_ftwb2_vs_ftwb1.png`
