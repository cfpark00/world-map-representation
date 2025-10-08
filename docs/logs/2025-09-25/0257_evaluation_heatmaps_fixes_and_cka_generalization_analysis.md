# Development Log - 2025-09-25 02:57 - Evaluation Heatmaps Fixes and CKA-Generalization Analysis

## Session Overview
Fixed evaluation heatmap visualizations for fine-tuning experiments, created FTWB3 heatmap script, and developed a new CKA-to-generalization correlation analysis with task reciprocity visualization.

## Main Tasks Completed

### 1. Fixed Evaluation Heatmap Annotations
- **Issue**: Annotation values in difference plots were confusing (positive values shown as red/bad)
- **Initial incorrect fix**: Attempted to negate annotation values while keeping colors
- **Final solution**: Reverted changes, kept `actual - predicted` calculation
- **Colormap fix**: Changed from `coolwarm` to `RdBu` for correct color mapping
- Fixed in 4 scripts: `plot_ft2_heatmap.py`, `plot_ft3_heatmap.py`, `plot_ftwb2_heatmap.py`, `plot_ftwb3_heatmap.py`

### 2. Created FTWB3 Heatmap Visualization
- **New script**: `/scratch/plots/evaluation/plot_ftwb3_heatmap.py`
- 3-task fine-tuning with warmup+bias (matching FT3 task combinations)
- Generates 3 heatmaps:
  - Actual FTWB3 Atlantis performance
  - Predicted performance (max of 3 FTWB1 models)
  - Performance vs Prediction difference
- Includes comparison with FT3 results
- **Key finding**: FTWB3 shows +0.120 average improvement over FT3
  - Trained tasks: +0.008 improvement
  - Untrained tasks (transfer): +0.204 improvement

### 3. CKA-to-Generalization Correlation Analysis
- **Created new directory**: `/scratch/cka_to_generalization/`
- **Script**: `plot_cka_generalization_correlation.py`

#### First Plot: CKA vs Generalization
- X-axis: CKA score between pt1-X and pt1-Y (Layer 5)
- Y-axis: FTWB1-X's normalized performance on task Y
- **Key improvements**:
  - Excluded crossing task due to training instabilities
  - Colors by training task (not target)
  - Added annotations (D→T format) on each point
  - Removed background grid, legend, and title
  - Made text much larger (fontsize 24 for ticks, 18 for labels)
  - Thicker bottom/left spines (2.5 width)
  - Dotted fit line with equation as title
- **Finding**: r=0.354, p=0.055 (borderline significant positive correlation)

#### Second Plot: Task Reciprocity Analysis
- X-axis: Sum of how much task X helps others (Σ X→i performance)
- Y-axis: Sum of how much task X benefits from others (Σ i→X performance)
- 6 points (excluding crossing), each representing one task
- **Key findings**:
  - **Net Givers** (help others more): Perimeter (+2.04), Angle (+1.97), Trianglearea (+1.77)
  - **Net Takers** (benefit more): Inside (-2.31), Compass (-1.81), Distance (-1.66)
- Reveals asymmetric transfer relationships between tasks

## Technical Details

### Heatmap Color Convention Fix
- **Problem**: Coolwarm colormap showed red=positive, blue=negative (counterintuitive)
- **Solution**: Changed to RdBu colormap
  - Now: Red=negative (underperformed), Blue=positive (exceeded expectations)
- Values and colors now align intuitively

### CKA Score Loading
- CKA scores from: `/scratch/cka_analysis_clean/cka_summary.csv`
- Using Layer 5 (most relevant for high-level features)
- PT1 experiments only (single-task models)

### Performance Normalization
- Same normalization as heatmaps:
  - Accuracy tasks: Linear interpolation
  - Error tasks: Log-ratio normalization
- Range: 0 (no improvement) to 1 (standard performance level)

## Files Created/Modified

### Created
- `/scratch/plots/evaluation/plot_ftwb3_heatmap.py`
- `/scratch/cka_to_generalization/plot_cka_generalization_correlation.py`
- `/scratch/cka_to_generalization/cka_vs_generalization_scatter.png`
- `/scratch/cka_to_generalization/task_reciprocity_scatter.png`

### Modified
- `/scratch/plots/evaluation/plot_ft2_heatmap.py` (colormap fix)
- `/scratch/plots/evaluation/plot_ft3_heatmap.py` (colormap fix)
- `/scratch/plots/evaluation/plot_ftwb2_heatmap.py` (colormap fix)
- `/scratch/plots/evaluation/plot_ftwb3_heatmap.py` (colormap fix after creation)

### Regenerated
- All 6 evaluation heatmap PNGs with corrected colormaps

## Key Insights

### Transfer Learning Asymmetry
- Some tasks are natural "teachers" (perimeter, angle, trianglearea)
- Others are natural "students" (inside, compass, distance)
- This asymmetry wasn't captured by CKA similarity alone

### CKA Predictive Power
- Without crossing: CKA explains ~12.5% of variance in generalization (R²=0.125)
- Moderate positive correlation suggests representation similarity matters but isn't everything
- Other factors (task difficulty, feature complexity) likely play major roles

### Warmup+Bias Effectiveness
- FTWB methods consistently outperform standard fine-tuning
- Especially effective for transfer to untrained tasks (+20% for FTWB3 vs FT3)
- Less impact on trained tasks (only +0.8% improvement)

## Next Steps Potential
- Investigate why certain tasks are better "teachers" vs "students"
- Analyze relationship between task reciprocity and task complexity
- Explore other layers' CKA scores for correlation analysis
- Create unified visualization combining all fine-tuning strategies