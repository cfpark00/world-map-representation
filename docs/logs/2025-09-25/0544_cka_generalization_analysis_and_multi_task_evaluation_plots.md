# Development Log - 2025-09-25 05:44 - CKA Generalization Analysis and Multi-Task Evaluation Plots

## Session Overview
Extended the CKA-to-generalization analysis with FT1 support, transposed heatmap layouts, and created comprehensive multi-task evaluation plots for FT2/FT3 and FTWB2/FTWB3 experiments with performance vs prediction comparisons.

## Main Tasks Completed

### 1. Enhanced CKA Analysis Scripts with FT1 Support
- **Added argparse support**: Modified both correlation and heatmap scripts to handle FT1 vs FTWB1
- **Created dual data loaders**: `load_ft1_performance()` and `load_ftwb1_performance()`
- **Consistent naming**: Output files now prefixed with experiment type (`ft1_`, `ftwb1_`)
- **Key findings comparison**:
  - FT1 correlation: r=0.273 (weaker than FTWB1's r=0.354)
  - Transfer performance: FT1=0.359 vs FTWB1=0.449 (25% improvement with warmup+bias)

### 2. Transposed Heatmap Layout with Top X-Axis
- **Matrix transposition**: Changed from (experiments × tasks) to (tasks × experiments)
  - Rows now represent tasks being evaluated
  - Columns represent trained models/experiments
- **Top x-axis positioning**: Both correlation and evaluation heatmaps now have x-axis labels on top
- **Consistent 'T' markers**: Training task indicators in top-left corner of cells
- **Improved readability**: Matches standard data visualization conventions

### 3. Multi-Task Evaluation Plot System
- **Created comprehensive script**: `plot_multi_task_evaluation.py` for FT2/FT3 and FTWB2/FTWB3
- **Dual-panel design**:
  - Top panel: Actual Atlantis performance (RdYlGn colormap, 0-1 scale)
  - Bottom panel: Performance vs Prediction difference (RdBu colormap, centered at 0)
- **Prediction methodology**: Max performance from single-task models trained on constituent tasks
- **Four complete visualizations**: FT2, FT3, FTWB2, FTWB3 with full analysis

### 4. Experiment Documentation and Mapping
- **Created comprehensive mapping file**: `experiment_task_mapping.txt`
- **Complete experiment documentation**:
  - Task descriptions and mappings (1-7 → task names)
  - FT1/FTWB1: 7 single-task experiments
  - FT2/FTWB2: 21 two-task combinations
  - FT3/FTWB3: 7 three-task combinations
- **Interpretation guides**: Colormap meanings, 'T' markers, performance scales

### 5. Visual Design Refinements
- **Colorbar optimization**: Reduced to 40% height (shrink=0.4) for proper proportions
- **Font size adjustments**: Larger annotations (10pt) for FT3/FTWB3 plots vs 9pt for FT2/FTWB2
- **Spacing optimization**:
  - FT2/FTWB2: hspace=0.08 (compact)
  - FT3/FTWB3: hspace=0.25 (prevents title overlap)
- **Figure dimensions**: Adaptive width (14" for 21 experiments, 8" for 7 experiments)

## Technical Details

### Heatmap Matrix Transformation
```python
# Old: matrix[i, j] = normalized (experiments × tasks)
# New: matrix[j, i] = normalized (tasks × experiments)
```

### Multi-Task Prediction Logic
- For each FT2/FT3 experiment trained on tasks [A, B, (C)]:
- For evaluation on task X: max(FT1-A→X, FT1-B→X, [FT1-C→X])
- Enables "Performance vs Prediction" difference analysis

### Key Performance Findings
- **FT2**: Trained 0.886, Transfer 0.367 (worse than expected: -0.117)
- **FT3**: Trained 0.912, Transfer 0.502 (slightly worse: -0.044)
- **FTWB2**: Trained 0.900, Transfer 0.585 (matches expectations: +0.002)
- **FTWB3**: Trained 0.920, Transfer 0.706 (exceeds expectations: +0.036)

## Files Created/Modified

### Created
- `plot_multi_task_evaluation.py` - Comprehensive multi-task evaluation visualization
- `experiment_task_mapping.txt` - Complete experiment and task mapping documentation
- Multi-task evaluation plots:
  - `ft2_evaluation_plot.png`
  - `ft3_evaluation_plot.png`
  - `ftwb2_evaluation_plot.png`
  - `ftwb3_evaluation_plot.png`

### Modified
- `plot_cka_generalization_correlation.py` - Added FT1 support, removed symmetric plots
- `plot_generalization_heatmap.py` - Renamed from `plot_ftwb1_heatmap.py`, added FT1 support
- Enhanced with argparse, dual data loading, transposed matrix layout

### Regenerated with New Naming
- `ft1_cka_vs_generalization_scatter.png`
- `ft1_task_reciprocity_scatter.png`
- `ft1_generalization_heatmap.png`
- `ftwb1_cka_vs_generalization_scatter.png`
- `ftwb1_task_reciprocity_scatter.png`
- `ftwb1_generalization_heatmap.png`

## Key Insights

### Transfer Learning Patterns
- **Warmup+bias consistently outperforms**: FTWB methods show better transfer than standard FT
- **Multi-task benefit increases with tasks**: 3-task models show stronger transfer than 2-task
- **Prediction accuracy varies**: FTWB methods meet/exceed predictions while FT methods fall short

### Task Asymmetry Consistency
- Same "giver" vs "taker" patterns across FT1/FTWB1:
  - Net givers: perimeter, angle, trianglearea
  - Net takers: inside, compass, distance
- Transfer relationships appear fundamental to task structure, not training method

### Visualization Design Lessons
- Matrix transposition improves interpretability (tasks as rows, experiments as columns)
- Adaptive figure sizing essential for different experiment counts (21 vs 7)
- Top x-axis positioning works better for wide heatmaps
- Proper colorbar scaling crucial for multi-panel layouts

## Next Steps Potential
- Investigate why FTWB methods exceed single-task predictions while FT methods don't
- Analyze correlation between task reciprocity patterns and multi-task performance
- Create unified comparison plots across all fine-tuning strategies
- Develop prediction models incorporating task interaction effects