# Development Log: Fine-Tuning Evaluation Analysis and Visualization
**Date**: 2025-09-21
**Time**: 15:23
**Focus**: Comprehensive analysis of PT1 fine-tuning experiments (FT1 and FT2)

## Summary
Conducted extensive analysis of fine-tuning experiment evaluations, focusing on comparing single-task (FT1) and two-task (FT2) fine-tuning performance across standard and Atlantis datasets. Created multiple visualization approaches to understand catastrophic forgetting and adaptation to new distributions.

## Key Activities

### 1. Initial Analysis Setup
- Read evaluation data from pt1_ft2-* experiments (two-task fine-tuning)
- Identified training data configuration from `/configs/data_generation/memo.txt`
- Discovered all 7 FT2 experiments had evaluations on both standard and Atlantis tasks

### 2. Atlantis Crossing Analysis
- Created initial plot comparing all FT2 experiments on `atlantis_crossing` task
- Identified that crossing uses binary accuracy metric (not error)
- Found models trained on crossing (FT2-4, FT2-7) performed slightly worse than expected
- Best performance: FT2-1 with 79.33% accuracy

### 3. Comprehensive Evaluation Plots
- Developed `plot_all_ft_evaluations.py` for all 7 evaluation tasks
- Separated standard and Atlantis task visualizations
- Applied log scale to high-variance error metrics
- Key findings:
  - Standard tasks: Models trained on task showed 81-96% improvement
  - Atlantis tasks: Similar dramatic improvements when trained on task
  - Triangle area showed worst performance even when trained

### 4. Normalized Accuracy Analysis
- Created normalized accuracy plots using formula: `(acc - baseline) / (1 - baseline)`
- Initially included clipping to [0,1], then removed to show negative values
- Discovered PT1 baseline was already >97% on standard tasks
- All FT models showed NEGATIVE normalized performance on standard tasks
- Strong positive performance on Atlantis tasks (up to 99% of max improvement)

### 5. Raw Performance Changes
- Modified standard task plots to show raw differences instead of normalized
- Revealed catastrophic degradation:
  - Distance: errors increased 100-500x
  - Triangle area: errors increased up to 682,000 units
  - Even accuracy tasks lost 1-22% accuracy points

### 6. Heatmap Visualizations
- Created 7x7 heatmaps for both FT1 and FT2 experiments
- Used 0=baseline/worse, 1=perfect scaling
- FT1 showed perfect diagonal pattern on Atlantis (single-task specialization)
- FT2 showed block diagonal patterns (two-task combinations)
- Both showed complete failure (0.00) on all standard tasks

### 7. File Organization
- Organized scratch directory into logical subfolders:
  - `analysis/` (atlantis, clusters, city_id)
  - `plots/` (evaluation, metrics, analysis)
  - `testing/` (gradients, circlecount, metrics)
  - `utils/`
- Moved ~40 files from scratch root to appropriate subdirectories

## Technical Insights

### Catastrophic Forgetting
- Fine-tuning on even 1-2 tasks completely destroyed PT1's capabilities
- Performance degradation was extreme (100-500x worse on numerical tasks)
- No recovery even when trained on the same task type

### Atlantis Adaptation Success
- Fine-tuning enabled excellent performance on out-of-distribution data
- Training on specific tasks crucial for best performance
- Cross-task transfer observed but limited

### Metrics Understanding
- Accuracy tasks: crossing, inside, compass (higher=better)
- Error tasks: distance, angle, perimeter, trianglearea (lower=better)
- Normalization revealed PT1 was already near-perfect on standard tasks

## Files Created/Modified

### Scripts Created
- `scratch/plots/evaluation/plot_atlantis_crossing_ft_comparison.py`
- `scratch/plots/evaluation/plot_all_ft_evaluations.py`
- `scratch/plots/evaluation/plot_normalized_accuracy.py`
- `scratch/plots/evaluation/plot_all_normalized_metrics.py`
- `scratch/plots/evaluation/plot_ft_heatmap.py`
- `scratch/plots/evaluation/plot_ft1_heatmap.py`

### Plots Generated
- `atlantis_crossing_ft_comparison.png`
- `ft_comparison_standard_tasks.png`
- `ft_comparison_atlantis_tasks.png`
- `ft_normalized_accuracy.png`
- `ft_all_normalized_standard.png`
- `ft_all_normalized_atlantis.png`
- `ft_performance_heatmap.png`
- `ft1_performance_heatmap.png`

## Key Findings
1. **Complete catastrophic forgetting** occurs even with minimal fine-tuning
2. **Successful adaptation** to Atlantis distribution with 70-99% of max improvement
3. **Training specificity matters** - models excel at trained tasks
4. **PT1 baseline excellence** left no room for improvement on standard tasks
5. **Trade-off is absolute** - gain Atlantis performance, lose standard completely

## Next Steps Implications
- Consider continual learning approaches to mitigate forgetting
- Investigate why triangle area remains problematic across all settings
- Explore whether mixing standard and Atlantis data during fine-tuning helps
- Analyze intermediate checkpoints to understand when forgetting occurs