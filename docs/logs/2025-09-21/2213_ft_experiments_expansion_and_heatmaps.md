# Development Log: Fine-Tuning Experiments Expansion and Heatmap Visualizations
**Date**: 2025-09-21
**Time**: 22:13
**Focus**: Expanding FT2 experiments to 21 variants, creating comprehensive heatmaps, and setting up FTWB2 experiments

## Summary
Extended the fine-tuning experiment infrastructure to support 21 FT2 experiments (up from 7), created comprehensive heatmap visualizations for FT1/FT2/FT3 experiments, and implemented predictive models for multi-task performance based on single-task results.

## Key Activities

### 1. Training Script Infrastructure Expansion
- Created training scripts for `pt1_ft2-9` through `pt1_ft2-21`
- Set up training scripts for `pt1_ftwb2-2` through `pt1_ftwb2-7`
- All scripts follow pattern: `uv run python src/training/train.py configs/training/ftset/[config].yaml --overwrite`

### 2. Evaluation Script Organization
- Filled batch evaluation scripts with absolute paths to avoid path issues
- Created hierarchical batch scripts:
  - `eval_pt1_ft1-b1to4.sh`, `eval_pt1_ft1-b5to7.sh` for FT1
  - `eval_pt1_ft2-b1to4.sh` through `eval_pt1_ft2-b21to21.sh` for FT2
  - `eval_pt1_ft3-b1to4.sh`, `eval_pt1_ft3-b5to7.sh` for FT3
- Fixed path issues by using absolute paths in all batch scripts

### 3. FT2 Task Mapping Discovery
Identified all 21 FT2 experiment configurations:
- FT2-1: distance, trianglearea
- FT2-2: angle, compass
- FT2-3: inside, perimeter
- FT2-4: crossing, distance
- FT2-5: trianglearea, angle
- FT2-6: compass, inside
- FT2-7: perimeter, crossing
- FT2-8: angle, distance
- FT2-9: compass, trianglearea
- FT2-10: angle, inside
- FT2-11: compass, perimeter
- FT2-12: crossing, inside
- FT2-13: distance, perimeter
- FT2-14: crossing, trianglearea
- FT2-15: compass, distance
- FT2-16: inside, trianglearea
- FT2-17: angle, perimeter
- FT2-18: compass, crossing
- FT2-19: distance, inside
- FT2-20: perimeter, trianglearea
- FT2-21: angle, crossing

### 4. Heatmap Visualization Development
Created three comprehensive heatmap scripts:
- **`plot_ft1_heatmap.py`**: 7x7 heatmap for single-task fine-tuning
- **`plot_ft2_heatmap.py`**: 21x7 heatmap for two-task fine-tuning
- **`plot_ft3_heatmap.py`**: 7x7 heatmap for three-task fine-tuning

Key improvements:
- Removed standard task plots (only show Atlantis performance)
- Replaced confusing black boxes with "T" markers in top-right of cells to indicate training tasks
- Added hard-coded task mappings for all experiments
- Implemented error handling (simple fail-fast approach)

### 5. FT2 Predictive Modeling
Implemented prediction model in `plot_ft2_heatmap.py`:
- **Model**: For FT2 trained on tasks i and j, prediction for task k = max(normalized(FT1_i on k), normalized(FT1_j on k))
- **Visualization**: Three-panel plot showing:
  1. Actual FT2 performance (green colormap)
  2. Predicted performance based on FT1 max (green colormap)
  3. Prediction error (predicted - actual) with coolwarm colormap
- **Key fix**: Corrected logic to normalize FT1 results FIRST, then take max (not max then normalize)

### 6. Alternative Predictive Models Discussion
Suggested phenomenological models for multi-task performance prediction:
- Average model: (FT1_i + FT1_j)/2
- Min model (pessimistic): min(FT1_i, FT1_j)
- Geometric mean: sqrt(FT1_i * FT1_j)
- Weighted by task similarity
- Superposition with decay: max(FT1_i, FT1_j) * λ
- Threshold models, complementarity models, harmonic mean
- Linear with interference term

### 7. FTWB2 Experiment Setup
- Created `plot_ftwb2_heatmap.py` based on FT2 script
- Configured for 7 experiments with same task combinations as FT2-1 through FT2-7
- Script ready but correctly fails with FileNotFoundError (experiments not yet evaluated)

## Technical Details

### Shuffling Behavior Clarification
- Training data shuffles ALL ROWS at beginning of each epoch
- Process: Shuffle indices → Create batches → Iterate → Repeat each epoch
- Ensures different batch compositions each epoch

### FT1 Task Mapping Verification
Confirmed correct mapping:
- FT1-1: distance
- FT1-2: trianglearea
- FT1-3: angle
- FT1-4: compass
- FT1-5: inside
- FT1-6: perimeter
- FT1-7: crossing

### File Management
- Removed redundant copy files with spaces in names
- Maintained consistent naming convention across all scripts
- Used absolute paths to avoid execution context issues

## Files Created/Modified

### New Scripts Created
- `/scripts/training/ftset/pt1_ft2-{9..21}.sh`
- `/scripts/training/ftset/pt1_ftwb2-{2..7}.sh`
- `/scripts/eval/ftset/eval_pt1_ft*-b*.sh` (batch scripts)
- `/scratch/plots/evaluation/plot_ft3_heatmap.py`
- `/scratch/plots/evaluation/plot_ftwb2_heatmap.py`

### Modified Scripts
- `/scratch/plots/evaluation/plot_ft1_heatmap.py` (removed standard plots, added T markers)
- `/scratch/plots/evaluation/plot_ft2_heatmap.py` (expanded to 21 experiments, added prediction)

## Next Steps
1. Run evaluations for pt1_ftwb2-{1..7} experiments
2. Test alternative predictive models against actual FT2 results
3. Consider extending analysis to FT3 (3-task) prediction models
4. Investigate task interference patterns from prediction errors

## Notes
- All heatmaps now use consistent 0-1 normalization (0=baseline, 1=perfect)
- "T" markers clearly indicate training tasks without visual clutter
- Prediction error analysis reveals interesting patterns about task interactions
- FTWB2 infrastructure ready for when experiments complete