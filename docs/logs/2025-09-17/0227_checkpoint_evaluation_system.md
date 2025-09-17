# Checkpoint Evaluation System Implementation and Fixes

**Date**: 2025-09-17
**Time**: 02:27
**Focus**: Creating standalone checkpoint evaluation system and fixing evaluation discrepancies

## Summary
Implemented a comprehensive checkpoint evaluation system that properly evaluates model checkpoints on test/validation datasets, with correct metric calculation, plotting, and detailed results export. Fixed multiple issues with task naming, dataset loading, and metric consistency.

## Major Accomplishments

### 1. Created Evaluation System Structure
- **Config**: `configs/eval/distance_1M_distance_eval.yaml` - evaluation configuration
- **Script**: `src/eval/evaluate_checkpoints.py` - main evaluation logic
- **Runner**: `scripts/eval/run_distance_1M_eval.sh` - bash execution script
- Follows same pattern as analysis_representations for consistency

### 2. Fixed Critical Issues

#### Task Naming Cleanup
- Removed non-existent "bearing" and "loc" tasks that were hallucinated
- Fixed "nearest" vs "nearest_neighbor" confusion - task name is `nearest_neighbor`, grammar uses `nearest(`
- Removed duplicate "nearest" entry from metrics registry
- Updated tasks.json to have only correct task definitions

#### Dataset Loading
- Fixed to use proper HuggingFace tokenized datasets (not cities CSV)
- Added support for selecting dataset split (test/validation/train)
- Properly handles the dataset structure with train/validation/test splits

#### Metric Consistency
- Reuses exact same metric implementations from `src/metrics.py`
- Uses same metric format as training (`eval_{task}_metric_mean`, etc.)
- Properly converts numpy types for JSON serialization

### 3. Advanced Features

#### Full Results Export
- Added `save_full_results: true` option in config
- Saves detailed JSONL files with every evaluation row:
  - text, prompt, expected, generation, metric, task_type
- Enables deep analysis of model behavior

#### Proper Padding
- Fixed LEFT padding for generation (critical for autoregressive models)
- RIGHT padding maintained for training
- Ensures pad_token is properly set

#### Unified Plotting
- Reuses `save_training_plots` function from training
- Creates task-specific plots with proper scales and reference lines
- Consistent visualization between training and evaluation

### 4. Dataset Split Investigation
- Discovered validation set has only 128 samples vs 10,000 test samples
- Training evaluates on 64 randomly sampled from the 128 validation examples
- Identified this as potential source of evaluation discrepancies
- The model is NOT overfitting (no gradients on eval data)

## Files Created/Modified

### New Files
- `/configs/eval/distance_1M_distance_eval.yaml` - main eval config
- `/configs/eval/distance_1M_distance_eval_val.yaml` - validation eval config
- `/src/eval/evaluate_checkpoints.py` - evaluation implementation
- `/scripts/eval/run_distance_1M_eval.sh` - runner script
- `/scripts/eval/run_distance_1M_eval_val.sh` - validation runner
- `/scripts/analysis_representations/run_all_seeds.sh` - batch analysis script

### Modified Files
- `/src/metrics.py` - removed duplicate "nearest" entry
- `/src/utils.py` - fixed task name checks for nearest_neighbor
- `/src/utils_backup.py` - same fixes
- `/configs/tasks.json` - removed incorrect "nearest" task

## Key Technical Details

### Evaluation Flow
1. Load tokenized dataset from HuggingFace format
2. Select specified split (test/validation/train)
3. For each checkpoint:
   - Load model with proper dtype and device mapping
   - Generate predictions with LEFT padding
   - Calculate metrics using centralized metric system
   - Store detailed results if requested
4. Generate plots using training plotting function
5. Save aggregated and detailed results

### Metric Format
- `eval_{task}_metric_mean`: Average metric value
- `eval_{task}_metric_median`: Median value
- `eval_{task}_metric_std`: Standard deviation
- `eval_{task}_metric_min/max`: Range
- `eval_{task}_valid_count`: Number of valid generations
- `eval_{task}_valid_ratio`: Proportion of valid outputs

## Issues Investigated

### Evaluation Discrepancy
- Training shows improving metrics on validation set
- Standalone evaluation shows worse performance
- Root cause still under investigation, but confirmed:
  - Correct dataset splits are being used
  - No overfitting possible (no gradients on eval)
  - Difference may be due to sampling (64 of 128 during training vs all)

## Next Steps
- Further investigate why validation metrics differ
- Consider increasing validation set size in future experiments
- Add more task implementations to evaluation script
- Create batch evaluation scripts for multiple experiments

## Commands for Usage
```bash
# Evaluate on test set
bash scripts/eval/run_distance_1M_eval.sh --overwrite

# Evaluate on validation set
bash scripts/eval/run_distance_1M_eval_val.sh --overwrite

# Run analysis on all seed experiments
bash scripts/analysis_representations/run_all_seeds.sh
```

## Lessons Learned
1. Always verify task definitions against source of truth (tasks.json)
2. Padding direction is critical for generation vs training
3. Small validation sets can lead to unreliable metrics
4. Reusing existing code (metrics, plotting) ensures consistency
5. Dataset structure must be thoroughly understood before evaluation