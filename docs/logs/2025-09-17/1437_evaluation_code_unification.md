# Development Log - 2025-09-17 14:37
## Evaluation Code Unification and Bug Fixes

### Summary
Major refactoring to unify all evaluation logic into a single source of truth, fixing critical bugs in post-training evaluation that were causing incorrect metrics for tasks with parameter equals signs.

### Critical Bug Discovery and Fix

#### The Multiple '=' Problem
**Problem**: Post-training evaluation was failing for tasks like `center`, `circlecount`, `randring` that have '=' signs in their parameters.

**Root Cause**:
- Post-training eval (`evaluate_checkpoints.py`) used naive `text.split('=', 1)`
- This splits at FIRST '=' instead of the LAST '='
- For `circlecount(c_3456,r=150)=23`, it was splitting at `r=` instead of answer separator
- Training eval (`utils.py`) correctly used `rfind('=')` for these tasks

**Impact**: Tasks with parameter '=' signs showed 0% accuracy in post-training evaluation while working fine during training.

#### Different Implementations Problem
**Problem**: Two separate evaluation implementations that diverged:
1. `src/utils.py:evaluate_with_generation` - Used during training
2. `src/eval/evaluate_checkpoints.py` - Used for post-training evaluation

**Key Differences Found**:
- Different prompt/completion splitting logic
- Post-training decoded only new tokens vs full output
- Different space handling in generated text
- Post-training lacked task-specific splitting logic

### Solution: Unified Evaluation Module

#### Created `src/evaluation.py`
Consolidated all evaluation logic into a single module with:
- Unified `evaluate_with_generation()` function
- Task-specific prompt/completion splitting:
  - Tasks with multiple '=': Use `rfind('=')` (center, circlecount, randring)
  - Other tasks: Use `find('=')`
- Proper tokenization with LEFT padding for generation
- Decoding FULL outputs (not just new tokens)
- Optional `return_details` flag:
  - `False` for training (memory efficient, only aggregated metrics)
  - `True` for post-training (saves every example for debugging)

#### Updated All Callers
- `src/training/train.py`: Uses unified evaluation with `return_details=False`
- `src/utils.py:GenerationEvalCallback`: Uses unified evaluation
- `src/eval/evaluate_checkpoints.py`: Uses unified evaluation with `return_details=True`

### Output Directory Organization

Restructured post-training evaluation output for clarity:
```
output_dir/
├── eval_config.yaml    # Config at root for easy access
├── dynamics/          # Training dynamics plots (renamed from summary/)
│   ├── loss.png
│   ├── distance.png
│   └── ...
└── eval_data/         # Evaluation data
    ├── evaluation_results.json
    └── detailed_results_checkpoint-*.jsonl
```

### Configuration Updates

Added `cities_csv` path to evaluation configs for tasks that need coordinate validation:
```yaml
cities_csv: "/data/datasets/cities/cities.csv"
```

### Key Insights

1. **Code Duplication is Dangerous**: The evaluation implementations diverged because they were maintained separately
2. **Task-Specific Logic is Critical**: Different tasks have different grammar patterns that must be handled correctly
3. **Loss Plot Mystery Solved**: Loss plots only appear when evaluating "all" checkpoints (reads from checkpoint-0/eval_results.json), not when evaluating specific checkpoints

### Files Modified

- Created: `src/evaluation.py` (single source of truth)
- Modified: `src/eval/evaluate_checkpoints.py` (use unified evaluation)
- Modified: `src/training/train.py` (use unified evaluation)
- Modified: `src/utils.py` (callback uses unified evaluation)
- Modified: `configs/eval/m1_10M/multi_task.yaml` (added cities_csv)
- Modified: `configs/eval/m1_10M_ft1/multi_task.yaml` (added cities_csv)

### Impact

- All 12 task types now evaluate correctly in post-training evaluation
- Single source of truth prevents future divergence
- Cleaner output directory structure
- Memory-efficient training evaluation with optional detailed results for analysis

### Testing Notes

The fix was validated by understanding that:
- Working tasks had no '=' in parameters: `dist()`, `compass()`, `crossing()`, `inside()`
- Failing tasks had '=' in parameters: `center(;in=)`, `circlecount(r=)`, `randring(r=,R=,n=)`

This systematic pattern confirmed the root cause and validated the fix.