# ResearchPy Philosophy and Major Code Cleanup
Date: 2025-08-31
Time: 00:18

## Summary
Major refactoring session applying the "ResearchPy Philosophy" (formerly monopy model) to clean up the WM_1 codebase. Unified training scripts, removed redundant code, and established clear separation between implementation (utils.py) and orchestration (scripts).

## Key Accomplishments

### 1. Fixed JSON Serialization Error
- **Problem**: `TypeError: Object of type int64 is not JSON serializable` during training
- **Solution**: Added `convert_numpy_to_python()` function to convert numpy types before JSON serialization
- Applied to both callback metrics and final results saving

### 2. Unified Training Scripts
- **Merged** `train_location.py` and `train_distance.py` into single `train.py`
- **Removed** 55+ lines of config validation, replaced with single `preprocess_config()` call
- Scripts were 99% identical, differing only in task-specific parsing/metrics
- Now uses `task_type` field in config to handle differences

### 3. Applied ResearchPy Philosophy
- **Core principle**: Implementation (HOW) goes to utils.py, Orchestration (WHAT/WHEN/WHETHER) goes to scripts
- **Fail-fast approach**: Removed all fallbacks in favor of explicit configuration
- **Created reusable functions**:
  - `preprocess_config()` - Validates and converts all config fields
  - `get_model()` - Initializes model from config
  - `get_dataset()` - Loads and prepares datasets
  - `init_experiment_directory()` - Manages exp directory with safety checks
  - `GenerationEvalCallback` - Moved to utils.py

### 4. Removed Unnecessary Code
- **Deleted** 26 lines of custom optimizer/scheduler setup
  - HuggingFace Trainer already handles this with `lr_scheduler_type="linear"`
- **Removed** explicit `data_collator` specification (framework handles defaults)
- **Cleaned up** unused imports throughout
- **Moved** all module-level code into `main()` function

### 5. Fixed Config Issues
- Added required `task_type` field to all configs
- Fixed `loss_mask_type` from string "null" to proper "full"
- Verified linear_with_warmup scheduler properly maps to HuggingFace's "linear"

### 6. Organized Task-Specific Code
- Moved `parse_distance()` next to `parse_location()` in utils.py
- Grouped related parsing functions together
- Task-specific logic uses clean if/else branches on `task_type`

### 7. Documentation Updates
- Renamed philosophy from "Monopy Model" to "ResearchPy Philosophy"
- Restructured document to be example-driven using actual code from refactoring
- Added lessons learned from today's session:
  - Don't over-engineer the Trainer
  - Group related functions
  - Require everything in config (no fallbacks)
  - Main function contains ALL setup
  - Trust framework defaults

## Files Modified

### Core Training Scripts
- `src/training/train.py` - New unified training script (replaced location/distance scripts)
- `src/training/train_location.py` - Archived
- `src/training/train_distance.py` - Archived
- `src/training/archive/train_old.py` - Moved old version to archive

### Utils and Helpers
- `src/utils.py` - Major additions:
  - `preprocess_config()` - Config validation
  - `get_model()` - Model initialization
  - `get_dataset()` - Dataset loading
  - `init_experiment_directory()` - Directory management
  - `convert_numpy_to_python()` - NumPy type conversion
  - `GenerationEvalCallback` - Moved from scripts
  - Reorganized parsing functions

### Configurations
- `configs/dist_100k_1M_10epochs.yaml` - Added task_type, fixed loss_mask_type
- `configs/loc_100k_200epochs_ar.yaml` - Added task_type, fixed loss_mask_type

### Documentation
- `claude_notes/tips/researchpy_philosophy.md` - New philosophy document (renamed from monopy)
- Restructured to show real examples from today's refactoring

## Cleanup Statistics
- **Lines removed**: ~200+ lines of redundant/unnecessary code
- **Files consolidated**: 3 training scripts → 1
- **Functions extracted to utils**: 6 major functions
- **Imports cleaned**: Removed 10+ unused imports

## Key Insights

### The ResearchPy Philosophy Works
- Clear separation of implementation and orchestration makes code much more readable
- Scripts now read like a story - each line is a meaningful step
- Utils.py contains chunky, complete operations (not over-modularized)

### Fail-Fast is Better
- Removing fallbacks caught several config issues immediately
- Explicit requirements make debugging much easier
- No hidden defaults that can cause silent failures

### Trust the Framework
- HuggingFace Trainer handles many things automatically (optimizer, scheduler, data collator)
- Don't reimplement what the framework provides
- Saved 26+ lines just by using built-in scheduler support

## Next Steps Recommended
1. Test the unified training script with both location and distance tasks
2. Verify all configs have required fields
3. Consider further utils.py additions for other repeated patterns
4. Update any remaining scripts to follow ResearchPy philosophy

## Session Duration
~4 hours of intensive refactoring and code cleanup