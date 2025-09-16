# Random Walk Evaluation Fixes and Old Format Cleanup

## Date: 2025-09-15 14:01

## Summary
Fixed critical issues with random walk evaluation, cleaned up old format references throughout the codebase, and created a standalone checkpoint evaluation script for random walk performance analysis.

## Major Accomplishments

### 1. Random Walk Evaluation Scoring Update
- **Changed scoring from binary to continuous**: Updated chain length evaluation from binary (1.0 if exact, 0.0 otherwise) to use exponential decay: `exp(-|actual_len - expected_len| / expected_len)`
- **Combined score now**: `validity_ratio × chain_length_ratio`
- More nuanced scoring that gives partial credit for near-misses in chain length
- Example: Being off by 1 city in a 10-city walk now scores ~0.90 instead of 0.0

### 2. Created Standalone RW Checkpoint Evaluation Script
- **Script**: `/n/home12/cfpark00/WM_1/scratch/eval_rw_checkpoints.py`
- **Features**:
  - Auto-detects all checkpoints in an experiment directory
  - Evaluates random walk performance using the same `evaluate_with_generation()` function as training
  - Filters validation set to find ALL random walk samples (no sampling)
  - Optional CLI parameter to limit samples for quick testing
  - Custom plotting with mean ± std error bars and median
  - Saves results to `scratch/tempplots/`

### 3. Critical Bug Fixes in Evaluation

#### Fixed Prompt/Completion Splitting
- **Problem**: Random walk was looking for `'WALK:'` delimiter (old format) that doesn't exist
- **Solution**: Now correctly splits at `'=`' like other tasks
- **Added fail-fast**: Removed ALL silent fallbacks - now crashes immediately if delimiter not found

#### Fixed RandomWalkCollator for Training
- **Problem**: Still looking for `'WALK:'` format in loss masking
- **Solution**: Updated to look for `'='` and mask everything before it
- Now correctly computes loss only on city sequence

#### Fixed max_new_tokens
- Changed from 200 to 224 tokens for random walk generation (accommodates longer sequences with padding)

### 4. Old Format Cleanup

#### Found and Fixed:
1. **RandomWalkCollator**: Was looking for `WALK:` format, now uses `rw(max,len)=` format
2. **parse_walk_transitions comment**: Updated from `walk_200=` to `rw(200,5)=` format
3. **Evaluation splitting logic**: Now correctly handles all current formats with no fallbacks

#### Identified but Not Fixed (will be deleted):
- `analyze_representations_higher.py`: Still uses old `walk_200=` format (marked for deletion)

### 5. Improved Random Walk Evaluation

#### Made Evaluation More Realistic
- **Now provides first city in prompt** during evaluation
- Prompt: `rw(66,13)=c_9361,` (includes first city)
- Model must continue from given starting point, can't choose easy starting city
- Makes task harder and more realistic

### 6. Plot Corrections
- Fixed random walk plot labels in `save_training_plots()`
- Changed from "Error (lower is better)" to "Combined Score (higher is better)"
- Updated reference lines to match new scoring (0.9 = excellent, 0.0 = failure)

## Key Technical Changes

### Files Modified
- `src/utils.py`:
  - Fixed evaluation prompt/completion splitting
  - Updated RandomWalkCollator
  - Fixed plot labels
  - Removed all silent fallbacks
- `scratch/eval_rw_checkpoints.py`: New standalone evaluation script

### Important Constants
- Max tokens for random walk: 224
- Temperature for evaluation: 0 (greedy decoding)
- Random walk format: `rw(max_distance,chain_length)=city_list`

## Issues Resolved
- Random walk evaluation was completely broken due to wrong format expectations
- Silent failures were hiding critical bugs
- Scoring was too harsh (binary chain length check)
- Model could choose easy starting cities

## Next Steps Recommended
- Delete `analyze_representations_higher.py` as planned
- Run full evaluation on existing models with fixed code
- Consider similar first-token prompting for other tasks if appropriate