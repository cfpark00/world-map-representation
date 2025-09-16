# Development Log - 2025-09-16 10:40
## Multitask Evaluation Fixes and Circlecount Parsing Bug

### Summary
Fixed critical issues with multitask model evaluation, including handling all 12 task types and discovering/fixing a major parsing bug in the circlecount metric calculation.

### Key Accomplishments

#### 1. Fixed Multitask Evaluation for All 12 Task Types
**Problem**: The evaluate_with_generation function was throwing errors for unknown task types when evaluating the m1_10M multitask model.

**Tasks Added Support For**:
- distance ✓ (already existed)
- randomwalk ✓ (already existed)
- trianglearea ✓ (already existed)
- angle ✓ (already existed)
- compass ✓ (new)
- inside ✓ (new)
- nearest_neighbor ✓ (new)
- perimeter ✓ (new)
- crossing ✓ (new)
- circlecount ✓ (new)
- center ✓ (new)
- randring ✓ (new)

**Special Handling**:
- Tasks with multiple `=` signs (center, circlecount, randring) now split at the LAST `=` to correctly separate parameters from results
- Added appropriate max_new_tokens for each task type based on expected output length
- Implemented proper metrics for each task type

#### 2. Discovered and Fixed Critical Circlecount Bug
**Bug Discovery**: The circlecount task was showing perfect 0.0 error, which seemed suspicious.

**Root Cause**:
- Circlecount format: `circlecount(c_123,r=456)=789` has TWO equal signs
- The regex `r'=(\d+)'` was matching the FIRST `=` (radius parameter `r=456`) instead of the actual count
- This meant we were comparing the radius value with itself, always getting 0 error!

**Fix Applied** (`src/utils.py` lines 1008-1036):
- Extract true count directly from completion (just the number)
- For generated output, use `re.findall(r'=(\d+)')` and take the LAST match
- This correctly extracts the count value, not the radius

**Impact**: Real circlecount errors are likely in the range of 2-60, not the perfect 0 that was being reported.

#### 3. Created Comprehensive Plotting Scripts
**plot_m1_10M_metrics.py** (in scratch/):
- Reads trainer_state.json from checkpoints to extract all evaluation history
- Handles mismatched metric counts across tasks
- Creates individual plots for each task and a summary plot
- Properly handles log scales and reference lines

**Key Features**:
- Reads from checkpoint-0/eval_results.json for initial metrics
- Extracts full training history from trainer_state.json
- Handles padding/alignment when tasks have different evaluation counts
- Creates both individual task plots and an all-tasks summary

#### 4. Surgical Plot Replacement for Circlecount
**reevaluate_circlecount.py** (in scratch/):
- Loads model checkpoints and re-evaluates with fixed parsing
- Evaluates ALL 41 checkpoints (not just a sample)
- Generates corrected circlecount.png to replace the buggy one
- Saves evaluation data to JSON for future reference

### Metrics Implementation Details

| Task Type | Metric | Range | Better Direction |
|-----------|--------|-------|------------------|
| distance | Absolute error | 0-4025 | Lower |
| randomwalk | Validity score | 0-1 | Higher |
| trianglearea | Absolute error | 0-3.24M | Lower |
| angle | Absolute error (degrees) | 0-180 | Lower |
| compass | Exact match accuracy | 0-1 | Higher |
| inside | Binary accuracy | 0-1 | Higher |
| nearest_neighbor | Jaccard similarity | 0-1 | Higher |
| perimeter | Absolute error | 0-20000 | Lower |
| crossing | Binary accuracy | 0-1 | Higher |
| circlecount | Absolute error | 0-1000 | Lower |
| center | Exact match accuracy | 0-1 | Higher |
| randring | Validity ratio | 0-1 | Higher |

### Files Modified
- `/src/utils.py`: Fixed evaluation for all 12 tasks, fixed circlecount parsing bug
- Created multiple scripts in `/scratch/`:
  - `plot_m1_10M_metrics.py`: General metrics plotting
  - `test_circlecount_generation.py`: Testing circlecount parsing
  - `surgery_circlecount_plot.py`: Initial attempt at plot replacement
  - `reevaluate_circlecount.py`: Full re-evaluation with fixed parsing

### Technical Notes
1. The `nearest_neighbor` task type is used in datasets but was being checked as just `nearest` - fixed to handle both
2. Checkpoint evaluations store metrics in trainer_state.json's log_history, not individual eval_results.json files
3. The circlecount bug affected all historical evaluations - would need re-evaluation to get true metrics
4. Some tasks weren't evaluated at every step, requiring padding/alignment for plotting

### Next Steps
- The reevaluate_circlecount.py script needs to be run to generate corrected metrics
- Consider re-evaluating other tasks to ensure no similar parsing bugs exist
- The fixed parsing will only affect future training runs unless historical checkpoints are re-evaluated

### Lessons Learned
1. Always verify suspiciously perfect metrics - 0.0 error should raise red flags
2. When parsing structured output with multiple separators, be explicit about which one to use
3. Test parsing logic on actual examples before trusting aggregate metrics
4. Complex formats (like those with parameters) need careful regex design