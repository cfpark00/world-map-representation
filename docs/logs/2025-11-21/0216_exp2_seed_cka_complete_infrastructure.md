# Exp2 Seed CKA Complete Infrastructure Creation

**Date:** 2025-11-21
**Time:** 02:16 AM EST
**Session Duration:** ~2 hours

## Summary

Created complete CKA analysis infrastructure for PT2/PT3 seed robustness (Exp2), with emphasis on eliminating redundant calculations and optimizing for parallel execution.

## Context

User requested CKA calculation scripts for PT2/PT3 seed variants, following the pattern established in Exp4. Key requirement: **only non-overlapping task pairs** (where models share NO training tasks).

## Tasks Completed

### 1. Initial Infrastructure Creation

**Created config generator:**
- `src/scripts/generate_exp2_seed_cka_configs.py`
- Automatically identifies non-overlapping PT2/PT3 pairs
- Generates configs for cross-seed comparisons
- Initial version: 504 configs (336 PT2 + 168 PT3)

**Non-overlapping pairs identified:**
- PT2: 14 pairs (out of 21 possible)
- PT3: 7 pairs (out of 21 possible)

### 2. Critical Bug Fix: Redundant Calculations

**Problem discovered by user:**
- Initial version computed both `orig vs 1` AND `1 vs orig`
- CKA is symmetric, so these are redundant!
- 504 configs → wasting 50% compute time

**Root cause:**
```python
# WRONG - bidirectional
for seed1 in seeds:
    for seed2 in seeds:
        if seed1 == seed2:
            continue
        # Creates 6 pairs: orig-1, orig-2, 1-orig, 1-2, 2-orig, 2-1
```

**Fix applied:**
```python
# CORRECT - unique pairs only
for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:
        # Creates 3 pairs: orig-1, orig-2, 1-2
```

**Result:** 504 configs → **252 configs (50% reduction)**

### 3. Execution Script Generator

**Created:**
- `src/scripts/generate_exp2_seed_cka_run_scripts.py`
- Generates layer-specific scripts (PT2/PT3 × layers 3,4,5,6)
- Master scripts for running all layers
- Combined master for PT2+PT3

**Scripts generated (11 total):**
- `run_pt2_seed_cka_l{3,4,5,6}.sh` (4 scripts)
- `run_pt3_seed_cka_l{3,4,5,6}.sh` (4 scripts)
- `run_pt2_seed_cka_all_layers.sh` (master)
- `run_pt3_seed_cka_all_layers.sh` (master)
- `run_exp2_seed_cka_all.sh` (combined master)

### 4. Parallel Execution: 4-Chunk Division

**User request:** Divide labor into 4 scripts for parallel execution

**Created:**
- `src/scripts/generate_exp2_seed_cka_chunked_scripts.py`
- Perfectly balanced 4-way split: 63 calcs per chunk
- Mixes PT2/PT3 and all layers for optimal distribution

**Scripts generated:**
- `run_exp2_seed_cka_chunk{1,2,3,4}.sh` (63 each)
- `run_exp2_seed_cka_4chunks_parallel.sh` (master)

### 5. Checkpoint Issue Resolution

**Problem encountered:**
- Initial configs specified `checkpoint_steps: [328146]`
- This is PT1-X's final checkpoint (42 epochs)
- PT2/PT3 only have 21 epochs, so checkpoint doesn't exist!

**User clarification:** "just take last for each"

**Solution implemented:**
```yaml
checkpoint_steps: null  # auto-detect
use_final_only: true   # use max checkpoint
```

**Updated `analyze_cka_pair.py`:**
- Added `use_final_only` flag
- Automatically selects `max(common_steps)` if enabled
- Works for any experiment (PT1, PT2, PT3, etc.)

### 6. Overwrite Flag Fix

**Problem:** `--overwrite` not being passed to Python scripts

**Fix:** Updated chunk generator to include `--overwrite` in all commands:
```bash
uv run python src/scripts/analyze_cka_pair.py {config_path} --overwrite
```

## Final Infrastructure

### Configs: 252 total
- **PT2:** 168 configs (14 pairs × 4 layers × 3 seed combos)
- **PT3:** 84 configs (7 pairs × 4 layers × 3 seed combos)
- **Seed combinations:** orig-vs-1, orig-vs-2, 1-vs-2 (unique only)
- **Location:** `configs/revision/exp2/{pt2,pt3}_seed_cka/`

### Execution Scripts: 15 total
- Layer-specific: 8 scripts
- Masters: 3 scripts
- Chunks: 4 scripts (63 calcs each)
- Location: `scripts/revision/exp2/cka_analysis/`

### Generator Scripts: 3 total
- `src/scripts/generate_exp2_seed_cka_configs.py`
- `src/scripts/generate_exp2_seed_cka_run_scripts.py`
- `src/scripts/generate_exp2_seed_cka_chunked_scripts.py`

## Technical Details

### PT2 Non-overlapping Pairs (14)
```
(1,2), (1,3), (1,6), (1,7)    # pt2-1 vs others
(2,3), (2,4), (2,7)            # pt2-2 vs others
(3,4), (3,5)                   # pt2-3 vs others
(4,5), (4,6)                   # pt2-4 vs others
(5,6), (5,7)                   # pt2-5 vs others
(6,7)                          # pt2-6 vs pt2-7
```

### PT3 Non-overlapping Pairs (7)
```
(1,2), (1,7)    # pt3-1 vs others
(2,3)           # pt3-2 vs pt3-3
(3,4)           # pt3-3 vs pt3-4
(4,5)           # pt3-4 vs pt3-5
(5,6)           # pt3-5 vs pt3-6
(6,7)           # pt3-6 vs pt3-7
```

### Config Structure
```yaml
exp1:
  name: pt2-1
  repr_dir: .../pt2-1/analysis_higher/distance_firstcity_last_and_trans_l5/representations
  task: distance
exp2:
  name: pt2-2_seed1
  repr_dir: .../revision/exp2/pt2-2_seed1/analysis_higher/angle_firstcity_last_and_trans_l5/representations
  task: angle
layer: 5
checkpoint_steps: null
use_final_only: true
city_filter: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$
kernel_type: linear
center_kernels: true
use_gpu: true
save_timeline_plot: false
output_dir: .../revision/exp2/cka_analysis/pt2-1_vs_pt2-2_seed1/layer5
```

## Usage

### Parallel Execution (Recommended)
```bash
# Terminal 1
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk1.sh

# Terminal 2
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk2.sh

# Terminal 3
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk3.sh

# Terminal 4
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk4.sh
```

### Sequential Execution
```bash
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_all.sh
```

### By Layer or Experiment Type
```bash
bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_l5.sh      # PT2 layer 5
bash scripts/revision/exp2/cka_analysis/run_pt3_seed_cka_all_layers.sh  # All PT3
```

## Output Structure

Results saved to: `data/experiments/revision/exp2/cka_analysis/`

```
cka_analysis/
├── pt2-1_vs_pt2-2_seed1/
│   └── layer5/
│       ├── cka_timeline.csv
│       ├── summary.json
│       └── config.yaml
└── ...
```

## Files Created/Modified

### Created:
- `src/scripts/generate_exp2_seed_cka_configs.py`
- `src/scripts/generate_exp2_seed_cka_run_scripts.py`
- `src/scripts/generate_exp2_seed_cka_chunked_scripts.py`
- `configs/revision/exp2/pt2_seed_cka/` (168 configs)
- `configs/revision/exp2/pt3_seed_cka/` (84 configs)
- `scripts/revision/exp2/cka_analysis/` (15 scripts)
- `docs/logs/2025-11-21/exp2_seed_cka_infrastructure.md`
- `docs/logs/2025-11-21/exp2_seed_cka_infrastructure_FINAL.md`
- `docs/logs/2025-11-21/0216_exp2_seed_cka_complete_infrastructure.md` (this file)

### Modified:
- `src/scripts/analyze_cka_pair.py`
  - Added `use_final_only` parameter
  - Auto-selects final checkpoint when enabled
  - Better error message when no checkpoints after filtering

## Key Decisions

1. ✅ **Only non-overlapping pairs** - Meaningful comparisons where models share no training tasks
2. ✅ **Unique seed pairs only** - Eliminated 50% redundancy (CKA is symmetric)
3. ✅ **Auto-detect final checkpoint** - Works across PT1/PT2/PT3 without hardcoding
4. ✅ **4-way parallel split** - Perfectly balanced chunks for efficient execution
5. ✅ **Automatic overwrite** - No manual cleanup needed between runs

## Optimization Impact

**Before optimization:**
- 504 configs (including redundant symmetric pairs)
- Hardcoded checkpoint that didn't exist for PT2/PT3
- Manual overwrite required

**After optimization:**
- 252 configs (50% reduction, optimal)
- Auto-detects correct final checkpoint per experiment
- Automatic overwrite enabled
- **Savings:** 50% compute time, 50% storage

## Next Steps

1. Run the 4 chunks in parallel (252 calculations total)
2. Create visualization scripts for seed robustness analysis
3. Generate CKA trends plot (PT1-X vs PT2 vs PT3)
4. Statistical analysis comparing intra-task vs inter-task CKA
5. Integration with broader 1→2→3 task regime analysis

## Research Impact

This infrastructure enables:
- **Seed robustness matrices** for PT2/PT3 (similar to Exp4's 21×21 for PT1-X)
- **Cross-regime CKA trends** showing how representations change from 1→2→3 tasks
- **Statistical validation** of representation stability across seeds
- **Task diversity effects** on representation similarity

All infrastructure is optimized, validated, and ready for execution.
