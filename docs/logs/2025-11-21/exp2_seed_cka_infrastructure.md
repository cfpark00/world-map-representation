# Exp2 Seed CKA Infrastructure Creation

**Date:** 2025-11-21
**Task:** Create CKA analysis infrastructure for PT2/PT3 seed robustness (exp2)

## Context

User asked to create CKA calculation infrastructure for PT2 and PT3 seed variants, focusing on **non-overlapping task pairs only** (meaningful comparisons where models don't share any training tasks).

## Previous State

- ✅ PT2/PT3 original models (seed 42): CKA already calculated for ALL pairs
  - PT2: 28 pairs × 4 layers = 112 results
  - PT3: 28 pairs × 4 layers = 112 results
  - Location: `data/experiments/cka_analysis_pt{2,3}/`

- ✅ PT2/PT3 seed variants: Representations extracted for all 4 layers
  - PT2: 14 models (pt2-{1-7}_seed{1,2})
  - PT3: 14 models (pt3-{1-7}_seed{1,2})
  - Note: pt2-8 and pt3-8 seeds not trained

- ❌ PT2/PT3 seed variants: NO CKA analyses existed

## What Was Created

### 1. Config Generator Script

**File:** `src/scripts/generate_exp2_seed_cka_configs.py`

**Features:**
- Automatically identifies non-overlapping task pairs
- Generates configs for cross-seed comparisons only (not same-seed)
- Covers all 4 layers (3, 4, 5, 6)
- Uses final checkpoint only (328146)

**Task Definitions:**
```python
PT2_TASKS = {
    'pt2-1': {'distance', 'trianglearea'},
    'pt2-2': {'angle', 'compass'},
    'pt2-3': {'inside', 'perimeter'},
    'pt2-4': {'crossing', 'distance'},
    'pt2-5': {'trianglearea', 'angle'},
    'pt2-6': {'compass', 'inside'},
    'pt2-7': {'perimeter', 'crossing'},
}

PT3_TASKS = {
    'pt3-1': {'distance', 'trianglearea', 'angle'},
    'pt3-2': {'compass', 'inside', 'perimeter'},
    'pt3-3': {'crossing', 'distance', 'trianglearea'},
    'pt3-4': {'angle', 'compass', 'inside'},
    'pt3-5': {'perimeter', 'crossing', 'distance'},
    'pt3-6': {'trianglearea', 'angle', 'compass'},
    'pt3-7': {'inside', 'perimeter', 'crossing'},
}
```

**Non-overlapping Pairs Identified:**
- **PT2: 14 pairs** (out of 21 possible)
  - (1,2), (1,3), (1,6), (1,7), (2,3), (2,4), (2,7), (3,4), (3,5), (4,5), (4,6), (5,6), (5,7), (6,7)

- **PT3: 7 pairs** (out of 21 possible)
  - (1,2), (1,7), (2,3), (3,4), (4,5), (5,6), (6,7)

**Seed Combinations:**
- 3 seeds: orig (42), seed1, seed2
- Cross-seed comparisons: orig-vs-1, orig-vs-2, 1-vs-2, and reverse
- Total: 6 comparisons per pair per layer

### 2. Configs Generated

**Total: 504 configs**

**PT2: 336 configs**
- 14 pairs × 4 layers × 6 seed combinations = 336
- Location: `configs/revision/exp2/pt2_seed_cka/`
- Structure: `pt2_seed_cka/pt2-{i}_vs_pt2-{j}/layer{N}_{seed1}_vs_{seed2}.yaml`

**PT3: 168 configs**
- 7 pairs × 4 layers × 6 seed combinations = 168
- Location: `configs/revision/exp2/pt3_seed_cka/`
- Structure: `pt3_seed_cka/pt3-{i}_vs_pt3-{j}/layer{N}_{seed1}_vs_{seed2}.yaml`

**Example Config:**
```yaml
exp1:
  name: pt2-1
  repr_dir: .../data/experiments/pt2-1/analysis_higher/distance_firstcity_last_and_trans_l5/representations
  task: distance
exp2:
  name: pt2-2_seed1
  repr_dir: .../data/experiments/revision/exp2/pt2-2_seed1/analysis_higher/angle_firstcity_last_and_trans_l5/representations
  task: angle
layer: 5
checkpoint_steps: [328146]
city_filter: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$
kernel_type: linear
center_kernels: true
use_gpu: true
save_timeline_plot: false
output_dir: .../data/experiments/revision/exp2/cka_analysis/pt2-1_vs_pt2-2_seed1/layer5
```

### 3. Execution Script Generator

**File:** `src/scripts/generate_exp2_seed_cka_run_scripts.py`

**Features:**
- Layer-specific scripts for parallel execution
- Master scripts for running all layers
- Combined master for PT2+PT3
- Progress tracking (shows count/total)

### 4. Execution Scripts Generated

**Total: 11 scripts**

**PT2: 5 scripts**
- `run_pt2_seed_cka_l3.sh` (84 calculations)
- `run_pt2_seed_cka_l4.sh` (84 calculations)
- `run_pt2_seed_cka_l5.sh` (84 calculations)
- `run_pt2_seed_cka_l6.sh` (84 calculations)
- `run_pt2_seed_cka_all_layers.sh` (master)

**PT3: 5 scripts**
- `run_pt3_seed_cka_l3.sh` (42 calculations)
- `run_pt3_seed_cka_l4.sh` (42 calculations)
- `run_pt3_seed_cka_l5.sh` (42 calculations)
- `run_pt3_seed_cka_l6.sh` (42 calculations)
- `run_pt3_seed_cka_all_layers.sh` (master)

**Combined: 1 script**
- `run_exp2_seed_cka_all.sh` (runs PT2 + PT3)

**Location:** `scripts/revision/exp2/cka_analysis/`

## Usage

### Run Everything
```bash
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_all.sh
```

### Run PT2 or PT3 Only
```bash
bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_all_layers.sh
bash scripts/revision/exp2/cka_analysis/run_pt3_seed_cka_all_layers.sh
```

### Run Single Layer
```bash
bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_l5.sh
bash scripts/revision/exp2/cka_analysis/run_pt3_seed_cka_l6.sh
```

## Output Locations

All CKA results will be saved to:
```
data/experiments/revision/exp2/cka_analysis/
├── pt2-1_vs_pt2-2_seed1/layer5/
│   ├── cka_results.json
│   ├── cka_values.csv
│   └── config.yaml
├── pt2-1_seed1_vs_pt2-2/layer5/
│   └── ...
└── ...
```

## Analysis Plan

Once CKA calculations are complete, these can be used for:

1. **Seed robustness analysis** (like Exp4's 21×21 matrix)
   - Intra-variant cross-seed CKA (same variant, different seeds)
   - Inter-variant cross-seed CKA (different variants, different seeds)

2. **CKA trends analysis** across 1-2-3 task regimes
   - PT1-X (single task) + Exp4 seeds
   - PT2 (two tasks) + Exp2 seeds
   - PT3 (three tasks) + Exp2 seeds
   - Plot CKA vs task diversity with seed error bars

3. **Statistical testing**
   - Does CKA increase/decrease with more training tasks?
   - Is seed variability stable across PT2/PT3?

## Statistics

**Total Infrastructure:**
- 504 config files
- 11 execution scripts
- 504 CKA calculations to be run
  - PT2: 336 (14 pairs × 4 layers × 6 seed combos)
  - PT3: 168 (7 pairs × 4 layers × 6 seed combos)

**Key Decisions:**
- ✅ Non-overlapping pairs ONLY (meaningful comparisons)
- ✅ Cross-seed comparisons ONLY (seed robustness)
- ✅ Final checkpoint only (not timeline analysis)
- ✅ All 4 layers (3, 4, 5, 6)

## Files Created/Modified

### Created:
- `src/scripts/generate_exp2_seed_cka_configs.py`
- `src/scripts/generate_exp2_seed_cka_run_scripts.py`
- `configs/revision/exp2/pt2_seed_cka/` (336 configs)
- `configs/revision/exp2/pt3_seed_cka/` (168 configs)
- `scripts/revision/exp2/cka_analysis/` (11 scripts)
- `docs/logs/2025-11-21/exp2_seed_cka_infrastructure.md` (this file)

## Next Steps

1. Run the CKA calculations (504 total)
2. Create visualization scripts for seed robustness matrices
3. Generate CKA trends plot (PT1 vs PT2 vs PT3)
4. Statistical analysis of seed stability

## Notes

- PT2-8 and PT3-8 deliberately excluded (seed variants not trained)
- Configs point to correct paths for orig (main experiments/) vs seeds (revision/exp2/)
- Scripts use `analyze_cka_pair.py` from exp4 infrastructure (CKA v2)
- All scripts are executable and follow minimalistic bash convention (no comments in bash files)
