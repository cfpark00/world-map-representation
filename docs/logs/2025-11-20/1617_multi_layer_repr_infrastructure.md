# Multi-Layer Representation Extraction Infrastructure for PT2/PT3 Seeds

**Date**: 2025-11-20
**Time**: 16:17
**Session Focus**: Creating complete 4-layer representation extraction infrastructure for all 63 trained models

## Overview

Completed setup of comprehensive multi-layer (layers 3, 4, 5, 6) representation extraction infrastructure for PT2 and PT3 seed experiments. This ensures all 63 trained models across PT1-X, PT2, and PT3 experiments have complete 4-layer representations needed for CKA trends analysis.

## Context

User requested verification that all 63 trained models have representations extracted for layers 3, 4, 5, 6. This is required for upcoming CKA trends analysis comparing non-overlapping task pairs across PT1-X, PT2, and PT3 experiments (following the analysis pattern from world-representation repo at `/n/home12/cfpark00/datadir/world-representation/scratch/cka_trends`).

## Status Check Results

### Complete (Already Had 4 Layers)
- **PT1-X (21 models)**: ✓ All 7 tasks × 3 seeds have layers 3,4,5,6
  - Original (seed 42): 7/7 ✓
  - Seed1: 7/7 ✓
  - Seed2: 7/7 ✓

- **PT2 Original (8 models)**: ✓ All 8 variants have layers 3,4,5,6
  - pt2-1 through pt2-8: 8/8 ✓

- **PT3 Original (8 models)**: ✓ All 8 variants have layers 3,4,5,6
  - pt3-1 through pt3-8: 8/8 ✓

### Missing Representations (Required Infrastructure)
- **PT2 Seed1 (7 models)**: ⚠ Only had layer 5
  - Needed: layers 3, 4, 6 for all 7 variants
  - pt2-8 skipped (not trained)

- **PT2 Seed2 (7 models)**: ✗ No representations
  - Needed: layers 3, 4, 5, 6 for all 7 variants
  - pt2-8 skipped (not trained)

- **PT3 Seed1 (7 models)**: ✗ No representations
  - Needed: layers 3, 4, 5, 6 for all 7 variants
  - pt3-8 skipped (not trained)

- **PT3 Seed2 (7 models)**: ✗ No representations
  - Needed: layers 3, 4, 5, 6 for all 7 variants
  - pt3-8 skipped (not trained)

**Total Missing**: 105 representation extraction configs needed

## Infrastructure Created

### 1. Config Generation Script

**File**: `src/scripts/generate_pt2_pt3_multilayer_repr_configs.py`

Key features:
- Generates representation extraction configs for layers 3, 4, 5, 6
- Skips layer 5 for PT2 seed1 (already exists)
- Uses task mapping to select FIRST task from each combination
- Proper output directory structure
- Consistent with existing configs

Task mappings used:
```python
PT2_TASKS = {
    1: 'distance',      # distance+trianglearea
    2: 'angle',         # angle+compass
    3: 'inside',        # inside+perimeter
    4: 'crossing',      # crossing+distance
    5: 'trianglearea',  # trianglearea+angle
    6: 'compass',       # compass+inside
    7: 'perimeter',     # perimeter+crossing
}

PT3_TASKS = {
    1: 'distance',      # distance+trianglearea+angle
    2: 'compass',       # compass+inside+perimeter
    3: 'crossing',      # crossing+distance+trianglearea
    4: 'angle',         # angle+compass+inside
    5: 'perimeter',     # perimeter+crossing+distance
    6: 'trianglearea',  # trianglearea+angle+compass
    7: 'inside',        # inside+perimeter+crossing
}
```

**Output**: 105 configs total
- PT2 seed1: 21 configs (7 variants × 3 layers: 3,4,6)
- PT2 seed2: 28 configs (7 variants × 4 layers: 3,4,5,6)
- PT3 seed1: 28 configs (7 variants × 4 layers: 3,4,5,6)
- PT3 seed2: 28 configs (7 variants × 4 layers: 3,4,5,6)

Config locations:
```
configs/revision/exp2/pt2_seed/extract_representations_multilayer/
configs/revision/exp2/pt3_seed/extract_representations_multilayer/
```

### 2. Run Script Generation

**File**: `src/scripts/generate_pt2_pt3_multilayer_run_scripts.py`

Generated 11 bash scripts:
- 5 PT2 scripts: 1 master + 4 individual (seed1_l3, seed1_l4, seed1_l6, seed2_all)
- 5 PT3 scripts: 1 master + 4 individual (seed1_all, seed2_all, seed1_first, seed2_first)
- 1 combined master script

Script locations:
```
scripts/revision/exp2/representation_extraction/
├── extract_pt2_all_multilayer.sh
├── extract_pt2_seed1_l3.sh
├── extract_pt2_seed1_l4.sh
├── extract_pt2_seed1_l6.sh
├── extract_pt2_seed2_all.sh
├── extract_pt3_all_multilayer.sh
├── extract_pt3_seed1_all.sh
├── extract_pt3_seed1_first.sh
├── extract_pt3_seed2_all.sh
├── extract_pt3_seed2_first.sh
└── extract_pt2_pt3_all_multilayer.sh (MASTER)
```

### 3. Config Details

Each config follows this structure:
```yaml
cities_csv: data/datasets/cities/cities.csv
device: cuda
layers: [X]  # One of: 3, 4, 5, 6
method:
  name: linear
n_test_cities: 1250
n_train_cities: 3250
perform_pca: true
probe_test: region:.* && city_id:^[1-9][0-9]{3,}$
probe_train: region:.* && city_id:^[1-9][0-9]{3,}$
save_repr_ckpts: [-2]  # Last checkpoint
seed: 42
experiment_dir: data/experiments/revision/exp2/pt{2,3}-X_seed{1,2}
output_dir: .../analysis_higher/{task}_firstcity_last_and_trans_l{X}
prompt_format: {task}_firstcity_last_and_trans
```

## Verification Process

Created verification script: `/tmp/check_pt1x_repr.sh`

This script checks all 21 PT1-X experiments (original + seed1 + seed2) for complete 4-layer representation extraction:
- Original PT1-X (seed 42): 7 tasks
- PT1-X Seed1: 7 tasks
- PT1-X Seed2: 7 tasks

All 21 experiments confirmed to have representations for layers 3, 4, 5, 6.

## Usage Instructions

### To Extract All Missing Representations

Run the master script:
```bash
bash scripts/revision/exp2/extract_pt2_pt3_all_multilayer.sh
```

This will sequentially run:
1. PT2 seed1 layers 3,4,6 (21 configs)
2. PT2 seed2 all layers (28 configs)
3. PT3 seed1 all layers (28 configs)
4. PT3 seed2 all layers (28 configs)

Total: 105 representation extractions

### To Extract Subsets

Individual scripts available for selective extraction:

**PT2 only:**
```bash
bash scripts/revision/exp2/representation_extraction/extract_pt2_all_multilayer.sh
```

**PT3 only:**
```bash
bash scripts/revision/exp2/representation_extraction/extract_pt3_all_multilayer.sh
```

**PT3 first variant (testing):**
```bash
bash scripts/revision/exp2/representation_extraction/extract_pt3_seed1_first.sh
bash scripts/revision/exp2/representation_extraction/extract_pt3_seed2_first.sh
```

## Expected Runtime

Based on typical representation extraction times:
- ~5-10 minutes per config (GPU dependent)
- 105 configs × ~7 minutes avg = ~735 minutes (~12 hours)

Recommendation: Run overnight or in batches

## Final Status Summary

### Models with Complete 4-Layer Representations
- ✓ PT1-X: 21/21 (7 tasks × 3 seeds)
- ✓ PT2 original: 8/8
- ✓ PT3 original: 8/8

### Models Ready for Extraction
- PT2 seed1: 7/7 (needs layers 3,4,6)
- PT2 seed2: 7/7 (needs layers 3,4,5,6)
- PT3 seed1: 7/7 (needs layers 3,4,5,6)
- PT3 seed2: 7/7 (needs layers 3,4,5,6)

**Total Trained Models**: 63
- PT1-X: 21 ✓ complete
- PT2: 22 (8 original ✓ + 14 seeds ⚠ needs extraction)
- PT3: 22 (8 original ✓ + 14 seeds ⚠ needs extraction)

**Infrastructure Status**: ✓ Complete (105 configs + 11 scripts ready)

## Next Steps

1. **Run extraction**: Execute master script to extract all 105 missing representations
2. **Verify outputs**: Check that all extractions completed successfully
3. **CKA trends setup**: Create configs for non-overlapping task pair CKA analysis
4. **Visualization**: Generate CKA trends plots following world-representation pattern

## Files Modified/Created

### Created
- `src/scripts/generate_pt2_pt3_multilayer_repr_configs.py` (105 configs generator)
- `src/scripts/generate_pt2_pt3_multilayer_run_scripts.py` (11 scripts generator)
- `configs/revision/exp2/pt2_seed/extract_representations_multilayer/*.yaml` (49 configs)
- `configs/revision/exp2/pt3_seed/extract_representations_multilayer/*.yaml` (56 configs)
- `scripts/revision/exp2/representation_extraction/extract_pt2_*_multilayer.sh` (5 scripts)
- `scripts/revision/exp2/representation_extraction/extract_pt3_*_multilayer.sh` (5 scripts)
- `scripts/revision/exp2/extract_pt2_pt3_all_multilayer.sh` (master script)
- `/tmp/check_pt1x_repr.sh` (verification script)
- `docs/logs/2025-11-20/pt2_pt3_multilayer_repr_extraction_summary.md`

### Read/Verified
- Checked PT1-X representation status (21 experiments)
- Verified PT2/PT3 original representations (16 experiments)
- Confirmed PT2/PT3 seed representation gaps (28 experiments)
- Read sample configs from exp2 PT3 to understand format

## Related Work

This infrastructure builds upon:
- Previous PT1-X multi-seed infrastructure (exp4)
- PT2/PT3 single-layer extraction (layer 5 only for some seeds)
- CKA trends analysis from world-representation repo

## Notes

- PT2-8 and PT3-8 skipped (not trained, variants unused)
- Layer 5 already extracted for PT2 seed1, so we skip it to avoid duplication
- All configs use `save_repr_ckpts: [-2]` to extract from last checkpoint
- Prompt formats use task-specific `{task}_firstcity_last_and_trans`
- Output directories follow pattern: `analysis_higher/{task}_firstcity_last_and_trans_l{layer}`

## Technical Details

### Task to Prompt Mapping
```python
TASK_TO_PROMPT = {
    'distance': 'distance_firstcity_last_and_trans',
    'trianglearea': 'trianglearea_firstcity_last_and_trans',
    'angle': 'angle_firstcity_last_and_trans',
    'compass': 'compass_firstcity_last_and_trans',
    'inside': 'inside_firstcity_last_and_trans',
    'perimeter': 'perimeter_firstcity_last_and_trans',
    'crossing': 'crossing_firstcity_last_and_trans',
}
```

### Directory Structure
```
data/experiments/revision/exp2/
├── pt2-1_seed1/analysis_higher/{task}_firstcity_last_and_trans_l{3,4,5,6}/
├── pt2-1_seed2/analysis_higher/{task}_firstcity_last_and_trans_l{3,4,5,6}/
├── pt3-1_seed1/analysis_higher/{task}_firstcity_last_and_trans_l{3,4,5,6}/
└── pt3-1_seed2/analysis_higher/{task}_firstcity_last_and_trans_l{3,4,5,6}/
...
```

## Impact

This infrastructure enables:
1. Complete 4-layer representation coverage for all 63 trained models
2. CKA trends analysis across PT1-X, PT2, PT3 for non-overlapping task pairs
3. Layer-wise analysis of representation formation
4. Systematic comparison of single-task, two-task, and three-task learning dynamics
5. Robustness analysis across different random seeds

## Previous Session Context

This work follows from earlier today's sessions:
1. Morning: Created exp3 PCA infrastructure (width ablation)
2. Afternoon: Created PT3 seed representation extraction (layer 5 only)
3. Late afternoon: Extended to complete 4-layer coverage for all PT2/PT3 seeds

The comprehensive infrastructure now supports the full revision experiment suite.
