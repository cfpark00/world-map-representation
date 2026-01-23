# Exp2 Seed CKA Infrastructure - FINAL (Fixed)

**Date:** 2025-11-21
**Status:** Complete and optimized

## Summary

Created complete CKA analysis infrastructure for PT2/PT3 seed robustness, focusing on:
- ✅ **Non-overlapping task pairs only** (meaningful comparisons)
- ✅ **Unique seed combinations only** (no redundant calculations due to CKA symmetry)
- ✅ **All 4 layers** (3, 4, 5, 6)

## Key Fix

**Initial version had redundant calculations:**
- Was computing both `orig vs 1` AND `1 vs orig` (symmetric duplicates)
- 504 configs → **252 configs (50% reduction!)**

**Fixed by using unique seed pairs only:**
```python
for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:  # Only unique pairs
```

## Final Numbers

### Total: 252 CKA calculations

**PT2: 168 calculations**
- 14 non-overlapping pairs × 4 layers × 3 unique seed combinations = 168
- Seed pairs: `orig-vs-1`, `orig-vs-2`, `1-vs-2`

**PT3: 84 calculations**
- 7 non-overlapping pairs × 4 layers × 3 unique seed combinations = 84
- Seed pairs: `orig-vs-1`, `orig-vs-2`, `1-vs-2`

## Non-overlapping Pairs

### PT2 (14 pairs)
Task combinations that share NO tasks:
```
pt2-1 (distance, trianglearea) vs:
  - pt2-2 (angle, compass)
  - pt2-3 (inside, perimeter)
  - pt2-6 (compass, inside)
  - pt2-7 (perimeter, crossing)

pt2-2 (angle, compass) vs:
  - pt2-3 (inside, perimeter)
  - pt2-4 (crossing, distance)
  - pt2-7 (perimeter, crossing)

pt2-3 (inside, perimeter) vs:
  - pt2-4 (crossing, distance)
  - pt2-5 (trianglearea, angle)

pt2-4 (crossing, distance) vs:
  - pt2-5 (trianglearea, angle)
  - pt2-6 (compass, inside)

pt2-5 (trianglearea, angle) vs:
  - pt2-6 (compass, inside)
  - pt2-7 (perimeter, crossing)

pt2-6 (compass, inside) vs:
  - pt2-7 (perimeter, crossing)
```

### PT3 (7 pairs)
Task combinations that share NO tasks:
```
pt3-1 (distance, trianglearea, angle) vs pt3-2 (compass, inside, perimeter)
pt3-1 (distance, trianglearea, angle) vs pt3-7 (inside, perimeter, crossing)
pt3-2 (compass, inside, perimeter) vs pt3-3 (crossing, distance, trianglearea)
pt3-3 (crossing, distance, trianglearea) vs pt3-4 (angle, compass, inside)
pt3-4 (angle, compass, inside) vs pt3-5 (perimeter, crossing, distance)
pt3-5 (perimeter, crossing, distance) vs pt3-6 (trianglearea, angle, compass)
pt3-6 (trianglearea, angle, compass) vs pt3-7 (inside, perimeter, crossing)
```

## Files Created

### Generator Scripts (2)
- `src/scripts/generate_exp2_seed_cka_configs.py` - Config generator
- `src/scripts/generate_exp2_seed_cka_run_scripts.py` - Script generator

### Configs (252)
- `configs/revision/exp2/pt2_seed_cka/` - 168 configs
- `configs/revision/exp2/pt3_seed_cka/` - 84 configs
- Structure: `{pt2,pt3}_seed_cka/{prefix}-{i}_vs_{prefix}-{j}/layer{N}_{seed1}_vs_{seed2}.yaml`

### Execution Scripts (11)
**PT2 (5 scripts):**
- `run_pt2_seed_cka_l3.sh` - 42 calculations (14 pairs × 3 seed combos)
- `run_pt2_seed_cka_l4.sh` - 42 calculations
- `run_pt2_seed_cka_l5.sh` - 42 calculations
- `run_pt2_seed_cka_l6.sh` - 42 calculations
- `run_pt2_seed_cka_all_layers.sh` - Master (168 total)

**PT3 (5 scripts):**
- `run_pt3_seed_cka_l3.sh` - 21 calculations (7 pairs × 3 seed combos)
- `run_pt3_seed_cka_l4.sh` - 21 calculations
- `run_pt3_seed_cka_l5.sh` - 21 calculations
- `run_pt3_seed_cka_l6.sh` - 21 calculations
- `run_pt3_seed_cka_all_layers.sh` - Master (84 total)

**Combined (1 script):**
- `run_exp2_seed_cka_all.sh` - Master for PT2+PT3 (252 total)

Location: `scripts/revision/exp2/cka_analysis/`

## Usage

### Run Everything
```bash
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_all.sh
```

### Run by Experiment Type
```bash
bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_all_layers.sh  # PT2 only
bash scripts/revision/exp2/cka_analysis/run_pt3_seed_cka_all_layers.sh  # PT3 only
```

### Run by Layer
```bash
bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_l5.sh  # PT2 layer 5
bash scripts/revision/exp2/cka_analysis/run_pt3_seed_cka_l6.sh  # PT3 layer 6
```

## Output Structure

Results saved to: `data/experiments/revision/exp2/cka_analysis/`

```
cka_analysis/
├── pt2-1_vs_pt2-2_seed1/
│   └── layer5/
│       ├── cka_results.json
│       ├── cka_values.csv
│       └── config.yaml
├── pt2-1_seed1_vs_pt2-2_seed2/
│   └── layer5/
│       └── ...
└── ...
```

## Next Steps

Once calculations complete, these enable:

1. **Seed robustness matrices** (similar to Exp4's 21×21)
2. **CKA trends across task regimes:**
   - PT1-X (single task, 7 variants)
   - PT2 (two tasks, 8 variants)
   - PT3 (three tasks, 8 variants)
   - Plot: CKA vs number of training tasks
3. **Statistical analysis:**
   - Intra-variant seed stability
   - Inter-variant task similarity
   - Layer-wise trends

## Optimization Impact

**Before fix:** 504 calculations (redundant)
**After fix:** 252 calculations (optimal)
**Savings:** 50% compute time, 50% storage

This is the minimal set needed because:
- CKA(A,B) = CKA(B,A) → only need one direction
- Only non-overlapping pairs are meaningful
- Only cross-seed pairs show robustness

## Technical Details

**Config format:**
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
checkpoint_steps: [328146]  # Final checkpoint only
city_filter: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$
kernel_type: linear
center_kernels: true
use_gpu: true
save_timeline_plot: false
output_dir: .../revision/exp2/cka_analysis/pt2-1_vs_pt2-2_seed1/layer5
```

**Path handling:**
- Original models (seed 42): `data/experiments/pt{2,3}-{1-8}/`
- Seed variants: `data/experiments/revision/exp2/pt{2,3}-{1-7}_seed{1,2}/`
- Note: pt2-8 and pt3-8 seeds not trained, excluded from analysis

## Documentation

This document supersedes the initial version that had redundant calculations.
All infrastructure is now optimized and ready to run.
