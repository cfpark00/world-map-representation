# Cross-Seed CKA Analysis: 21×21 Matrix

## Overview

Complete infrastructure for **21×21 CKA similarity matrix** comparing:
- **7 original PT1 experiments** (pt1-1 through pt1-7, trained with seed 42)
- **7 seed1 PT1 experiments** (pt1-1_seed1 through pt1-7_seed1, trained with seed 1)
- **7 seed2 PT1 experiments** (pt1-1_seed2 through pt1-7_seed2, trained with seed 2)

All using **correct task-specific prompts**: `distance_firstcity_last_and_trans`, `trianglearea_firstcity_last_and_trans`, etc.

## Generated Infrastructure

### Representation Extraction Configs
- **56 configs total** in `configs/analysis_representation_higher/`
  - `seed1/` - 28 configs (7 tasks × 4 layers)
  - `seed2/` - 28 configs (7 tasks × 4 layers)
  - Original experiments already have representations

### CKA Analysis Configs
- **924 configs total** in `configs/analysis_v2/cka_cross_seed/`
  - 231 unique pairs × 4 layers
  - Covers all combinations:
    - Original vs Original
    - Original vs Seed1
    - Original vs Seed2
    - Seed1 vs Seed1
    - Seed1 vs Seed2
    - Seed2 vs Seed2

## Workflow

### Step 1: Extract Representations for Seed1

```bash
bash scripts/revision/exp4/representation_extraction/extract_seed1_representations.sh
```

Extracts representations for all pt1-X_seed1 experiments (28 jobs).

### Step 2: Extract Representations for Seed2

**IMPORTANT**: Only run this AFTER seed2 experiments are trained!

```bash
bash scripts/revision/exp4/representation_extraction/extract_seed2_representations.sh
```

Extracts representations for all pt1-X_seed2 experiments (28 jobs).

### Step 3: Compute 21×21 CKA Matrix

```bash
bash scripts/revision/exp4/cka_analysis/run_all_cka_cross_seed.sh
```

Computes CKA for all 231 pairs at layer 5.

## Current Status

✅ **Ready**:
- Original PT1 representations (already exist)
- Seed1 experiments trained
- All configs generated
- All scripts ready

⏳ **Waiting**:
- Seed2 experiments need to be trained first
- Use configs in `/configs/revision/exp4/pt1_single_task_seed/pt1-X/pt1-X_seed2.yaml`

## Matrix Structure

The 21×21 matrix will contain:

```
         pt1-1  pt1-2  ... pt1-7  pt1-1_s1  ... pt1-7_s1  pt1-1_s2  ... pt1-7_s2
pt1-1      1.0    ?    ...   ?        ?      ...    ?        ?      ...    ?
pt1-2       .    1.0   ...   ?        ?      ...    ?        ?      ...    ?
...
pt1-7       .     .    ... 1.0       ?      ...    ?        ?      ...    ?
pt1-1_s1    .     .    ...   .      1.0     ...    ?        ?      ...    ?
...
pt1-7_s1    .     .    ...   .       .      ...   1.0       ?      ...    ?
pt1-1_s2    .     .    ...   .       .      ...    .       1.0     ...    ?
...
pt1-7_s2    .     .    ...   .       .      ...    .        .      ...   1.0
```

## Key Files

**Scripts**:
- `scripts/revision/exp4/representation_extraction/extract_seed1_representations.sh` - Extract seed1 (ready to run)
- `scripts/revision/exp4/representation_extraction/extract_seed2_representations.sh` - Extract seed2 (run after training)
- `scripts/revision/exp4/cka_analysis/run_all_cka_cross_seed.sh` - Compute all CKA

**Generators**:
- `src/scripts/generate_repr_configs_seed1_simple.py --seeds 1,2` - Generate repr configs
- `src/scripts/generate_cka_configs_cross_seed.py --seeds 1,2` - Generate CKA configs

**Configs**:
- `configs/analysis_representation_higher/seed1/` - Seed1 repr extraction
- `configs/analysis_representation_higher/seed2/` - Seed2 repr extraction
- `configs/analysis_v2/cka_cross_seed/` - All CKA comparisons

## Expected Insights

The 21×21 CKA matrix will reveal:

1. **Within-seed task similarity**: How different tasks organize within same seed
2. **Cross-seed robustness**: How similar same task is across different seeds
3. **Task × Seed interactions**: Whether certain tasks are more/less robust to seed changes
4. **Representation stability**: Diagonal blocks show within-seed structure

High CKA for same task across seeds → Robust representations
Low CKA for different tasks within seed → Task-specific representations

## Notes

- All representation extractions use seed 42 for PCA probe training
- Original experiments: trained with seed 42
- Seed1 experiments: trained with seed 1
- Seed2 experiments: trained with seed 2 (not yet trained!)
- City filter: `region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$`
