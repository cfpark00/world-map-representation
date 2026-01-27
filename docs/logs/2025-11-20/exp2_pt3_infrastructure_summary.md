# Exp2 PT3 Infrastructure Summary

**Date**: 2025-11-20
**Task**: Create representation extraction and PCA plotting infrastructure for PT3 (three-task training)

## Overview

Created complete infrastructure for analyzing representations from PT3 models:
- **PT3**: Three-task pretraining experiments
- **7 variants** (pt3-1 through pt3-7), each with 2 seeds
- **14 total experiments** (pt3-8 not trained yet)

## Files Created

### Configs Generated (56 total)

#### Representation Extraction (14 configs)
- PT3-1 through PT3-7, seeds 1-2
- Layer 5 only, first task from each triple
- Path: `configs/revision/exp2/pt3_seed/extract_representations/`

#### PCA Timeline (42 configs)
- 14 experiments × 3 types (mixed, raw, na)
- Path: `configs/revision/exp2/pt3_seed/pca_timeline/`

### Scripts Generated (12 total)

#### Representation Extraction (8 scripts)
- `extract_pt3-{1-7}.sh` - Individual variant scripts
- `extract_all_pt3.sh` - Master script for all variants

#### PCA Timeline (4 scripts)
- `pca_pt3_mixed.sh` - Mixed PCA for all PT3
- `pca_pt3_raw.sh` - Raw PCA for all PT3
- `pca_pt3_na.sh` - No-Atlantis PCA for all PT3
- `pca_pt3_all.sh` - Master script for all PCA

## PT3 Task Combinations

Each PT3 variant trains on three tasks. For representation extraction, we use the **first task** from each triple:

```
pt3-1: distance+trianglearea+angle      → Extract: distance
pt3-2: compass+inside+perimeter         → Extract: compass
pt3-3: crossing+distance+trianglearea   → Extract: crossing
pt3-4: angle+compass+inside             → Extract: angle
pt3-5: perimeter+crossing+distance      → Extract: perimeter
pt3-6: trianglearea+angle+compass       → Extract: trianglearea
pt3-7: inside+perimeter+crossing        → Extract: inside
pt3-8: distance+trianglearea+compass    → NOT TRAINED YET
```

## Training Status

✓ **Trained** (14 experiments):
- pt3-1_seed1, pt3-1_seed2
- pt3-2_seed1, pt3-2_seed2
- pt3-3_seed1, pt3-3_seed2
- pt3-4_seed1, pt3-4_seed2
- pt3-5_seed1, pt3-5_seed2
- pt3-6_seed1, pt3-6_seed2
- pt3-7_seed1, pt3-7_seed2

✗ **Not Trained** (2 experiments):
- pt3-8_seed1, pt3-8_seed2

## Usage

### Extract representations (all PT3)
```bash
bash scripts/revision/exp2/pt3_seed/extract_representations/extract_all_pt3.sh
```

Or individually:
```bash
bash scripts/revision/exp2/pt3_seed/extract_representations/extract_pt3-1.sh
bash scripts/revision/exp2/pt3_seed/extract_representations/extract_pt3-2.sh
# ... etc
```

### Generate PCA visualizations (all PT3)
```bash
bash scripts/revision/exp2/pt3_seed/pca_timeline/pca_pt3_all.sh
```

Or by type:
```bash
bash scripts/revision/exp2/pt3_seed/pca_timeline/pca_pt3_mixed.sh
bash scripts/revision/exp2/pt3_seed/pca_timeline/pca_pt3_raw.sh
bash scripts/revision/exp2/pt3_seed/pca_timeline/pca_pt3_na.sh
```

## Output Locations

All outputs go to: `/data/experiments/revision/exp2/`

Structure:
```
pt3-{1-7}_seed{1-2}/
  analysis_higher/
    {task}_firstcity_last_and_trans_l5/
      representations/     # Extracted representations
      pca_timeline/        # Mixed PCA visualizations
      pca_timeline_raw/    # Raw PCA visualizations
      pca_timeline_na/     # No-Atlantis PCA
```

## PCA Types

1. **mixed**: Linear probe alignment (PC1→x, PC2→y, PC3→r0), probe trained without Atlantis, test includes Atlantis
2. **raw**: Pure PCA without probe alignment
3. **na** (No Atlantis): Probe trained AND tested without Atlantis

## Comparison to Other Experiments

### Exp1 (PT1 Seed Robustness)
- ✓ Already has repr extraction and PCA infrastructure
- 3 experiments: pt1_seed{1,2,3}
- Uses distance task for all

### Exp2 PT2 (Two-task Seed Robustness)
- ✓ Already has repr extraction and PCA infrastructure
- 16 experiments: pt2-{1-8}_seed{1,2}
- Uses first task from each pair

### Exp2 PT3 (Three-task Seed Robustness) - **NEW**
- ✓ Now has complete infrastructure
- 14 experiments: pt3-{1-7}_seed{1,2}
- Uses first task from each triple

### Exp3 (Width Ablation)
- ✓ Has infrastructure for trained models
- 8 trained: pt1_wide + pt1_wide_ftwb{1-7}
- 8 not trained: pt1_narrow + pt1_narrow_ftwb{1-7}

### Exp4 (Single-task PT1-X)
- ✓ Has complete infrastructure
- 21+ experiments with multiple seeds

## Generation Scripts

- `src/scripts/generate_pt3_repr_pca_configs.py` - Generates all config files
- `src/scripts/generate_pt3_run_scripts.py` - Generates all bash scripts

Both scripts are rerunnable and will overwrite existing files.

## Notes

- PT3-8 configs and scripts are deliberately excluded (not trained)
- When PT3-8 training completes, re-run the generator scripts to create its infrastructure
- All PT3 experiments use layer 5 only (consistent with other revision experiments)
