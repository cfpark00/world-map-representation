# Exp3 PCA Scripts Summary

**Date**: 2025-11-20  
**Task**: Create representation extraction and PCA plotting infrastructure for Exp3 (width ablation)

## Overview

Created complete infrastructure for analyzing representations from exp3 models:
- **pt1_wide**: 2× width (256 hidden, 1024 intermediate, 8 heads)
- **pt1_narrow**: ½ width (64 hidden, 256 intermediate, 2 heads)
- **ftwb versions**: 7 fine-tuned versions of each (14 total)

## Files Created

### Configs Generated (62 total)

#### Representation Extraction (16 configs)
- **Base models** (2): `pt1_wide`, `pt1_narrow`
  - Layer 5 only, distance task
  - Path: `configs/revision/exp3/representation_extraction/{model}/`
  
- **Fine-tuned models** (14): `pt1_wide_ftwb{1-7}`, `pt1_narrow_ftwb{1-7}`
  - Layer 5 only, task-specific
  - Path: `configs/revision/exp3/representation_extraction/{model}/`

#### PCA Timeline (46 configs)
- **Base models** (4): 2 models × 2 types (mixed, raw)
  - Path: `configs/revision/exp3/pca_timeline/{model}_{type}/`
  
- **Fine-tuned models** (42): 14 models × 3 types (mixed, raw, na)
  - Path: `configs/revision/exp3/pca_timeline/{model}_{type}/`

### Scripts Generated (15 total)

#### Representation Extraction (4 scripts)
- `extract_base_models.sh` - Extract pt1_wide and pt1_narrow
- `extract_wide_ftwb.sh` - Extract all wide fine-tuned models
- `extract_narrow_ftwb.sh` - Extract all narrow fine-tuned models
- `extract_all.sh` - Master script to run all extraction

#### PCA Timeline (11 scripts)
- `pca_base_models_mixed.sh` - Mixed PCA for base models
- `pca_base_models_raw.sh` - Raw PCA for base models
- `pca_wide_ftwb_mixed.sh` - Mixed PCA for wide ftwb
- `pca_wide_ftwb_raw.sh` - Raw PCA for wide ftwb
- `pca_wide_ftwb_na.sh` - No-Atlantis PCA for wide ftwb
- `pca_narrow_ftwb_mixed.sh` - Mixed PCA for narrow ftwb
- `pca_narrow_ftwb_raw.sh` - Raw PCA for narrow ftwb
- `pca_narrow_ftwb_na.sh` - No-Atlantis PCA for narrow ftwb
- `pca_base_models_all.sh` - All base model PCA
- `pca_wide_ftwb_all.sh` - All wide ftwb PCA
- `pca_narrow_ftwb_all.sh` - All narrow ftwb PCA
- `pca_all.sh` - Master script for all PCA

## PCA Types Explained

### Base Models (pt1_wide, pt1_narrow)
1. **mixed**: Linear probe alignment (PC1→x, PC2→y, PC3→r0), probe trained without Atlantis, test includes Atlantis
2. **raw**: Pure PCA without probe alignment, same filters as mixed

### Fine-tuned Models (ftwb versions)
1. **mixed**: Same as base models
2. **raw**: Same as base models  
3. **na** (No Atlantis): Probe trained AND tested without Atlantis
   - This is unique to ftwb versions
   - Allows comparing how the model represents Atlantis when probe hasn't seen it

## Task Mappings

```
Task 1: distance
Task 2: trianglearea
Task 3: angle
Task 4: compass
Task 5: inside
Task 6: perimeter
Task 7: crossing
```

## Usage

### Extract representations
```bash
# All at once
bash scripts/revision/exp3/representation_extraction/extract_all.sh

# Or separately
bash scripts/revision/exp3/representation_extraction/extract_base_models.sh
bash scripts/revision/exp3/representation_extraction/extract_wide_ftwb.sh
bash scripts/revision/exp3/representation_extraction/extract_narrow_ftwb.sh
```

### Generate PCA visualizations
```bash
# All at once
bash scripts/revision/exp3/pca_timeline/pca_all.sh

# Or by category
bash scripts/revision/exp3/pca_timeline/pca_base_models_all.sh
bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_all.sh
bash scripts/revision/exp3/pca_timeline/pca_narrow_ftwb_all.sh

# Or by type
bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_mixed.sh
bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_raw.sh
bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_na.sh
```

## Output Locations

All outputs go to: `/n/home12/cfpark00/WM_1/data/experiments/revision/exp3/`

Structure:
```
{model}/
  analysis_higher/
    {task}_firstcity_last_and_trans_l5/
      representations/     # Extracted representations
      pca_timeline/        # Mixed PCA visualizations
      pca_timeline_raw/    # Raw PCA visualizations
      pca_timeline_na/     # No-Atlantis PCA (ftwb only)
```

## Key Differences from Exp4

1. **Only layer 5**: Exp4 uses layers 3, 4, 5, 6; exp3 focuses on layer 5
2. **NA type for ftwb**: The "na" (no atlantis) PCA type is unique to ftwb models
   - Helps analyze how models generalize to Atlantis when probe hasn't seen it
3. **Base models use distance task**: pt1_wide and pt1_narrow aren't task-specific, so we use distance as default

## Generation Scripts

- `src/scripts/generate_exp3_configs.py` - Generates all config files
- `src/scripts/generate_exp3_run_scripts.py` - Generates all bash scripts

Both scripts are rerunnable and will overwrite existing files.
