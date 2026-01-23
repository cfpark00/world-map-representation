# Exp1 Representation Extraction & PCA Infrastructure

**Date:** 2025-11-21
**Time:** 14:02
**Session:** Revision Exp1 infrastructure completion

## Overview

Created infrastructure for representation extraction and PCA timeline visualization for revision/exp1 FTWB1 and FTWB2 models across all seeds. Also created a histogram analysis comparing distance vs non-distance task delta values.

## Tasks Completed

### 1. FTWB2 vs FTWB1 Colorbar

Created standalone horizontal colorbar for FTWB2 vs FTWB1 difference plots:
- `src/scripts/generate_ftwb2_vs_ftwb1_colorbar.py`
- RdBu colormap, centered at 0, range -0.5 to 0.5
- Output: `data/experiments/revision/exp1/plots/ftwb2_vs_ftwb1_colorbar_horizontal.png`

### 2. FTWB2 Distance Delta Histogram

Created histogram comparing delta (actual - predicted) for FTWB2 experiments involving distance task vs not:
- `src/scripts/plot_ftwb2_distance_delta_histogram.py`
- Aggregated across all 4 seeds (588 total values)
- Key findings:
  - With distance: mean = -0.185 ± 0.180 (168 values)
  - Without distance: mean = 0.077 ± 0.094 (420 values)
- Suggests distance task involvement correlates with underperformance vs prediction
- Output: `data/experiments/revision/exp1/plots/ftwb2_distance_delta_histogram.png`

### 3. Representation Extraction & PCA Configs

Created config generator: `src/scripts/generate_revision_exp1_repr_pca_configs.py`

**Coverage:**
- FTWB1: original + seeds 1,2,3 (28 models total)
- FTWB2: seeds 1,2,3 only (63 models - original already done)
- Layer 5 extraction
- 3 PCA types per model: mixed, na, raw

**Configs generated:**
- 91 representation extraction configs
- 273 PCA timeline configs
- Total: 364 configs

**Config locations:**
- `configs/revision/exp1/representation_extraction/`
- `configs/revision/exp1/pca_timeline/`

### 4. Execution Scripts

Created script generator: `src/scripts/generate_revision_exp1_repr_pca_scripts.py`

**Scripts generated (22 total):**

Representation extraction (11):
- `extract_original_ftwb1.sh` - Original pt1_ftwb1-{1-7}
- `extract_seed{1,2,3}_ftwb1.sh` - Seed FTWB1 models
- `extract_seed{1,2,3}_ftwb2.sh` - Seed FTWB2 models
- `extract_all_ftwb1.sh`, `extract_all_ftwb2.sh`, `extract_all.sh` - Master scripts

PCA timeline (11):
- `pca_original_ftwb1.sh` - Original pt1_ftwb1-{1-7}
- `pca_seed{1,2,3}_ftwb1.sh` - Seed FTWB1 models
- `pca_seed{1,2,3}_ftwb2.sh` - Seed FTWB2 models
- `pca_all_ftwb1.sh`, `pca_all_ftwb2.sh`, `pca_all.sh` - Master scripts

**Script locations:**
- `scripts/revision/exp1/representation_extraction/`
- `scripts/revision/exp1/pca_timeline/`

### 5. Bug Fix: PCA Config Format

Fixed incorrect PCA config format for "raw" type:
- Changed `type: raw` to `type: pca` (raw not implemented in code)
- Changed axis mapping values from strings (`'x'`, `'y'`, `'z'`) to integers (`0`, `1`, `2`)
- Fixed output directory suffix to `pca_timeline_raw`

## Status Check

**Original FTWB models:**
- pt1_ftwb1-{1-7}: NO `analysis_higher` - need extraction
- pt1_ftwb2-{1-21}: YES `analysis_higher` - already complete

**Commands to run:**
```bash
# Extract representations for original FTWB1
bash scripts/revision/exp1/representation_extraction/extract_original_ftwb1.sh

# Generate PCA visualizations
bash scripts/revision/exp1/pca_timeline/pca_original_ftwb1.sh

# For all seeds
bash scripts/revision/exp1/representation_extraction/extract_all.sh
bash scripts/revision/exp1/pca_timeline/pca_all.sh
```

## Files Created/Modified

### New Files:
- `src/scripts/generate_ftwb2_vs_ftwb1_colorbar.py`
- `src/scripts/plot_ftwb2_distance_delta_histogram.py`
- `src/scripts/generate_revision_exp1_repr_pca_configs.py`
- `src/scripts/generate_revision_exp1_repr_pca_scripts.py`
- 364 config files in `configs/revision/exp1/`
- 22 script files in `scripts/revision/exp1/`
- `data/experiments/revision/exp1/plots/ftwb2_vs_ftwb1_colorbar_horizontal.png`
- `data/experiments/revision/exp1/plots/ftwb2_distance_delta_histogram.png`
