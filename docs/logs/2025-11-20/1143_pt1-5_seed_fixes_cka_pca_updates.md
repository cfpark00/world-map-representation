# Session Log: 2025-11-20 11:43 - PT1-5 Seed Fixes, CKA Analysis, and PCA Timeline Updates

## Overview
Major session fixing pt1-5 (inside task) seed infrastructure due to seed2 training failure, updating CKA analysis paths, regenerating correlation plots, and configuring PCA timeline visualizations.

## Main Tasks Completed

### 1. PT1-5 Seed4 Infrastructure (Inside Task Seed Replacement)
**Problem**: pt1-5_seed2 training failed (inside task is brittle)
**Solution**: Use seed3 instead of seed2 in 21×21 matrix, seed4 for future 28×28 analysis

**Created:**
- 4 representation extraction configs for pt1-5_seed4 (layers 3,4,5,6)
  - Location: `configs/revision/exp4/representation_extraction/seed4/pt1-5_seed4/`
- Extraction script: `scripts/revision/exp4/representation_extraction/extract_pt1-5_seed4_representations.sh`
- 112 CKA configs (28 pairs × 4 layers) for pt1-5_seed4 comparisons
  - Generated via `src/scripts/generate_pt1-5_seed4_cka_configs.py`
  - Compares pt1-5_seed4 vs all other experiments (orig, seed1, seed2, seed3)
- Run script: `scripts/revision/exp4/cka_analysis/run_pt1-5_seed4_cka.sh`

**Seed Mapping for PT1-5:**
- 21×21 matrix: orig (seed42), seed1, **seed3** (not seed2)
- Future 28×28: orig, seed1, seed3, **seed4**

### 2. CKA Config Path Fixes
**Problem**: Many CKA configs pointed to wrong paths
- Original experiments at `data/experiments/pt1-X/`
- Configs incorrectly pointed to `data/experiments/revision/exp4/pt1-X/`

**Solution:**
- Created `src/scripts/fix_cka_config_paths.py`
- Fixed 147 CKA configs to use correct paths
- Added `get_experiment_path()` function to handle original vs seeded experiment paths

### 3. Updated 21×21 CKA Plotting Scripts
**Updated all 3 layer plot scripts** to reflect pt1-5 seed changes:
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix.py` (layer 5)
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l4.py` (layer 4)
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l6.py` (layer 6)

**Changes:**
- `model_names[14]` now uses `pt1-5_seed3` instead of `pt1-5_seed2`
- Labels show "s3" instead of "s2" for pt1-5
- Added comments explaining seed2 training failure

### 4. Reorganized run_seed3_cka.sh Script
**Updated execution order** to prioritize pt1-5:
1. pt1-5_seed3 layer 5 FIRST
2. pt1-5_seed3 layer 4
3. Everything else layer 5
4. Everything else layer 4

### 5. Regenerated CKA Correlation Plots
**Path fixes for** `scratch/cka_to_generalization_v2/plot_cka_generalization_correlation.py`:
- Updated CKA matrix path to absolute
- Updated PT1 baseline paths to absolute
- Added p-value to plot title

**New results with corrected 21×21 matrix:**
- Pearson r = 0.405, p = 0.0264 (statistically significant)
- R² = 0.164 (16.4% variance explained)
- 30 data points (6 tasks × 5 targets, excluding crossing)

**Explained p-value test:**
- Null hypothesis: no correlation between CKA and transfer performance
- p = 0.0264 means 2.64% chance of seeing this correlation if null true
- p < 0.05 threshold → reject null → correlation is real

### 6. PCA Timeline Configuration Overhaul
**Goal**: Exclude Atlantis from probe training but include in visualization

**Generated configs for all seeds (1, 2, 3):**
- Updated existing mixed configs (21 configs) to exclude Atlantis
- Generated new raw (pure PCA) configs:
  - `seed1_raw/`: 7 configs
  - `seed2_raw/`: 6 configs (no pt1-5_seed2)
  - `seed3_raw/`: 7 configs
  - `original_raw/`: 7 configs

**Final probe configuration:**
- `probe_train: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$`
  - Excludes Atlantis from linear probe training
- `probe_test: region:.* && city_id:^[1-9][0-9]{3,}$`
  - Includes Atlantis in visualization (preserves original color scheme)

**Generated run scripts:**
- `pca_timeline_original_all.sh` (7 tasks, mixed)
- `pca_timeline_original_raw_all.sh` (7 tasks, raw)
- `pca_timeline_seed1_all.sh` (7 tasks, mixed)
- `pca_timeline_seed1_raw_all.sh` (7 tasks, raw)
- `pca_timeline_seed2_all.sh` (7 tasks, mixed)
- `pca_timeline_seed2_raw_all.sh` (6 tasks, raw)
- `pca_timeline_seed3_all.sh` (7 tasks, mixed)
- `pca_timeline_seed3_raw_all.sh` (7 tasks, raw)

**Color scheme explanation:**
- Regions sorted alphabetically, assigned sequential colors
- Including Atlantis in probe_test maintains original color mapping
- Excluding it from probe_train prevents Atlantis from affecting linear probe

### 7. 7×7 Averaged CKA Matrix
**Verified seed swap handling:**
- 21×21 matrix positions 12-14: pt1-5, pt1-5_seed1, pt1-5_seed3
- 7×7 averaging automatically uses correct seeds per position
- Diagonal averaging: 6 cross-seed comparisons (excludes self-comparisons)
- For pt1-5: averages orig×seed1, orig×seed3, seed1×seed3

## Key Files Created/Modified

**New Scripts:**
- `src/scripts/generate_pt1-5_seed4_cka_configs.py`
- `src/scripts/fix_cka_config_paths.py`
- `src/scripts/generate_all_pca_timeline_configs.py`
- `src/scripts/generate_pca_timeline_run_scripts.py`
- `src/scripts/fix_pca_timeline_atlantis.py`
- `src/scripts/update_pca_timeline_filters.py`

**Modified Scripts:**
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix.py` (all 3 layer versions)
- `scripts/revision/exp4/cka_analysis/run_seed3_cka.sh`
- `scratch/cka_to_generalization_v2/plot_cka_generalization_correlation.py`

**Config Summary:**
- 112 new CKA configs for pt1-5_seed4
- 147 fixed CKA configs (path corrections)
- 55 updated PCA timeline configs (Atlantis handling)
- 27 new raw PCA timeline configs

## Statistics and Context

**R² vs p-value relationship** (for n=30):
- Minimum |r| for p<0.05: ~0.36
- Our r = 0.405 > 0.36 → significant
- No direct R²→p formula without sample size

**21×21 CKA Matrix Status:**
- All 231 pairs loaded successfully
- Missing: 0 pairs
- Uses pt1-5_seed3 instead of pt1-5_seed2

## Next Steps
- Run pt1-5_seed4 representation extraction
- Run pt1-5_seed4 CKA analysis
- Execute PCA timeline visualizations for all seeds
- Consider full 28×28 matrix expansion if needed

## Notes
- pt1-5 (inside task) is brittle across seeds
- Seed infrastructure now handles this gracefully
- PCA color scheme preserved by including Atlantis in probe_test
- All configs use consistent Atlantis exclusion from probe training
