# 16:26 - Canonical Experiments Documentation & Visualization Export

## Summary
Documented the complete canonical experiment set (179 experiments) and exported all PCA 3D visualizations for Group 1 (PT1-X) and Group 2 (PT1 + FTWB) to `data/vis/` for easy access.

## Tasks Completed

### 1. Created Canonical Experiments Documentation
- **Created**: `docs/canonical_experiments.md`
- **Purpose**: Comprehensive inventory of all 179 experiments used for the paper revision
- **Structure**:
  - **Group 1** (63 experiments): PT1-X, PT2, PT3 with 3 seeds each
  - **Group 2** (116 experiments): PT1 base (4 seeds) + FTWB1 (28) + FTWB2 (84)
- **Key details documented**:
  - Exact directory paths for all experiments
  - Task definitions for each experiment type
  - Seed naming convention (seed0 = original seed42)
  - **Important exception**: pt1-5 uses seed3 instead of seed2 (training failed)

### 2. Verified Group 1 PCA 3D Visualizations (PT1-X)
- **Checked**: 21 experiments × 2 types (mixed, raw) = 42 files
- **Status**: All 42 files present ✓
- **Location**: `analysis_higher/*_firstcity_last_and_trans_l5/pca_timeline{,_raw}/`

### 3. Verified & Generated Group 2 PCA 3D Visualizations
- **PT1 Base**: Found seeds missing plotly files
  - Created: `src/scripts/generate_pt1_base_pca_configs.py`
  - Created: `scripts/revision/exp1/pca/run_pt1_base_pca_all.sh`
  - Generated 12 configs (4 models × 3 types: mixed, raw, na)
  - Ran script to generate all missing plotly files
- **Final status**: All 116 experiments × 3 types = 348 files ✓

### 4. Exported Visualizations to data/vis/
- **group1_pt1/** (42 files):
  - Naming: `pt1-{1-7}_seed{0,1,2}_{mixed,raw}.html`
  - Note: pt1-5 uses seed3 in place of seed2
  - Renamed originals from no-seed to seed0 for consistency

- **group2/** (348 files):
  - Naming: `pt1_seed{0,1,2,3}_{mixed,raw,na}.html` (base models)
  - Naming: `pt1_seed{0,1,2,3}_ftwb1-{1-7}_{mixed,raw,na}.html`
  - Naming: `pt1_seed{0,1,2,3}_ftwb2-{1-21}_{mixed,raw,na}.html`

### 5. Created Zip Archives
- `data/vis/group1_pt1.zip` - 121 MB (42 HTML files)
- `data/vis/group2.zip` - 1.0 GB (348 HTML files)

### 6. Updated PT2 CKA Matrix Plots
- Added numeric annotations to 21×21 matrix (fontsize=7)
- Added red triangles to 7×7 averaged matrices for partially overlapping task pairs
- Partial overlap pairs: (1,4), (1,5), (2,5), (2,6), (3,6), (3,7), (4,7)

## Files Created/Modified

### New Files
- `docs/canonical_experiments.md` - Complete experiment inventory
- `src/scripts/generate_pt1_base_pca_configs.py` - Config generator for PT1 base PCA
- `scripts/revision/exp1/pca/run_pt1_base_pca_all.sh` - Execution script
- `configs/revision/exp1_pt1_pca/*.yaml` - 12 PCA timeline configs
- `data/vis/group1_pt1.zip` - Exported visualizations
- `data/vis/group2.zip` - Exported visualizations

### Modified Files
- `src/scripts/plot_21x21_pt2_cka_matrix_l5.py` - Added numbers and red triangles

## Naming Conventions Established
- `seed0` = original (seed 42)
- `seed1`, `seed2`, `seed3` = additional seeds
- For pt1-5 (inside task): seed2 training failed, seed3 replaces it

## Visualization Types
- **mixed**: Linear probe (x,y from regression) + residual PC for 3rd dimension
- **raw**: Pure PCA (first 3 components)
- **na** (no atlantis): Same as mixed but probe trained excluding Atlantis cities
