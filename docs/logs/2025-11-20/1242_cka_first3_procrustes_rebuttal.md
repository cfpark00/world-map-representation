# Session Log: 2025-11-20 12:42 - CKA First-3-PCs and Procrustes Distance Analysis for Rebuttals

## Overview
Implemented PCA-based CKA and Procrustes distance analysis on first 3 principal components to address rebuttal concerns. CKA was showing similar inter-seed and inter-task variability despite clear geometric differences in 3D PCA visualizations. Added Procrustes analysis as a shape-sensitive alternative metric.

## Problem Context
**Issue**: In raw PCA 3D plots, each task generates interpretable geometric shapes with reasonable seed variance. However, CKA analysis destroys this distinction - inter-seed and inter-task variability appear similar.

**Root cause**: CKA is invariant to any invertible linear transformation, so it cannot distinguish between:
- Task A with shape X in 3D space
- Task B with shape Y in 3D space (different geometric structure)

If both span similar subspace dimensions, CKA treats them as similar even though **geometry is completely different**.

## Solution: Dual Analysis Approach

### 1. CKA on First 3 PCs
**Purpose**: Measure subspace similarity while focusing on meaningful variance

**Implementation**:
- Each experiment gets independent PCA (3 components)
- PCA fitted on non-Atlantis cities only
- CKA computed on PCA-reduced representations
- Shows: "Do they span similar subspaces?"

### 2. Procrustes Distance on First 3 PCs
**Purpose**: Measure geometric shape similarity (CKA's blind spot)

**Implementation**:
- Same PCA preprocessing as CKA
- Procrustes optimally aligns point clouds (rotation + scaling)
- Measures residual distance after alignment
- Shows: "Do they have the same geometric shape?"

**Expected results**:
- **Low Procrustes** → same task, different seeds (similar shapes)
- **High Procrustes** → different tasks (different geometric structures)

## Main Tasks Completed

### 1. CKA First-3-PCs Infrastructure

**Created Core Analysis Script**: `src/scripts/analyze_cka_pair_pca.py`
- Applies PCA to each checkpoint independently
- `n_pca_components` configurable (set to 3)
- `pca_train_filter` excludes Atlantis from PCA fitting
- `city_filter` controls which cities used for CKA computation
- Outputs: timeline CSV, summary JSON, optional timeline plot

**Generated 1351 Config Files**: `configs/revision/exp4/cka_cross_seed_first3/`
- Reused structure from full CKA configs
- Added `n_pca_components: 3`
- Added `pca_train_filter: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$`
- Output to `cka_analysis_first3` directory
- Created via `src/scripts/generate_cka_first3_configs.py`

**Bash Scripts** in `scripts/revision/exp4/cka_analysis_first3/`:
- `run_14x14_l5_first3.sh` - Layer 5, original + seed1 only (14×14 matrix)
- `run_21x21_cka_first3.sh` - Layers 5 & 4, excludes seed3
- `run_all_cka_cross_seed_first3.sh` - All layers, all seeds

**Visualization Script**: `src/analysis/cka_v2/visualization/plot_14x14_cka_matrix_first3.py`
- Generates 14×14 CKA heatmap (original + seed1, layer 5)
- Generates 7×7 task-averaged matrix
- Bar plot: intra-task vs inter-task CKA (excluding crossing task)
- Uses magma colormap with three-slope normalization
- Run via: `plot_14x14_cka_l5_first3.sh`

### 2. Procrustes Distance Analysis (New!)

**Created Core Analysis Script**: `src/scripts/analyze_procrustes_pair_pca.py`
- Same PCA preprocessing as CKA analysis
- Uses `scipy.spatial.procrustes` for optimal alignment
- Procrustes distance = standardized residual after alignment
- Measures pure geometric shape differences

**Procrustes Logic**:
1. Center both point clouds (remove translation)
2. Normalize to unit variance (remove scaling)
3. Find optimal rotation via SVD
4. Compute residual distance (shape difference)

**Key properties**:
- Invariant to: translation, rotation, uniform scaling, reflection
- Sensitive to: **geometric shape** and point-to-point correspondence
- Range: 0 (identical) to ~1 (very different)

**Generated 1351 Config Files**: `configs/revision/exp4/procrustes_cross_seed_first3/`
- Reused CKA first3 configs structure
- Changed output paths: `cka_analysis_first3` → `procrustes_analysis_first3`
- Created via `src/scripts/generate_procrustes_first3_configs.py`

**Bash Scripts** in `scripts/revision/exp4/procrustes_analysis_first3/`:
- `run_14x14_l5_first3.sh` - Run 14×14 Procrustes analysis

**Visualization Scripts**:
1. **Distance plot**: `src/analysis/cka_v2/visualization/plot_14x14_procrustes_matrix_first3.py`
   - Viridis colormap (lower = more similar)
   - Shows raw Procrustes distances
   - Run via: `plot_14x14_procrustes_l5_first3.sh`

2. **Similarity plot**: `src/analysis/cka_v2/visualization/plot_14x14_procrustes_similarity_first3.py`
   - Converts distance to similarity: `1 - distance`
   - Magma colormap (higher = more similar, CKA-style)
   - Three-slope normalization for visual consistency
   - Run via: `plot_14x14_procrustes_similarity_l5_first3.sh`

Both generate:
- 14×14 full matrix heatmap
- 7×7 task-averaged matrix
- Bar plot: intra-task vs inter-task comparison

### 3. Naming Convention Update: First5 → First3

**Context**: Initially implemented with 5 PCs, changed to 3 PCs for consistency with visualization

**Renamed everywhere**:
- Directory: `cka_analysis_first5` → `cka_analysis_first3`
- Configs: `cka_cross_seed_first5` → `cka_cross_seed_first3`
- Scripts: All `first5` → `first3` in filenames and paths
- Python: `plot_14x14_cka_matrix_first5.py` → `plot_14x14_cka_matrix_first3.py`

**Updated**:
- All 1351 config files regenerated with `n_pca_components: 3`
- All output paths updated to use `first3`
- All script references updated
- All visualization labels updated to "First 3 PCs"

### 4. Testing and Validation

**Tested CKA analysis** on pt1-1 vs pt1-2 (layer 6):
- Successfully loaded 41 checkpoints, filtered to 1
- Applied PCA (3 components) independently to each
- 4413 common cities after filtering
- Computed CKA on 4413×3 matrices
- Result: CKA = 0.4605
- Runtime: ~3.3 seconds per checkpoint

**Tested Procrustes analysis** on pt1-1 vs pt1-2 (layer 5):
- Same PCA preprocessing
- 4413 common cities, 3 PCs each
- Procrustes distance = 0.6197
- Similarity = 1 - 0.6197 = 0.3803
- Runtime: ~0.3 seconds per checkpoint (10x faster than CKA!)

## Technical Details

### CKA Configuration Structure
```yaml
center_kernels: true
checkpoint_steps: [328146]
city_filter: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$
exp1:
  name: pt1-1
  repr_dir: data/experiments/pt1-1/analysis_higher/distance_firstcity_last_and_trans_l5/representations
  task: distance
exp2:
  name: pt1-2
  repr_dir: data/experiments/pt1-2/analysis_higher/trianglearea_firstcity_last_and_trans_l5/representations
  task: trianglearea
kernel_type: linear
layer: 5
output_dir: /n/home12/cfpark00/WM_1/data/experiments/revision/exp4/cka_analysis_first3/pt1-1_vs_pt1-2/layer5
n_pca_components: 3
pca_train_filter: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$
```

### Filter Logic
- `pca_train_filter`: Used when fitting PCA (excludes Atlantis)
- `city_filter`: Used when computing CKA/Procrustes (also excludes Atlantis)
- Both filters exclude Atlantis and small cities (< 1000 ID)

### Output Structure
**CKA outputs** (`cka_analysis_first3/`):
- `cka_timeline.csv` - CKA value per checkpoint
- `summary.json` - Statistics (final, mean, std, min, max)
- `cka_timeline.png` - Optional timeline plot

**Procrustes outputs** (`procrustes_analysis_first3/`):
- `procrustes_timeline.csv` - Distance per checkpoint
- `summary.json` - Statistics
- `procrustes_timeline.png` - Timeline plot

## Key Files Created

**Analysis Scripts:**
- `src/scripts/analyze_cka_pair_pca.py` - CKA with PCA preprocessing
- `src/scripts/analyze_procrustes_pair_pca.py` - Procrustes with PCA preprocessing

**Config Generators:**
- `src/scripts/generate_cka_first3_configs.py` - Generate CKA configs (updated from first5)
- `src/scripts/generate_procrustes_first3_configs.py` - Generate Procrustes configs

**Visualization Scripts:**
- `src/analysis/cka_v2/visualization/plot_14x14_cka_matrix_first3.py`
- `src/analysis/cka_v2/visualization/plot_14x14_procrustes_matrix_first3.py`
- `src/analysis/cka_v2/visualization/plot_14x14_procrustes_similarity_first3.py`

**Bash Execution Scripts:**
- `scripts/revision/exp4/cka_analysis_first3/run_14x14_l5_first3.sh`
- `scripts/revision/exp4/cka_analysis_first3/plot_14x14_cka_l5_first3.sh`
- `scripts/revision/exp4/procrustes_analysis_first3/run_14x14_l5_first3.sh`
- `scripts/revision/exp4/procrustes_analysis_first3/plot_14x14_procrustes_l5_first3.sh`
- `scripts/revision/exp4/procrustes_analysis_first3/plot_14x14_procrustes_similarity_l5_first3.sh`

**Config Files:**
- 1351 CKA configs in `configs/revision/exp4/cka_cross_seed_first3/`
- 1351 Procrustes configs in `configs/revision/exp4/procrustes_cross_seed_first3/`

## Comparison: CKA vs Procrustes

| Metric | What it measures | Invariant to | Sensitive to | Range |
|--------|-----------------|--------------|--------------|-------|
| **CKA** | Subspace similarity | Any invertible linear transform | Intrinsic dimension | 0-1 (higher = similar) |
| **Procrustes** | Geometric shape | Rotation, translation, scale | Point cloud geometry | 0-1 (lower = similar) |

**Complementary analysis**: CKA shows dimensional similarity, Procrustes shows shape similarity

## Expected Rebuttal Impact

**Problem we're addressing**:
- Reviewers: "PCA shows clear task structure but your quantitative analysis doesn't"
- Old approach: CKA alone (blind to shape differences)

**New approach**:
- **CKA on first 3 PCs**: "Tasks span similar 3D subspaces" (explains high CKA)
- **Procrustes on first 3 PCs**: "But geometric shapes are distinct" (low intra-task, high inter-task distance)

**Key claim**: Task representations have consistent geometric structure across seeds (low Procrustes variance), but different tasks have different geometric shapes (high Procrustes distance between tasks).

## Next Steps
1. Run 14×14 CKA analysis (layer 5, original + seed1)
2. Run 14×14 Procrustes analysis (layer 5, original + seed1)
3. Generate all visualization plots
4. Compare CKA vs Procrustes matrices side-by-side
5. Write rebuttal text explaining complementary metrics

## Notes
- Procrustes is ~10× faster than CKA (no kernel matrix computation)
- Both use identical PCA preprocessing (ensures fair comparison)
- Atlantis excluded from PCA training to avoid bias
- Three-slope normalization maintains visual consistency across plots
- Similarity plots (`1 - distance`) allow direct visual comparison with CKA
