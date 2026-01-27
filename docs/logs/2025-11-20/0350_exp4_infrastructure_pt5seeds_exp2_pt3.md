# Exp4 Infrastructure Expansion + PT1-5 Additional Seeds + Exp2 PT3 Setup

**Date:** 2025-11-20
**Time:** 03:50
**Session Duration:** ~2 hours

## Summary
Major infrastructure expansion for exp4 CKA analysis: upgraded from 14x14 to full 28x28 matrix support (4 seeds), added layer 6 analysis, created additional seeds for pt1-5, and properly organized pt3 (three-task) experiments in exp2.

## Main Tasks Completed

### 1. Fixed CKA Config Format Errors
- **Problem:** CKA configs had wrong format causing `KeyError: 'exp1'`
- **Root Cause:** Initial generation used flat structure (`experiment1_dir`, `layer_index`) instead of nested `exp1`/`exp2`
- **Fix Applied:**
  - Examined working config at `configs/revision/exp4/cka_cross_seed/pt1-5_vs_pt1-7/layer5.yaml`
  - Found correct format requires nested exp1/exp2 dictionaries with `name`, `repr_dir`, `task` fields
  - Regenerated ALL 462 configs (231 pairs × 2 layers) with proper structure
  - Updated to include: `center_kernels`, `checkpoint_steps: [328146]`, `kernel_type: linear`, `use_gpu: true`

### 2. Created Seed-Specific CKA Run Scripts
- **Motivation:** Don't want to recalculate already-computed 14x14 CKAs
- **Implementation:**
  - Created `run_seed2_cka.sh` - runs only configs involving seed2 (350 configs: 175 pairs × 2 layers)
  - Created `run_seed3_cka.sh` - runs only configs involving seed3 (350 configs: 175 pairs × 2 layers)
  - Updated `run_21x21_cka.sh` - excludes seed3 configs (runs 21x21 matrix only)
- **Pattern:** Each script uses `if [[ "$config" == *"seedX"* ]]` to filter configs

### 3. Expanded to 28x28 Matrix (4 Seeds Total)
- **Scope:** Original (seed 42) + seed1 + seed2 + seed3 = 28 models (7 tasks × 4 seeds)
- **Infrastructure Created:**
  - Seed3 representation extraction configs (28 configs in `configs/revision/exp4/repr_extraction_higher/seed3/`)
  - Seed3 representation extraction script (`extract_seed3_representations.sh`)
  - Seed3 PCA timeline configs (7 configs in `configs/revision/exp4/pca_timeline/seed3/`)
  - Seed3 PCA timeline script (`pca_timeline_seed3_all.sh`)
  - Seed3 CKA configs (350 configs for all seed3 comparisons)
- **Total CKA Infrastructure:** 833 configs total (21x21 matrix + seed3 comparisons)

### 4. Added Layer 6 CKA Analysis
- **Request:** "can you make cka calc and plot for layer 6 as well?"
- **Implementation:**
  - Updated `generate_cka_configs_cross_seed.py` default layers to `[4, 5, 6]` instead of `[3, 4, 5, 6]`
  - Updated `generate_cka_configs_seed3.py` to generate layers 4, 5, 6
  - Created `generate_cka_configs_21x21_layer6.py` for 21x21 matrix layer 6
  - Generated 231 layer 6 configs for 21x21 matrix
  - Generated 525 layer 6 configs for seed3 (175 pairs × 3 layers)
  - Created `plot_21x21_cka_matrix_l6.py` (adapted from layer 5 version)
  - Created `plot_21x21_cka_l6.sh` wrapper script
  - Created `run_21x21_cka_l6.sh` execution script (231 configs, excludes seed3)
- **Bug Fix:** Initial generation used wrong path for original experiments
  - Problem: Used `data/experiments/revision/exp4/pt1-X/` for all models
  - Correct: Originals are in `data/experiments/pt1-X/`, seeds are in `data/experiments/revision/exp4/pt1-X_seedY/`
  - Fixed with conditional path logic based on seed suffix

### 5. Created PT1-5 Additional Seeds (4-7)
- **Request:** "just for pt1-5 i want seed 4,5,6,7"
- **Rationale:** Inside task showing interesting behavior, want more seeds for robustness
- **Created:**
  - 4 training configs: `configs/revision/exp4/pt1_single_task_seed/pt1-5/pt1-5_seed{4,5,6,7}.yaml`
  - 4 training scripts: `scripts/revision/exp4/pt1_single_task_seed/pt1-5/pt1-5_seed{4,5,6,7}.sh`
  - All configs identical to seed3 but with different seed values (4, 5, 6, 7)

### 6. Organized PT3 (Three-Task) Experiments in Exp2
- **Context:** PT3 is three-task pretraining (8 combinations), part of exp2 seed robustness
- **Problem:** Configs/scripts were mistakenly created in exp5, needed to be in exp2
- **PT3 Task Combinations:**
  1. pt3-1: distance + trianglearea + angle
  2. pt3-2: compass + inside + perimeter
  3. pt3-3: crossing + distance + trianglearea
  4. pt3-4: angle + compass + inside
  5. pt3-5: perimeter + crossing + distance
  6. pt3-6: trianglearea + angle + compass
  7. pt3-7: inside + perimeter + crossing
  8. pt3-8: distance + trianglearea + compass
- **Implementation:**
  - Created `generate_pt3_seed_configs.py` to generate 16 configs (8 variants × 2 seeds)
  - Created `generate_pt3_seed_scripts.py` to generate 16 training scripts
  - Created `generate_pt3_meta_scripts_exp2.py` to generate 8 meta scripts (`run_pt3-X_all.sh`)
  - Moved everything from exp5 to exp2
  - Fixed paths in all configs (exp5 → exp2) and scripts
  - Fixed double nesting issue (`pt3_seed/pt3_seed/` → `pt3_seed/`)
- **Final Structure:**
  - `configs/revision/exp2/pt3_seed/pt3-{1-8}/pt3-{1-8}_seed{1,2}.yaml`
  - `scripts/revision/exp2/pt3_seed/pt3-{1-8}/pt3-{1-8}_seed{1,2}.sh`
  - `scripts/revision/exp2/run_pt3-{1-8}_all.sh` (meta scripts matching pt2 pattern)

### 7. Exp4 Meta Scripts Debate (Reverted)
- **Initial Attempt:** Tried to add representation extraction and PCA to exp4 meta scripts
- **Problem:** Extraction/PCA scripts run ALL tasks, not individual ones
- **User Feedback:** "WTF?????? [...] revert bullshit you did for exp4"
- **Resolution:** Reverted exp4 `run_pt1-X_all.sh` scripts to simple training-only version
- **Lesson:** Exp4 infrastructure is different from exp2 (whole-pipeline extraction vs individual)

## Key Technical Details

### CKA Config Format (Correct)
```yaml
center_kernels: true
checkpoint_steps: [328146]
city_filter: "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$"
exp1:
  name: pt1-1_seed2
  repr_dir: data/experiments/revision/exp4/pt1-1_seed2/analysis_higher/distance_firstcity_last_and_trans_l5/representations
  task: distance
exp2:
  name: pt1-2_seed1
  repr_dir: data/experiments/revision/exp4/pt1-2_seed1/analysis_higher/trianglearea_firstcity_last_and_trans_l5/representations
  task: trianglearea
kernel_type: linear
layer: 5
output_dir: /data/experiments/revision/exp4/cka_analysis/pt1-1_seed2_vs_pt1-2_seed1/layer5
save_timeline_plot: false
use_gpu: true
```

### Path Logic for Original vs Seed Experiments
```python
# Originals (no seed suffix) are in data/experiments/
base_path1 = f"data/experiments/revision/exp4/{model1_name}" if seed1 else f"data/experiments/{model1_name}"
base_path2 = f"data/experiments/revision/exp4/{model2_name}" if seed2 else f"data/experiments/{model2_name}"
```

### Seed-Specific CKA Filtering
```bash
# Run only seed2 configs
for config in configs/revision/exp4/cka_cross_seed/*/layer5.yaml; do
    if [[ "$config" == *"seed2"* ]]; then
        uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
    fi
done
```

### Files Created/Modified

**CKA Infrastructure:**
- `src/scripts/generate_cka_configs_seed3.py` - Generate seed3 CKA configs (updated to 3 layers)
- `src/scripts/generate_cka_configs_21x21_layer6.py` - Generate layer 6 configs for 21x21
- `scripts/revision/exp4/cka_analysis/run_seed2_cka.sh` - Run only seed2 CKAs
- `scripts/revision/exp4/cka_analysis/run_seed3_cka.sh` - Run only seed3 CKAs
- `scripts/revision/exp4/cka_analysis/run_21x21_cka_l6.sh` - Run layer 6 CKAs
- `scripts/revision/exp4/cka_analysis/plot_21x21_cka_l6.sh` - Plot layer 6 matrix
- `src/analysis/cka_v2/visualization/plot_21x21_cka_matrix_l6.py` - Layer 6 visualization

**PT1-5 Additional Seeds:**
- `configs/revision/exp4/pt1_single_task_seed/pt1-5/pt1-5_seed{4,5,6,7}.yaml`
- `scripts/revision/exp4/pt1_single_task_seed/pt1-5/pt1-5_seed{4,5,6,7}.sh`

**PT3 (Exp2):**
- `src/scripts/generate_pt3_seed_configs.py`
- `src/scripts/generate_pt3_seed_scripts.py`
- `src/scripts/generate_pt3_meta_scripts_exp2.py`
- `configs/revision/exp2/pt3_seed/` (16 configs)
- `scripts/revision/exp2/pt3_seed/` (16 scripts)
- `scripts/revision/exp2/run_pt3-{1-8}_all.sh` (8 meta scripts)

**Seed3 Infrastructure:**
- `configs/revision/exp4/repr_extraction_higher/seed3/` (28 configs)
- `scripts/revision/exp4/representation_extraction/extract_seed3_representations.sh`
- `configs/revision/exp4/pca_timeline/seed3/` (7 configs)
- `scripts/revision/exp4/pca_timeline/pca_timeline_seed3_all.sh`

## Config/Script Counts

**Exp4 CKA Infrastructure:**
- 21x21 matrix (orig+seed1+seed2): 462 configs (231 pairs × 2 layers) → now 693 configs (231 pairs × 3 layers with layer 6)
- Seed3 comparisons: 350 configs (175 pairs × 2 layers) → now 525 configs (175 pairs × 3 layers)
- **Total:** 1,218 CKA configs across all seeds and layers

**Exp2 PT3:**
- Training configs: 16 (8 variants × 2 seeds)
- Training scripts: 16 individual + 8 meta scripts

**PT1-5 Seeds:**
- 4 additional seed configs and scripts (seeds 4-7)

## Important Notes

- **28x28 Matrix Ready:** Full infrastructure for 4-seed analysis (original, seed1, seed2, seed3)
- **Layer 6 Added:** All CKA configs now include layers 4, 5, and 6
- **Seed-Specific Scripts:** Can run CKA for specific seeds without recalculating existing results
- **PT3 in Exp2:** Three-task experiments properly organized alongside two-task (pt2)
- **Path Consistency:** Fixed bug where original experiments used wrong base path
- **Config Format:** All CKA configs now use correct nested exp1/exp2 structure

## Next Steps (User Direction Needed)

1. Run seed3 training when ready
2. Extract representations for seed3
3. Run seed3 CKA analysis (525 configs)
4. Plot 28x28 CKA matrix
5. Decide on PT3 analysis scope (currently set up for exp2)
6. Run PT1-5 additional seeds (4-7) when needed

## Research Context Updates Needed

- Added PT3 (three-task) as part of exp2 seed robustness
- Expanded exp4 from 21x21 to 28x28 matrix capability (4 seeds)
- Added layer 6 analysis for all CKA comparisons
- PT1-5 (inside task) getting extra attention with 8 total seeds (orig + 1-7)
