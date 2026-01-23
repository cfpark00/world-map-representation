# Same-Task CKA Analysis and 21×21 Matrix Setup
**Date:** 2025-11-21 12:34

## Summary
Created same-task CKA analysis to show multi-task training increases representational alignment even for the same task. Fixed and set up infrastructure for computing full 21×21 PT2 CKA matrices at layers 3, 4, 6.

## 1. PT2 21×21 CKA Matrix Plotting (Layers 3-6)

### Context
User wanted 7×7 CKA plot for PT2-1 through PT2-7, but discovered full 21×21 matrix (7 variants × 3 seeds) only existed for layer 5.

### Work Done
- Created `src/scripts/plot_21x21_pt2_cka_matrix_l5.py` to plot PT2 21×21 CKA matrix at layer 5
- Initially had wrong directory (`cka_analysis_pt2` instead of `revision/exp2/cka_analysis_all`)
- Fixed after user correction
- Applied exp4 styling: no axis labels, adaptive text color (white if CKA < 0.6, black otherwise), gray grid lines
- Generates three plots:
  - 21×21 full matrix
  - 7×7 averaged across seeds (without SEM)
  - 7×7 averaged with SEM annotations
- Created bash script: `scripts/revision/exp2/plot_pt2_21x21_cka_l5.sh`

**Key styling features:**
- Three-slope colormap normalization (0.0-0.4 compressed, 0.4-0.6 medium, 0.6-1.0 expanded)
- Magma colormap
- No axis labels/ticks
- Font size 24 for values without SEM, 18 with SEM
- Thick spines (3.75) and tick marks

## 2. Same-Task CKA Trends Analysis

### Motivation
User wanted to demonstrate: "even for same tasks, multi-task nature increases CKA"

This isolates the effect of multi-task training by comparing **different seeds of the same task** (intra-task) rather than different tasks.

### Implementation

**Data Collection (`src/scripts/collect_same_task_cka_trends.py`):**
- Extracts off-diagonal entries from 3×3 seed blocks (orig, seed1, seed2)
- For each variant, gets 3 unique seed pairs: (orig, seed1), (orig, seed2), (seed1, seed2)
- PT1: 6 variants × 3 pairs = 18 values per layer (excludes pt1-7 crossing)
- PT2: 6 variants × 3 pairs = 18 values at layer 5 (excludes pt2-7 crossing)
- PT3: 6 variants × 3 pairs = 18 values at layer 5 (excludes pt3-7/8 crossing)
- Special handling for PT1-5 seed remapping (seed3→seed2, seed4→seed3)
- Marks variants containing crossing task with `is_crossing` flag
- Data sources:
  - PT1: `data/experiments/revision/exp4/cka_analysis/`
  - PT2/PT3: `data/experiments/revision/exp2/cka_analysis_all/`

**Plotting (`src/scripts/plot_same_task_cka_trends.py`):**
- Initially included crossing task with dotted red line
- User clarification: wanted **cross-task** (different tasks), not crossing task
- Updated to load both:
  - Same-task data (from `same_task_cka_summary.csv`)
  - Cross-task data (from `cka_summary.csv`, filtered to non-overlapping)
- Generates two versions:
  1. `same_task_cka_trends.png` - Same-task only (solid green line)
  2. `same_task_cka_trends_with_cross.png` - Same-task (solid green) + cross-task (dotted gray)
- No legend, no y-axis label (user request)
- Dimmed individual points (alpha=0.15) to make lines more visible

**Results (Layer 5):**
- Same-task CKA: PT1=0.786, PT2=0.912, PT3=0.906
- Cross-task CKA: PT1=0.655, PT2=0.873, PT3=0.851
- **Key finding:** Same-task CKA > Cross-task CKA, showing multi-task training increases alignment even for identical tasks

**Scripts created:**
- `scripts/revision/exp2/collect_same_task_cka_trends.sh`
- `scripts/revision/exp2/plot_same_task_cka_trends.sh`

## 3. General CKA Trends Plot Improvements

Updated `src/scripts/plot_cka_trends.py` to dim individual data points (alpha=0.15 instead of 0.3) for better line visibility.

## 4. Setting Up Full 21×21 PT2 CKA Computation (Layers 3, 4, 6)

### Problem Discovery
- Full 21×21 matrix only existed for layer 5
- Layers 3, 4, 6 only had same-seed comparisons in `cka_analysis_same_seed/`
- Need full 21×21 for all layers

### Investigation
- Configs already exist: 210 configs per layer in `configs/revision/exp2/cka_analysis_all/`
- Run scripts existed but called wrong file: `src/analysis/cka_v2/analyze_cka.py` (doesn't exist)
- Should call: `src/scripts/analyze_cka_pair.py`

### Fixes Applied

**1. Fixed config generator (`generate_exp2_pt2_all_pairs_cka_configs.py`):**
- Changed from `repr_path` (single file) to `repr_dir` (checkpoint directory)
- Added missing fields to match layer 5 format:
  - `task`, `layer`, `checkpoint_steps`, `city_filter`, `kernel_type`, `center_kernels`, `use_gpu`, `save_timeline_plot`
- Uses analysis_higher representation directories

**2. Fixed run script generator (`generate_exp2_pt2_all_pairs_run_scripts.py`):**
- Changed to call `src/scripts/analyze_cka_pair.py` (not `src/analysis/cka_v2/analyze_cka.py`)
- Added `--overwrite` flag to all commands

**3. Created 4 meta execution scripts:**
```bash
scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk1.sh  # Layer 3 (210 jobs)
scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk2.sh  # Layer 4 (210 jobs)
scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk3.sh  # Layer 6 (210 jobs)
scripts/revision/exp2/cka_analysis_all/check_pt2_21x21_l346_status.sh  # Status monitor
```

### Regeneration Commands
```bash
uv run python src/scripts/generate_exp2_pt2_all_pairs_cka_configs.py
uv run python src/scripts/generate_exp2_pt2_all_pairs_run_scripts.py
```

### Execution
User started running all chunks in parallel (630 total CKA computations).

## Files Created/Modified

**New files:**
- `src/scripts/collect_same_task_cka_trends.py`
- `src/scripts/plot_same_task_cka_trends.py`
- `src/scripts/plot_21x21_pt2_cka_matrix_l5.py`
- `scripts/revision/exp2/collect_same_task_cka_trends.sh`
- `scripts/revision/exp2/plot_same_task_cka_trends.sh`
- `scripts/revision/exp2/plot_pt2_21x21_cka_l5.sh`
- `scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk1.sh`
- `scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk2.sh`
- `scripts/revision/exp2/cka_analysis_all/run_pt2_21x21_l346_chunk3.sh`
- `scripts/revision/exp2/cka_analysis_all/check_pt2_21x21_l346_status.sh`
- `data/experiments/revision/exp2/cka_trends/same_task_cka_summary.csv`
- `data/experiments/revision/exp2/cka_trends/same_task_cka_trends.png`
- `data/experiments/revision/exp2/cka_trends/same_task_cka_trends_with_cross.png`
- `data/experiments/revision/exp2/cka_analysis_all/pt2_cka_21x21_l5.png`
- `data/experiments/revision/exp2/cka_analysis_all/pt2_cka_7x7_averaged_l5.png`
- `data/experiments/revision/exp2/cka_analysis_all/pt2_cka_7x7_averaged_with_sem_l5.png`

**Modified files:**
- `src/scripts/plot_cka_trends.py` - Dimmed points (alpha=0.15)
- `src/scripts/generate_exp2_pt2_all_pairs_cka_configs.py` - Fixed config format
- `src/scripts/generate_exp2_pt2_all_pairs_run_scripts.py` - Fixed script paths, added --overwrite
- All scripts in `scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_l*.sh` - Regenerated with correct paths

## Key Insights

1. **Multi-task training increases representational alignment for same tasks**: The same-task CKA analysis provides strong evidence that multi-task training causes convergence even when comparing different random seeds of the same task.

2. **Complete layer coverage needed**: Having full 21×21 matrices at all layers (3, 4, 5, 6) will enable comprehensive analysis of how representational similarity changes across network depth.

3. **Infrastructure robustness**: The existing config/script generation infrastructure is solid but requires careful attention to config format compatibility between generators and analysis scripts.
