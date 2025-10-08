# CKA Timeline Analysis - Complete Data Collection and Visualization
**Date**: 2025-09-24 14:02
**Session**: CKA analysis completion with timeline visualizations

## Summary
Completed comprehensive CKA (Centered Kernel Alignment) analysis for all experiment pairs (pt1, pt2, pt3) across all layers (3, 4, 5, 6), fixing incomplete data issues and creating timeline visualizations for non-overlapping training set pairs.

## Major Tasks Completed

### 1. CKA Data Organization and Collection
- Created `/n/home12/cfpark00/WM_1/scripts/analysis/collect_all_cka_data.py` to aggregate all CKA results
- Generated three output files in `/n/home12/cfpark00/WM_1/scratch/cka_analysis_clean/`:
  - `cka_summary.csv`: 308 pair-layer combinations with summary statistics
  - `cka_checkpoints.csv`: 12,628 checkpoint measurements (847KB)
  - `cka_organized.json`: Hierarchical organization with special statistics

### 2. Fixed Incomplete Layer 3 Data
**Problem**: pt2 layer 3 CKA computation had only 4-7 checkpoints instead of 41
**Root Cause**: Representation extraction didn't create the expected `representations/` folder structure

**Solution**:
- Created proper configs with `save_repr_ckpts: [-2]` (saves all checkpoints)
- Re-ran representation extraction for pt2 layer 3
- Successfully obtained all 41 checkpoints for complete analysis

**Scripts Created**:
- `/scripts/analysis/create_all_missing_configs.sh`: Generate configs for missing layers
- `/scripts/analysis/run_all_missing_representations.sh`: Extract all missing representations

### 3. Timeline Visualization Implementation
Created `/scripts/analysis/plot_cka_timelines_non_overlapping.py` that:
- Plots CKA evolution over training steps (log scale x-axis)
- Filters for non-overlapping training set pairs only
- Creates 12 individual plots + 1 overview grid
- Saved in `/scratch/cka_analysis_clean/timelines_non_overlap/`

### 4. Key Findings

#### Training Data Overlap Structure:
- **pt1**: 7 models, each trained on 1 distinct task → All 21 pairs non-overlapping
- **pt2**: 8 models, each trained on 2 tasks → 18/28 pairs non-overlapping
- **pt3**: 8 models, each trained on 3 tasks → 9/28 pairs non-overlapping

#### CKA Results for Non-Overlapping Pairs:
| Experiment | Layer | Mean CKA ± Std | # Pairs |
|------------|-------|----------------|---------|
| pt1 | 3 | 0.211 ± 0.118 | 21 |
| pt1 | 4 | 0.450 ± 0.309 | 21 |
| pt1 | 5 | 0.477 ± 0.346 | 21 |
| pt1 | 6 | 0.434 ± 0.337 | 21 |
| pt2 | 3 | 0.374 ± 0.183 | 18 |
| pt2 | 4 | 0.784 ± 0.069 | 18 |
| pt2 | 5 | 0.886 ± 0.045 | 18 |
| pt2 | 6 | 0.820 ± 0.080 | 18 |
| pt3 | 3 | 0.444 ± 0.049 | 9 |
| pt3 | 4 | 0.747 ± 0.137 | 9 |
| pt3 | 5 | 0.831 ± 0.069 | 9 |
| pt3 | 6 | 0.833 ± 0.069 | 9 |

**Key Insight**: Models trained on multiple tasks (pt2, pt3) develop much more similar representations in higher layers (0.75-0.89 CKA) compared to single-task models (pt1: 0.43-0.48 CKA), even when their training data doesn't overlap.

## File Structure Changes
Created new organization in `/scratch/cka_analysis_clean/`:
```
cka_analysis_clean/
├── cka_checkpoints.csv          # All checkpoint CKA values
├── cka_summary.csv              # Summary statistics
├── cka_organized.json           # Hierarchical organization
└── timelines_non_overlap/       # Timeline plots
    ├── cka_timelines_overview.png
    └── cka_timeline_{prefix}_l{layer}_non_overlap.png (12 files)
```

## Scripts and Tools Created
1. `collect_all_cka_data.py`: Comprehensive data aggregation
2. `plot_cka_timelines_non_overlapping.py`: Timeline visualization with log scale
3. `create_all_missing_configs.sh`: Config generation for missing layers
4. `run_all_missing_representations.sh`: Batch representation extraction
5. YAML configs for CKA computation in `/configs/analysis_cka_l{3,4,6}/`
6. Batch scripts in `/scripts/analysis/cka/` for each prefix-layer combination

## Technical Notes
- CKA computation uses the original `compute_cka_from_representations.py` script
- Representation extraction requires `save_repr_ckpts: [-2]` to save all checkpoints
- The `-2` value means "save all checkpoints" not "second to last"
- Layer indices: 3=early, 4-5=middle, 6=late transformer layers

## Issues Resolved
- Fixed incomplete checkpoint data for pt2 layer 3 (was 4-7, now 41 checkpoints)
- Corrected training_overlap calculation based on actual training sets, not probe tasks
- Organized all outputs in consistent directory structure

## Next Steps Possible
- Analyze CKA dynamics during early vs late training
- Compare overlapping vs non-overlapping pair distributions
- Investigate why multi-task models converge to similar representations
- Create publication-ready figures with statistical significance tests