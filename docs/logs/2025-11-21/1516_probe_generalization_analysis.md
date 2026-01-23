# 15:16 - Probe Generalization Analysis for OOD Evaluation

## Summary
Created a comprehensive linear probe generalization analysis to evaluate how well spatial representations transfer to out-of-distribution (OOD) Atlantis cities across all FTWB2 models and 4 training seeds.

## Tasks Completed

### 1. Probe Generalization Evaluation Script
- **Created**: `src/scripts/evaluate_probe_generalization.py`
- **Purpose**: Train linear probe on non-Atlantis cities to predict x,y coordinates, then test on Atlantis (OOD) and baseline (held-out non-Atlantis)
- **Key parameters**:
  - `n_train`: 4000 cities for training (enforces no Atlantis)
  - `n_baseline`: 100 held-out non-Atlantis cities for baseline comparison
  - `layer`: Configurable, defaults to layer 5
- **Outputs per model**:
  - `results.json`: Summary statistics (train/test R², distance errors)
  - `test_predictions.csv`: Individual Atlantis city predictions and errors
  - `baseline_predictions.csv`: Individual baseline city predictions and errors
  - `visualization.png`: PCA scatter with Atlantis predictions

### 2. Config Generation for All FTWB2 Models
- **Generated configs for 4 seeds × 21 FTWB2 models = 84 total**:
  - `configs/revision/exp1/probe_generalization/pt1_seed1_ftwb2-{1-21}.yaml`
  - `configs/revision/exp1/probe_generalization/original/pt1_ftwb2-{1-21}.yaml`
  - `configs/revision/exp1/probe_generalization/seed2/pt1_seed2_ftwb2-{1-21}.yaml`
  - `configs/revision/exp1/probe_generalization/seed3/pt1_seed3_ftwb2-{1-21}.yaml`
- **Task-specific repr_dir**: Each FTWB2 model uses different task prefix based on its training tasks
  - E.g., ftwb2-1 → distance, ftwb2-2 → angle, ftwb2-3 → inside, etc.

### 3. Execution Scripts
- Created run scripts for each seed:
  - `scripts/revision/exp1/probe_generalization/run_all_probe_gen.sh` (seed1)
  - `scripts/revision/exp1/probe_generalization/original/run_all_probe_gen.sh`
  - `scripts/revision/exp1/probe_generalization/seed2/run_all_probe_gen.sh`
  - `scripts/revision/exp1/probe_generalization/seed3/run_all_probe_gen.sh`

### 4. Histogram Plotting Script
- **Created**: `src/scripts/plot_probe_generalization_histogram.py`
- **Features**:
  - Pools ALL individual city errors from all 84 models (4 seeds × 21 FTWB2)
  - Splits Atlantis samples by whether model was trained WITH or WITHOUT distance task
  - Log-scale x-axis (xlim min = 2)
  - Mean vertical lines for each group
  - Mann-Whitney U test p-value in title
- **Output**: `data/experiments/revision/exp1/plots/probe_generalization_histogram.png`

### 5. Key Results (All 4 Seeds Pooled)

| Group | N Samples | Mean | Std | Median |
|-------|-----------|------|-----|--------|
| Baseline (non-Atlantis) | 8,400 | 18.6 | 16.2 | 14.0 |
| Atlantis (no distance task) | 5,220 | 107.8 | 104.9 | 77.6 |
| Atlantis (with distance task) | 2,088 | 508.3 | 288.3 | 476.7 |

**Statistical Test**: Mann-Whitney U between Atlantis groups: p ≈ 0 (extremely significant)

### 6. Key Findings
- **Distance task severely impairs probe generalization**: Models trained WITH distance task have ~5x worse probe generalization error to Atlantis (508 vs 108)
- **Baseline performance is excellent**: ~19 units error on held-out real cities (coordinate range: x=-1800 to 1800, y=-900 to 900)
- **OOD gap exists for all models**: Even best models (no distance task) show ~6x worse performance on Atlantis vs baseline
- **Consistent across seeds**: Results pooled from 4 independent training seeds show robust pattern

### 7. Technical Notes
- Initial negative R² confusion: Atlantis cities form a tight cluster, so predicting the true mean gives R²=0; any prediction spread gives negative R²
- Switched to L1 distance error as primary metric (more interpretable)
- Original seed uses different base path (`data/experiments/pt1_ftwb2-X/` vs `revision/exp1/`)

## Files Created

### Python Scripts:
- `src/scripts/evaluate_probe_generalization.py` - Main evaluation script
- `src/scripts/plot_probe_generalization_histogram.py` - Histogram visualization

### Configs (84 total):
- `configs/revision/exp1/probe_generalization/*.yaml` - seed1 (21 configs)
- `configs/revision/exp1/probe_generalization/original/*.yaml` - original (21 configs)
- `configs/revision/exp1/probe_generalization/seed2/*.yaml` - seed2 (21 configs)
- `configs/revision/exp1/probe_generalization/seed3/*.yaml` - seed3 (21 configs)

### Run Scripts (4):
- `scripts/revision/exp1/probe_generalization/run_all_probe_gen.sh`
- `scripts/revision/exp1/probe_generalization/original/run_all_probe_gen.sh`
- `scripts/revision/exp1/probe_generalization/seed2/run_all_probe_gen.sh`
- `scripts/revision/exp1/probe_generalization/seed3/run_all_probe_gen.sh`

### Plot Script:
- `scripts/revision/exp1/plots/plot_probe_generalization_histogram.sh`

## Output Location
- Results: `data/experiments/revision/exp1/pt1_seed{X}_ftwb2-{Y}/probe_generalization/atlantis/`
- Plots: `data/experiments/revision/exp1/plots/`
  - `probe_generalization_histogram.png`
  - `probe_generalization_summary.json`

## Research Implications
The finding that distance-task training impairs probe generalization to OOD cities suggests that:
1. Distance task may cause the model to overfit to training city relationships
2. The distance task representations may be less transferable/generalizable
3. Multi-task training composition significantly affects OOD behavior
