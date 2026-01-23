# Exp3 Evaluation Infrastructure Setup - 2025-11-21

## Summary
Created complete evaluation and plotting infrastructure for revision/exp3, following the same pattern as exp1.

## What is Exp3?

**Research Question**: Does model architecture (wide vs narrow) affect fine-tuning performance when compute is held constant?

**Model Variants**:
- **Wide**: 2× width (256 hidden, 1024 intermediate, 8 heads), ½ epochs (3)
- **Narrow**: ½ width (64 hidden, 256 intermediate, 2 heads), 2× epochs (12)

**Models Trained**:
- Base models: `pt1_wide`, `pt1_narrow`
- FTWB1: 7 single-task fine-tuned models per architecture (14 total)
- FTWB2: 6 two-task fine-tuned models per architecture (12 total)
  - Experiments: 2, 4, 9, 12, 13, 15 (subset of all 21 combinations)

**Total**: 28 models

## Infrastructure Created

### 1. Configuration Generators

**`src/scripts/generate_revision_exp3_eval_configs.py`**
- Creates evaluation configs for all 28 models
- 15 evaluations per model (7 atlantis + 7 normal + 1 multi-task)
- Total: 420 configs
- Organization:
  - `configs/revision/exp3/eval/wide/base/`
  - `configs/revision/exp3/eval/wide/ftwb1-{1-7}/`
  - `configs/revision/exp3/eval/wide/ftwb2-{2,4,9,12,13,15}/`
  - `configs/revision/exp3/eval/narrow/base/`
  - `configs/revision/exp3/eval/narrow/ftwb1-{1-7}/`
  - `configs/revision/exp3/eval/narrow/ftwb2-{2,4,9,12,13,15}/`

### 2. Execution Script Generators

**`src/scripts/generate_revision_exp3_eval_scripts.py`**
- Creates 6 batch scripts for running evaluations
- Scripts:
  - `scripts/revision/exp3/eval/eval_wide_base_ftwb1.sh` (120 evals)
  - `scripts/revision/exp3/eval/eval_wide_ftwb2.sh` (90 evals)
  - `scripts/revision/exp3/eval/eval_wide_all.sh` (master)
  - `scripts/revision/exp3/eval/eval_narrow_base_ftwb1.sh` (120 evals)
  - `scripts/revision/exp3/eval/eval_narrow_ftwb2.sh` (90 evals)
  - `scripts/revision/exp3/eval/eval_narrow_all.sh` (master)

### 3. Plotting Scripts

**`src/scripts/plot_revision_exp3_ftwb1_heatmaps.py`**
- Creates 7×7 generalization matrices for FTWB1 models
- One plot per architecture (wide, narrow)
- Shows trained task vs eval task performance
- Normalized metrics: 0 = atlantis baseline, 1 = standard baseline

**`src/scripts/plot_revision_exp3_ftwb2_heatmaps.py`**
- Creates 7×6 performance matrices for FTWB2 models
- One plot per architecture (wide, narrow)
- Shows 7 tasks × 6 experiments
- Marks training tasks with 'T'

**`src/scripts/plot_revision_exp3_ftwb2_vs_ftwb1.py`**
- Creates difference plots: Actual - Expected
- Expected = max performance from FTWB1 single-task specialists
- Shows where two-task training beats single-task training
- One plot per architecture (wide, narrow)

## Normalization Methodology

Follows exp1's approach:

**For error metrics** (distance, trianglearea, angle, perimeter):
- Lower is better
- Uses log-ratio: `log(baseline_atlantis / value) / log(baseline_atlantis / baseline_standard)`

**For accuracy metrics** (crossing, inside, compass):
- Higher is better
- Uses linear: `(value - baseline_atlantis) / (baseline_standard - baseline_atlantis)`

**Scale**:
- 0.0 = no improvement over atlantis baseline
- 1.0 = reaches standard baseline (fully trained)
- Values can exceed 1.0 if better than baseline

## Usage

### Step 1: Run Evaluations

```bash
# Run all wide model evaluations
bash scripts/revision/exp3/eval/eval_wide_all.sh

# Run all narrow model evaluations
bash scripts/revision/exp3/eval/eval_narrow_all.sh

# Or run in parts:
bash scripts/revision/exp3/eval/eval_wide_base_ftwb1.sh
bash scripts/revision/exp3/eval/eval_wide_ftwb2.sh
bash scripts/revision/exp3/eval/eval_narrow_base_ftwb1.sh
bash scripts/revision/exp3/eval/eval_narrow_ftwb2.sh
```

**Runtime estimate**: ~6-8 hours for all 420 evaluations

### Step 2: Generate Plots

After evaluations complete:

```bash
# FTWB1 heatmaps (7×7 matrices)
uv run python src/scripts/plot_revision_exp3_ftwb1_heatmaps.py

# FTWB2 heatmaps (7×6 matrices)
uv run python src/scripts/plot_revision_exp3_ftwb2_heatmaps.py

# FTWB2 vs FTWB1 difference plots
uv run python src/scripts/plot_revision_exp3_ftwb2_vs_ftwb1.py
```

**Output location**: `data/experiments/revision/exp3/plots/`

**Generated plots**:
- `wide_ftwb1_evaluation_heatmap.png`
- `narrow_ftwb1_evaluation_heatmap.png`
- `wide_ftwb2_evaluation_heatmap.png`
- `narrow_ftwb2_evaluation_heatmap.png`
- `wide_ftwb2_vs_ftwb1.png`
- `narrow_ftwb2_vs_ftwb1.png`

## Expected Analysis

The plots will enable comparison of:

1. **Architecture Effects**:
   - Do wide models generalize better than narrow models?
   - Are there task-specific differences in architecture preference?

2. **Fine-tuning Performance**:
   - How do FTWB1 single-task models perform on transfer tasks?
   - Do wide/narrow models show different transfer patterns?

3. **Multi-task Synergy**:
   - Does two-task training (FTWB2) beat single-task specialists?
   - Are synergies different for wide vs narrow architectures?

4. **Compute Efficiency**:
   - With matched compute (wide: fewer epochs, narrow: more epochs), which architecture is more efficient?

## Files Created

**Generator scripts** (2):
- `src/scripts/generate_revision_exp3_eval_configs.py`
- `src/scripts/generate_revision_exp3_eval_scripts.py`

**Plotting scripts** (3):
- `src/scripts/plot_revision_exp3_ftwb1_heatmaps.py`
- `src/scripts/plot_revision_exp3_ftwb2_heatmaps.py`
- `src/scripts/plot_revision_exp3_ftwb2_vs_ftwb1.py`

**Configs** (420):
- `configs/revision/exp3/eval/{wide,narrow}/{base,ftwb1-{1-7},ftwb2-{2,4,9,12,13,15}}/*.yaml`

**Execution scripts** (6):
- `scripts/revision/exp3/eval/eval_{wide,narrow}_{base_ftwb1,ftwb2,all}.sh`

## Next Steps

1. **Run evaluations** (both wide and narrow)
2. **Generate plots** after evaluations complete
3. **Analyze results** to answer research question
4. **Compare with exp1** to understand seed vs architecture effects
5. **Write rebuttal section** on architecture ablation

## Status

- ✅ Infrastructure complete (configs, scripts, plotting)
- ⏳ Evaluations pending (420 total)
- ⏳ Plots pending (after evaluations)
- ⏳ Analysis pending (after plots)

---

**Date**: 2025-11-21
**Context**: Rebuttal phase preparation for exp3 (model width ablation)
