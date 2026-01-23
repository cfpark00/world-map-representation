# Exp3 Evaluation and Plotting Infrastructure - 2025-11-21 01:26

## Session Summary
Created complete evaluation and plotting infrastructure for revision/exp3 (model width ablation study), following exp1's pattern. Successfully ran wide model evaluations and generated all 3 analysis plots.

## Background: What is Exp3?

**Research Question**: Does model architecture (wide vs narrow) affect fine-tuning performance when compute is held constant?

**Model Variants**:
- **Wide**: hidden=256, intermediate=1024, heads=8, epochs=3
- **Narrow**: hidden=64, intermediate=256, heads=2, epochs=12
- **Wider** (NEW): hidden=512, intermediate=2048, heads=16, epochs=3

**Models Trained** (28 total):
- Base: pt1_wide, pt1_narrow
- FTWB1: 7 single-task fine-tuned per architecture (14 total)
- FTWB2: 6 two-task fine-tuned per architecture (12 total)
  - Experiments: 2, 4, 9, 12, 13, 15 (subset of 21)

## Tasks Completed

### 1. Understanding Exp1 Evaluation Pipeline

**Reviewed exp1's approach:**
- Evaluates all models on 7 tasks (atlantis + normal versions)
- Calculates normalized metrics (0 = atlantis baseline, 1 = standard baseline)
- Uses log-ratio for error metrics, linear for accuracy metrics
- Creates 3 types of plots:
  - FTWB1: 7×7 generalization matrices
  - FTWB2: 21×7 (or 6×7 for exp3) performance matrices
  - FTWB2 vs FTWB1: Actual - Expected difference plots

### 2. Created Evaluation Infrastructure

**Generator Scripts:**
- `src/scripts/generate_revision_exp3_eval_configs.py`
  - Creates 420 evaluation configs (28 models × 15 evals each)
  - Organization: `configs/revision/exp3/eval/{wide,narrow}/`

- `src/scripts/generate_revision_exp3_eval_scripts.py`
  - Creates 16 execution scripts (7 chunks per model type + 2 masters)
  - **Chunking strategy**: ~30 evals (2 models) per script
  - Enables parallel execution across compute nodes

**Generated Configs**: 420 YAML files
- Wide: 210 configs (base + 7 FTWB1 + 6 FTWB2)
- Narrow: 210 configs (same)

**Generated Scripts**: 16 bash scripts
- Wide: 7 chunked scripts + 1 master
- Narrow: 7 chunked scripts + 1 master

### 3. Created Plotting Scripts

**Three plotting scripts created:**

1. **`src/scripts/plot_revision_exp3_ftwb1_heatmaps.py`**
   - 7×7 generalization matrices
   - Rows = trained task, Cols = eval task
   - Shows single-task fine-tuning transfer

2. **`src/scripts/plot_revision_exp3_ftwb2_heatmaps.py`**
   - 7×6 performance matrices (6 FTWB2 experiments)
   - Marks training tasks with 'T'
   - Shows two-task fine-tuning performance

3. **`src/scripts/plot_revision_exp3_ftwb2_vs_ftwb1.py`**
   - Actual - Expected difference plots
   - Expected = max(FTWB1 single-task specialists)
   - Shows synergy/interference from multi-task training

All scripts include:
- Safety checks (skip if evaluations don't exist)
- Same normalization as exp1 (log-ratio/linear)
- Statistical summaries (trained vs transfer)

### 4. Ran Wide Model Evaluations

User confirmed all 7 wide evaluation scripts completed:
```bash
bash scripts/revision/exp3/eval/eval_wide_base_ftwb1-1.sh
bash scripts/revision/exp3/eval/eval_wide_ftwb1-2_ftwb1-3.sh
bash scripts/revision/exp3/eval/eval_wide_ftwb1-4_ftwb1-5.sh
bash scripts/revision/exp3/eval/eval_wide_ftwb1-6_ftwb1-7.sh
bash scripts/revision/exp3/eval/eval_wide_ftwb2-2_ftwb2-4.sh
bash scripts/revision/exp3/eval/eval_wide_ftwb2-9_ftwb2-12.sh
bash scripts/revision/exp3/eval/eval_wide_ftwb2-13_ftwb2-15.sh
```

### 5. Generated Wide Model Plots

Successfully created all 3 plots for wide model:

**Plot 1: FTWB1 Generalization Matrix (7×7)**
- Trained task (diagonal): 0.803 ± 0.146
- Transfer (off-diagonal): 0.491 ± 0.267
- File: `wide_ftwb1_evaluation_heatmap.png`

**Plot 2: FTWB2 Performance Matrix (7×6)**
- Trained tasks: 0.856 ± 0.186
- Transfer tasks: 0.496 ± 0.274
- File: `wide_ftwb2_evaluation_heatmap.png`

**Plot 3: FTWB2 - FTWB1 Difference**
- Trained tasks diff: +0.042 ± 0.186 (slightly better)
- Transfer tasks diff: -0.056 ± 0.219 (slightly worse)
- Overall diff: -0.028
- File: `wide_ftwb2_vs_ftwb1.png`

**Key Finding**: Two-task training (FTWB2) performs approximately the same as single-task specialists (FTWB1), with slight advantages on trained tasks but slight disadvantages on transfer.

### 6. Created pt1_wider Configuration

At user request, created an even wider model variant:
- **pt1_wider**: hidden=512, intermediate=2048, heads=16, epochs=3
- Config: `configs/revision/exp3/pt1_wider/pt1_wider.yaml`
- Script: `scripts/revision/exp3/pt1_wider/pt1_wider.sh`
- Purpose: Test even larger width scaling

## Files Created

**Generator Scripts** (2):
- `src/scripts/generate_revision_exp3_eval_configs.py`
- `src/scripts/generate_revision_exp3_eval_scripts.py`

**Plotting Scripts** (3):
- `src/scripts/plot_revision_exp3_ftwb1_heatmaps.py`
- `src/scripts/plot_revision_exp3_ftwb2_heatmaps.py`
- `src/scripts/plot_revision_exp3_ftwb2_vs_ftwb1.py`

**Configs** (420 + 1):
- `configs/revision/exp3/eval/{wide,narrow}/{base,ftwb1-*,ftwb2-*}/*.yaml`
- `configs/revision/exp3/pt1_wider/pt1_wider.yaml`

**Scripts** (16 + 1):
- `scripts/revision/exp3/eval/eval_{wide,narrow}_*.sh`
- `scripts/revision/exp3/pt1_wider/pt1_wider.sh`

**Plots** (3):
- `data/experiments/revision/exp3/plots/wide_ftwb1_evaluation_heatmap.png`
- `data/experiments/revision/exp3/plots/wide_ftwb2_evaluation_heatmap.png`
- `data/experiments/revision/exp3/plots/wide_ftwb2_vs_ftwb1.png`

**Documentation** (1):
- `docs/logs/2025-11-21/exp3_evaluation_infrastructure.md`

## Key Decisions

1. **Chunking Strategy**: Broke evaluations into ~30 evals (2 models) per script to enable parallel execution
2. **Skip Narrow**: User decided to focus on wide model only for now
3. **Safety Checks**: Added existence checks to plotting scripts to skip models without evaluations
4. **Normalization**: Used same method as exp1 (log-ratio for errors, linear for accuracy)
5. **pt1_wider**: Added even wider model variant (4× baseline width) for additional scaling test

## Status

**Wide Model:**
- ✅ Training complete (base + 7 FTWB1 + 6 FTWB2)
- ✅ Evaluations complete (210 evals)
- ✅ Plots generated (3 plots)

**Narrow Model:**
- ✅ Training complete (base + 7 FTWB1)
- ❌ Evaluations not run
- ❌ Plots not generated
- Decision: Skipping for now

**Wider Model:**
- ✅ Config created
- ⏳ Training pending

## Next Steps

1. **Optional**: Run narrow evaluations and generate narrow plots
2. **Optional**: Train pt1_wider and create evaluation infrastructure
3. **Analysis**: Compare wide vs narrow (if narrow evals run) to answer research question
4. **Rebuttal**: Use exp3 plots for architecture ablation section

## Research Context

Exp3 addresses reviewer questions about model capacity effects:
- Does width vs depth matter for fine-tuning with fixed compute?
- Are generalization patterns architecture-dependent?
- What's the optimal architecture for this task domain?

The infrastructure enables systematic comparison across model widths while controlling for total compute budget.

---

**Session Duration**: ~1.5 hours
**Date**: 2025-11-21 01:26
**Context**: Rebuttal phase, exp3 model width ablation study
