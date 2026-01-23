# Session Log: Complete Exp1 Infrastructure Setup

**Date:** 2025-11-20
**Time:** 18:01 EST
**Duration:** Full session
**Topic:** Complete evaluation, representation extraction, and FTWB1 infrastructure for revision/exp1

---

## Session Overview

Set up complete infrastructure for revision/exp1 experiments including evaluation (990 configs), representation extraction (87 configs), and discovered/added missing FTWB1 models (21 models) needed for computing expected generalization baseline.

---

## Major Accomplishments

### 1. Evaluation Infrastructure (CORRECTED)

**Initial Setup (INCOMPLETE):**
- Created 462 configs for atlantis tasks only

**Corrected Setup:**
- Discovered evaluations need ALL task types (not just atlantis)
- Regenerated to include: 7 atlantis + 7 normal + 1 multi-task per model
- **Total: 990 evaluation configs** (66 models × 15 tasks)

**Files Created:**
- Configs: `configs/revision/exp1/eval/seed{1,2,3}/{base,ftwb2-1..21}/`
  - Each model has 15 configs: atlantis_{task}.yaml, {task}.yaml, multi_task.yaml
- Scripts: `scripts/revision/exp1/eval/`
  - 9 batch scripts (3 seeds × 3 batches)
  - 2 master scripts (sequential and by-seed)
  - Individual scripts for each seed/batch combination

**Key Settings:**
- `checkpoints: last` - Only evaluate final checkpoint
- `save_full_results: false` - Aggregate metrics only
- `temperature: 0.0, do_sample: false, top_k: 1` - Deterministic greedy decoding

**Bug Fix:**
- Fixed `evaluate_checkpoints.py` KeyError when accessing `detailed_results`
- Now safely handles `save_full_results=False` case

### 2. Representation Extraction Infrastructure

**Setup:**
- 66 configs for base + ftwb2 models (1 config per model)
- Each extracts layer 5 representations from last checkpoint
- Uses first trained task for extraction prompt

**Files Created:**
- Configs: `configs/revision/exp1/representation_extraction/seed{1,2,3}/{base,ftwb2-1..21}/`
- Scripts: `scripts/revision/exp1/representation_extraction/`
  - 9 batch scripts
  - 2 master scripts (sequential and by-seed)

**Task Assignment:**
- Base models: distance (most common task)
- FTWB2 models: First task from training pair
  - Example: ftwb2-1 trained on [distance, trianglearea] → extracts using distance

### 3. FTWB1 Infrastructure (CRITICAL ADDITION)

**Why Needed:**
- Paper needs to compute: **Actual - Expected generalization**
- Expected = max performance from single-task specialists (FTWB1)
- Actual = performance from multi-task models (FTWB2)
- Without FTWB1: Can only show raw performance, not relative improvement

**What Was Created:**

#### Training (21 models = 3 seeds × 7 tasks)
- Configs: `configs/revision/exp1/training/seed{1,2,3}/ftwb1-{1..7}_{task}.yaml`
- Scripts:
  - `train_seed{1,2,3}_ftwb1_all.sh` (individual seeds)
  - `train_all_ftwb1_sequential.sh` (master)
- Training params: Same as FTWB2 (lr=1e-5, 30 epochs, etc.)

#### Evaluation (315 configs = 21 models × 15 tasks)
- Configs: `configs/revision/exp1/eval/seed{1,2,3}/ftwb1-{1..7}/`
- Script: `eval_all_ftwb1.sh`

#### Representation Extraction (21 configs)
- Configs: `configs/revision/exp1/representation_extraction/seed{1,2,3}/ftwb1-{1..7}/`
- Script: `extract_all_ftwb1.sh`

**Task Mapping:**
```
ftwb1-1: distance
ftwb1-2: trianglearea
ftwb1-3: angle
ftwb1-4: compass
ftwb1-5: inside
ftwb1-6: perimeter
ftwb1-7: crossing
```

### 4. Evaluation Heatmap Visualization

**Created:** `src/scripts/plot_revision_exp1_ftwb2_heatmaps.py`

**Generates:** 3 plots (one per seed) showing 21×7 heatmaps

**Output:** `data/experiments/revision/exp1/plots/seed{1,2,3}_ftwb2_evaluation_heatmap.png`

**Results Summary:**
- Seed 1: Trained=0.861±0.205, Transfer=0.548±0.299
- Seed 2: Trained=0.931±0.141, Transfer=0.640±0.304
- Seed 3: Trained=0.896±0.163, Transfer=0.632±0.282

**Normalized Improvement Metric:**
- 0.0 = atlantis baseline (untrained/zero-shot)
- 1.0 = standard baseline (fully trained)
- Scale shows how much model improved relative to maximum possible improvement

**Two Formulas:**
1. **Error metrics** (distance, trianglearea, angle, perimeter):
   - Uses log-ratio: `log(baseline_atlantis/value) / log(baseline_atlantis/baseline_standard)`
   - Captures multiplicative improvements (1000→100 = 100→10)

2. **Accuracy metrics** (crossing, inside, compass):
   - Uses linear: `(value - baseline_atlantis) / (baseline_standard - baseline_atlantis)`
   - Natural for bounded [0,1] metrics

---

## File Organization

### Generation Scripts Created
1. `src/scripts/generate_revision_exp1_eval_configs.py` - Eval configs (990)
2. `src/scripts/generate_revision_exp1_eval_scripts.py` - Eval batch scripts (9+2)
3. `src/scripts/generate_revision_exp1_repr_configs.py` - Repr configs (66)
4. `src/scripts/generate_revision_exp1_repr_scripts.py` - Repr batch scripts (9+2)
5. `src/scripts/generate_revision_exp1_ftwb1_configs.py` - FTWB1 training configs (21)
6. `src/scripts/generate_revision_exp1_ftwb1_train_scripts.py` - FTWB1 training scripts (4)
7. `src/scripts/generate_revision_exp1_ftwb1_eval_configs.py` - FTWB1 eval configs (315)
8. `src/scripts/generate_revision_exp1_ftwb1_repr_configs.py` - FTWB1 repr configs (21)
9. `src/scripts/plot_revision_exp1_ftwb2_heatmaps.py` - Visualization script

### Master Scripts
- `scripts/revision/exp1/eval/eval_all_sequential.sh` - All evals (990)
- `scripts/revision/exp1/eval/eval_all_ftwb1.sh` - FTWB1 evals (315)
- `scripts/revision/exp1/representation_extraction/extract_all_sequential.sh` - All reprs (66)
- `scripts/revision/exp1/representation_extraction/extract_all_ftwb1.sh` - FTWB1 reprs (21)
- `scripts/revision/exp1/training/train_all_ftwb1_sequential.sh` - FTWB1 training (21)

### Documentation Created
1. `docs/logs/2025-11-20/revision_exp1_eval_setup.md` - Eval infrastructure
2. `docs/logs/2025-11-20/revision_exp1_repr_setup.md` - Repr infrastructure
3. `docs/logs/2025-11-20/exp1_normalized_metrics_explanation.md` - Metric details
4. `docs/logs/2025-11-20/exp1_ftwb1_setup.md` - FTWB1 infrastructure
5. `scripts/revision/exp1/README.md` - Quick reference guide

---

## Complete Exp1 Model Inventory

| Model Type | Count | Status |
|------------|-------|--------|
| Base (pt1_seed{1,2,3}) | 3 | ✅ Trained, ✅ Evaluated |
| FTWB1 (single-task) | 21 | ⏳ **Needs Training** |
| FTWB2 (two-task) | 63 | ✅ Trained, ✅ Evaluated |
| **Total** | **87** | 66/87 complete |

---

## Total Infrastructure Created

### Configs Generated
- Evaluation: 1,305 configs (990 ftwb2 + 315 ftwb1)
- Representation: 87 configs (66 ftwb2 + 21 ftwb1)
- Training: 21 configs (ftwb1 only)
- **Total: 1,413 config files**

### Scripts Generated
- Evaluation: 11 scripts (9 batch + 2 master)
- Representation: 11 scripts (9 batch + 2 master)
- Training: 4 scripts (3 seed + 1 master)
- **Total: 26 executable scripts**

---

## Key Decisions & Rationale

### 1. Why Evaluate Both Atlantis and Normal Tasks?
- Original pt1 experiments evaluated both
- Atlantis = OOD generalization
- Normal = in-distribution performance
- Multi-task = combined evaluation
- Provides complete performance picture

### 2. Why Only Last Checkpoint?
- Different from original pt1 (which evaluated all checkpoints)
- Exp1 focuses on final performance, not training dynamics
- Reduces computational cost significantly
- Still allows full evaluation across all task types

### 3. Why Log-Ratio for Errors?
- Errors improve multiplicatively (1000 → 100 → 10)
- Log-ratio treats equal relative improvements equally
- Going from terrible→bad is as valuable as bad→good
- Prevents ceiling effects on already-good models

### 4. Why FTWB1 is Critical?
- Core paper contribution is about multi-task synergy
- Need baseline showing what specialists can achieve
- Expected generalization = max(ftwb1 performances)
- Actual - Expected = true multi-task benefit
- Without it: Can only show raw numbers, not relative improvement

---

## Next Steps (User Action Required)

### Immediate Priority
1. **Train FTWB1 models** (21 training runs)
   ```bash
   bash scripts/revision/exp1/training/train_all_ftwb1_sequential.sh
   ```

### After FTWB1 Training
2. **Evaluate FTWB1** (315 evaluations)
   ```bash
   bash scripts/revision/exp1/eval/eval_all_ftwb1.sh
   ```

3. **Extract FTWB1 representations** (21 extractions)
   ```bash
   bash scripts/revision/exp1/representation_extraction/extract_all_ftwb1.sh
   ```

### Analysis Phase
4. **Update plotting script** to include expected generalization baseline
5. **Generate difference plots** (Actual - Expected)
6. **Compute multi-task synergy metrics**

---

## Technical Notes

### Temperature Settings
- All evaluations use `temperature: 0.0`
- Deterministic greedy decoding
- Reproducible results
- Represents model's "best guess"

### Checkpoint Selection
- `save_repr_ckpts: [-2]` = last checkpoint
- Different from `-1` which would be most recent
- `-2` ensures we get the final trained checkpoint

### Dataset Paths
- Atlantis tasks: `data/datasets/{task}_100k_atlantis_required`
- Normal tasks: `data/datasets/{task}_1M_no_atlantis`
- Multi-task: `data/datasets/multitask_pt1`
- FTWB1 training: `data/datasets/ftwb1-{1..7}`

---

## Summary Statistics

**Time Investment:**
- Config generation: ~10 seconds per generator script
- Script generation: ~5 seconds per generator script
- Documentation: Comprehensive guides created
- Total setup: Complete infrastructure in single session

**Code Quality:**
- All scripts follow repo conventions (minimal bash, no comments)
- All configs validated with existing evaluation infrastructure
- Reusable generation scripts for future modifications
- Comprehensive documentation for reproducibility

**Completeness:**
- ✅ All evaluation infrastructure (1,305 configs)
- ✅ All representation extraction (87 configs)
- ✅ All training configs for FTWB1 (21 configs)
- ✅ Master scripts for easy execution
- ✅ Visualization pipeline (heatmaps generated)
- ✅ Bug fixes (evaluate_checkpoints.py)
- ⏳ FTWB1 training pending (user action)

---

## Files Modified/Created This Session

### New Directories
- `configs/revision/exp1/eval/seed{1,2,3}/{base,ftwb2-1..21,ftwb1-1..7}/`
- `configs/revision/exp1/representation_extraction/seed{1,2,3}/{base,ftwb2-1..21,ftwb1-1..7}/`
- `configs/revision/exp1/training/seed{1,2,3}/`
- `scripts/revision/exp1/eval/`
- `scripts/revision/exp1/representation_extraction/`
- `scripts/revision/exp1/training/`
- `data/experiments/revision/exp1/plots/`

### Key Files Modified
- `src/eval/evaluate_checkpoints.py` - Fixed detailed_results KeyError

### Documentation Created
- 5 markdown files in `docs/logs/2025-11-20/`
- 1 README in `scripts/revision/exp1/`

---

## Session End Notes

This session completed the full infrastructure for revision/exp1, ensuring all models can be properly evaluated and analyzed. The critical addition of FTWB1 models ensures the paper can demonstrate the core contribution: multi-task learning provides benefits beyond what a collection of single-task specialists can achieve.

All infrastructure is ready. Next session should focus on training FTWB1 models and running the complete analysis pipeline.
