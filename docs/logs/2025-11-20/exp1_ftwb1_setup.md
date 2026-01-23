# Revision Exp1 FTWB1 Setup

**Date:** 2025-11-20
**Task:** Create FTWB1 (single-task fine-tuning) infrastructure for computing expected generalization baseline

## Why FTWB1 is Needed

For the paper, we need to compute **actual generalization - expected generalization**.

**Expected generalization** = max performance from any single-task model (ftwb1) on a given task
**Actual generalization** = performance from multi-task model (ftwb2) on that task

This allows us to measure whether multi-task training provides benefits **beyond** what can be achieved by simply having multiple single-task specialists.

## What Was Created

### 1. Training Configs (21 total)

**Location:** `configs/revision/exp1/training/`

**Structure:**
```
configs/revision/exp1/training/
├── seed1/
│   ├── ftwb1-1_distance.yaml
│   ├── ftwb1-2_trianglearea.yaml
│   ├── ftwb1-3_angle.yaml
│   ├── ftwb1-4_compass.yaml
│   ├── ftwb1-5_inside.yaml
│   ├── ftwb1-6_perimeter.yaml
│   └── ftwb1-7_crossing.yaml
├── seed2/
│   └── [same structure]
└── seed3/
    └── [same structure]
```

**Task Mapping:**
| FTWB1 # | Task | Dataset |
|---------|------|---------|
| 1 | distance | data/datasets/ftwb1-1 |
| 2 | trianglearea | data/datasets/ftwb1-2 |
| 3 | angle | data/datasets/ftwb1-3 |
| 4 | compass | data/datasets/ftwb1-4 |
| 5 | inside | data/datasets/ftwb1-5 |
| 6 | perimeter | data/datasets/ftwb1-6 |
| 7 | crossing | data/datasets/ftwb1-7 |

### 2. Training Scripts (4 total)

**Location:** `scripts/revision/exp1/training/`

```
- train_seed1_ftwb1_all.sh (7 models)
- train_seed2_ftwb1_all.sh (7 models)
- train_seed3_ftwb1_all.sh (7 models)
- train_all_ftwb1_sequential.sh (master script)
```

### 3. Evaluation Configs (315 total)

**Location:** `configs/revision/exp1/eval/seed{1,2,3}/ftwb1-{1..7}/`

- 15 configs per model (7 atlantis + 7 normal + 1 multi)
- 21 models × 15 = 315 configs

**Evaluation Script:**
- `scripts/revision/exp1/eval/eval_all_ftwb1.sh` (315 evaluations)

### 4. Representation Extraction Configs (21 total)

**Location:** `configs/revision/exp1/representation_extraction/seed{1,2,3}/ftwb1-{1..7}/`

- 1 config per model (extracts using trained task)

**Extraction Script:**
- `scripts/revision/exp1/representation_extraction/extract_all_ftwb1.sh` (21 extractions)

## Training Configuration

All FTWB1 models use identical hyperparameters to FTWB2:

```yaml
training:
  batch_size: 128
  eval_batch_size: 64
  learning_rate: 1e-5
  num_epochs: 30
  optimizer: adamw
  scheduler: linear_with_warmup
  warmup_steps: 50
  weight_decay: 0.01
```

**Initialization:** From corresponding seed's base model (`pt1_seed{1,2,3}/checkpoints/final`)

## Running the Pipeline

### Step 1: Train FTWB1 Models

```bash
# All seeds sequentially (recommended)
bash scripts/revision/exp1/training/train_all_ftwb1_sequential.sh

# Or individual seeds (can run in parallel)
bash scripts/revision/exp1/training/train_seed1_ftwb1_all.sh
bash scripts/revision/exp1/training/train_seed2_ftwb1_all.sh
bash scripts/revision/exp1/training/train_seed3_ftwb1_all.sh
```

### Step 2: Evaluate FTWB1 Models

```bash
bash scripts/revision/exp1/eval/eval_all_ftwb1.sh
```

### Step 3: Extract Representations

```bash
bash scripts/revision/exp1/representation_extraction/extract_all_ftwb1.sh
```

## Complete Exp1 Model Inventory

After training FTWB1, the complete exp1 will have:

| Model Type | Count | Description |
|------------|-------|-------------|
| Base (pt1_seed{1,2,3}) | 3 | Pretrained on all 7 tasks |
| FTWB1 (single-task FT) | 21 | 3 seeds × 7 tasks |
| FTWB2 (two-task FT) | 63 | 3 seeds × 21 combinations |
| **Total** | **87** | **Complete model set** |

## Expected Generalization Computation

For the paper plots, the expected generalization baseline is computed as:

```python
def get_expected_generalization(task, ftwb2_exp_num):
    """
    Get expected generalization for a task from FTWB2 experiment.

    Args:
        task: Target task to evaluate
        ftwb2_exp_num: FTWB2 experiment number (1-21)

    Returns:
        Best performance among single-task models trained on
        any of the tasks in this FTWB2 experiment
    """
    trained_tasks = TRAINING_DATA_2TASK[ftwb2_exp_num]  # e.g., ["distance", "trianglearea"]

    best_perf = None
    for trained_task in trained_tasks:
        ftwb1_num = TASK_TO_NUM[trained_task]  # Map task to ftwb1 number
        perf = load_ftwb1_performance(ftwb1_num, task)

        if best_perf is None or is_better(perf, best_perf, task):
            best_perf = perf

    return best_perf
```

**Interpretation:**
- If FTWB2 performs **better** than expected → multi-task synergy
- If FTWB2 performs **equal** to expected → no synergy, just max of specialists
- If FTWB2 performs **worse** than expected → negative transfer

## Generation Scripts

Created utility scripts to auto-generate all configs and scripts:

1. `src/scripts/generate_revision_exp1_ftwb1_configs.py` - Training configs
2. `src/scripts/generate_revision_exp1_ftwb1_train_scripts.py` - Training scripts
3. `src/scripts/generate_revision_exp1_ftwb1_eval_configs.py` - Eval configs
4. `src/scripts/generate_revision_exp1_ftwb1_repr_configs.py` - Repr configs

## Status

- ✅ Training configs created (21)
- ✅ Training scripts created (4)
- ✅ Eval configs created (315)
- ✅ Eval scripts created (1 master)
- ✅ Repr configs created (21)
- ✅ Repr scripts created (1 master)
- ⏳ **Models need to be trained** (21 models to train)
- ⏳ **Models need to be evaluated** (315 evaluations after training)
- ⏳ **Representations need to be extracted** (21 extractions after training)

## Next Steps

1. **Train FTWB1 models** using training scripts (~21 training runs)
2. **Run evaluations** after training completes
3. **Extract representations** after evaluations complete
4. **Update plotting script** to include expected generalization baseline
5. **Generate final paper plots** showing actual vs expected generalization

## Notes

- FTWB1 models are essential for the paper's core contribution
- Without FTWB1, we can only show raw performance, not relative improvement over single-task baseline
- The "expected generalization" is what a simple ensemble of specialists would achieve
- The gap between actual and expected measures true multi-task learning benefits
