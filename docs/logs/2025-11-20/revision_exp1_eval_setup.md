# Revision Exp1 Evaluation Setup

**Date:** 2025-11-20
**Task:** Set up evaluations for revision/exp1 experiments

## Summary

Created evaluation configs and batch scripts for all revision/exp1 models (3 seeds × 22 models = 66 base + ftwb2 models).

Each model is evaluated on 15 different tasks:
- 7 atlantis (OOD) tasks
- 7 normal (in-distribution) tasks
- 1 multi-task evaluation

## What Was Created

### 1. Evaluation Configs (990 total)

**Location:** `configs/revision/exp1/eval/`

**Structure:**
```
configs/revision/exp1/eval/
├── seed1/
│   ├── base/           # 15 configs: 7 atlantis_*.yaml + 7 *.yaml + multi_task.yaml
│   ├── ftwb2-1/        # 15 configs
│   ├── ftwb2-2/        # 15 configs
│   ...
│   └── ftwb2-21/       # 15 configs
├── seed2/
│   └── [same structure as seed1]
└── seed3/
    └── [same structure as seed1]
```

**Key Config Settings:**
- `checkpoints: last` - Only evaluates the FINAL checkpoint (not all checkpoints)
- `save_full_results: false` - Only saves aggregated metrics
- `eval_batch_size: 512` - Efficient batch size
- Evaluates on 3 types of tasks:
  - **Atlantis (OOD):** atlantis_distance, atlantis_trianglearea, atlantis_angle, atlantis_compass, atlantis_inside, atlantis_perimeter, atlantis_crossing
  - **Normal (ID):** distance, trianglearea, angle, compass, inside, perimeter, crossing
  - **Multi-task:** Combined evaluation on all tasks

### 2. Batch Evaluation Scripts (9 total)

**Location:** `scripts/revision/exp1/eval/`

**Scripts:**
```
eval_seed1_base_ftwb2-1-7.sh    (120 evaluations: 15 tasks × 8 models)
eval_seed1_ftwb2-8-14.sh        (105 evaluations: 15 tasks × 7 models)
eval_seed1_ftwb2-15-21.sh       (105 evaluations: 15 tasks × 7 models)

eval_seed2_base_ftwb2-1-7.sh    (120 evaluations)
eval_seed2_ftwb2-8-14.sh        (105 evaluations)
eval_seed2_ftwb2-15-21.sh       (105 evaluations)

eval_seed3_base_ftwb2-1-7.sh    (120 evaluations)
eval_seed3_ftwb2-8-14.sh        (105 evaluations)
eval_seed3_ftwb2-15-21.sh       (105 evaluations)
```

**Total:** 990 evaluations (66 models × 15 tasks)

## Running Evaluations

Each script can be run independently:

```bash
# Run from project root
bash scripts/revision/exp1/eval/eval_seed1_base_ftwb2-1-7.sh
bash scripts/revision/exp1/eval/eval_seed1_ftwb2-8-14.sh
bash scripts/revision/exp1/eval/eval_seed1_ftwb2-15-21.sh
# ... etc
```

**Recommended:** Run scripts in parallel across seeds to maximize efficiency:
```bash
# In separate terminals or SLURM jobs
bash scripts/revision/exp1/eval/eval_seed1_base_ftwb2-1-7.sh &
bash scripts/revision/exp1/eval/eval_seed2_base_ftwb2-1-7.sh &
bash scripts/revision/exp1/eval/eval_seed3_base_ftwb2-1-7.sh &
```

## Output Structure

After running, evaluation results will be stored at:

```
data/experiments/revision/exp1/
├── pt1_seed1/
│   └── evals/
│       ├── atlantis_distance/
│       │   └── eval_data/
│       │       └── evaluation_results.json
│       ├── atlantis_trianglearea/
│       ├── ... (all 7 atlantis tasks)
│       ├── distance/
│       ├── trianglearea/
│       ├── ... (all 7 normal tasks)
│       └── multi_task/
├── pt1_seed1_ftwb2-1/
│   └── evals/
│       └── [same structure: 15 directories]
...
└── pt1_seed3_ftwb2-21/
    └── evals/
        └── [same structure: 15 directories]
```

## Generation Scripts

Created two utility scripts:

1. **`src/scripts/generate_revision_exp1_eval_configs.py`**
   - Generates all 990 evaluation config files (3 seeds × 22 models × 15 tasks)
   - Can be re-run if configs need to be regenerated

2. **`src/scripts/generate_revision_exp1_eval_scripts.py`**
   - Generates the 9 batch bash scripts
   - Can be re-run if script organization needs to change

## Next Steps

1. **Run evaluations** using the batch scripts
2. **Adapt plotting script** to visualize results (similar to ftwb2_evaluation_plot.png)
3. **Aggregate across seeds** for final publication plots

## Notes

- Each evaluation only processes the LAST checkpoint (not all checkpoints during training)
- This is different from the original pt1 evaluations which tracked all checkpoints
- Results will be aggregated across 3 seeds for the final analysis
- Evaluations include both:
  - **Atlantis (OOD):** Out-of-distribution generalization on unseen cities
  - **Normal (ID):** In-distribution performance on standard test split
  - **Multi-task:** Combined evaluation across all tasks
- This matches the evaluation structure of the original pt1_ftwb2-X experiments
