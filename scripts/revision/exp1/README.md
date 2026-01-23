# Revision Exp1 Scripts

Scripts for running evaluations and representation extraction on revision/exp1 experiments (3 seeds × 22 models = 66 models total).

## Directory Structure

```
scripts/revision/exp1/
├── eval/                           # Evaluation scripts (990 total evaluations)
│   ├── eval_all_sequential.sh     # Run all evaluations sequentially (seed1→seed2→seed3)
│   ├── eval_all_by_seed.sh        # Run all evaluations grouped by batch
│   ├── eval_seed{1,2,3}_base_ftwb2-1-7.sh    # Base + ftwb2 1-7 (120 evals each)
│   ├── eval_seed{1,2,3}_ftwb2-8-14.sh        # FTWB2 8-14 (105 evals each)
│   └── eval_seed{1,2,3}_ftwb2-15-21.sh       # FTWB2 15-21 (105 evals each)
│
└── representation_extraction/      # Representation extraction scripts (66 total)
    ├── extract_all_sequential.sh  # Run all extractions sequentially (seed1→seed2→seed3)
    ├── extract_all_by_seed.sh     # Run all extractions grouped by batch
    ├── extract_seed{1,2,3}_base_ftwb2-1-7.sh   # Base + ftwb2 1-7 (8 extractions each)
    ├── extract_seed{1,2,3}_ftwb2-8-14.sh       # FTWB2 8-14 (7 extractions each)
    └── extract_seed{1,2,3}_ftwb2-15-21.sh      # FTWB2 15-21 (7 extractions each)
```

## Quick Start

### Run Everything Sequentially

**Evaluations (990 total):**
```bash
bash scripts/revision/exp1/eval/eval_all_sequential.sh
```

**Representation Extraction (66 total):**
```bash
bash scripts/revision/exp1/representation_extraction/extract_all_sequential.sh
```

### Run by Batches

**Evaluations - grouped by model range:**
```bash
# Seed 1 only
bash scripts/revision/exp1/eval/eval_seed1_base_ftwb2-1-7.sh
bash scripts/revision/exp1/eval/eval_seed1_ftwb2-8-14.sh
bash scripts/revision/exp1/eval/eval_seed1_ftwb2-15-21.sh

# Or all seeds, batch by batch
bash scripts/revision/exp1/eval/eval_all_by_seed.sh
```

**Representation Extraction - grouped by model range:**
```bash
# Seed 1 only
bash scripts/revision/exp1/representation_extraction/extract_seed1_base_ftwb2-1-7.sh
bash scripts/revision/exp1/representation_extraction/extract_seed1_ftwb2-8-14.sh
bash scripts/revision/exp1/representation_extraction/extract_seed1_ftwb2-15-21.sh

# Or all seeds, batch by batch
bash scripts/revision/exp1/representation_extraction/extract_all_by_seed.sh
```

### Run Individual Seeds

You can also run individual seed batches in parallel for faster completion:

```bash
# Terminal 1
bash scripts/revision/exp1/eval/eval_seed1_base_ftwb2-1-7.sh

# Terminal 2
bash scripts/revision/exp1/eval/eval_seed2_base_ftwb2-1-7.sh

# Terminal 3
bash scripts/revision/exp1/eval/eval_seed3_base_ftwb2-1-7.sh
```

## What Gets Evaluated/Extracted

### Models (66 total across 3 seeds)
- **Base models (3):** `pt1_seed{1,2,3}`
- **FTWB2 models (63):** `pt1_seed{1,2,3}_ftwb2-{1..21}`

### Evaluations (15 per model)
Each model is evaluated on:
- 7 atlantis (OOD) tasks: atlantis_distance, atlantis_trianglearea, atlantis_angle, atlantis_compass, atlantis_inside, atlantis_perimeter, atlantis_crossing
- 7 normal (ID) tasks: distance, trianglearea, angle, compass, inside, perimeter, crossing
- 1 multi-task evaluation

**Total:** 66 models × 15 tasks = **990 evaluations**

### Representation Extraction (1 per model)
Each model has representations extracted for:
- **Layer:** 5 (deepest layer)
- **Task:** First trained task (distance for base, varies for ftwb2)
- **Checkpoint:** Last checkpoint only

**Total:** 66 models × 1 extraction = **66 representation extractions**

## Execution Order Strategies

### 1. Sequential (Simplest)
Run `eval_all_sequential.sh` or `extract_all_sequential.sh`
- **Pros:** Simple, single command
- **Cons:** Slowest (no parallelization)

### 2. By Seed (Parallel Friendly)
Run `eval_all_by_seed.sh` or `extract_all_by_seed.sh`
- **Pros:** Completes each batch across all seeds before moving on
- **Cons:** Still sequential overall

### 3. Manual Parallel (Fastest)
Run individual scripts in parallel across different terminals/SLURM jobs
- **Pros:** Maximum parallelization, fastest completion
- **Cons:** Requires manual coordination

## Output Locations

### Evaluation Results
```
data/experiments/revision/exp1/
└── pt1_seed{1,2,3}_ftwb2-{1..21}/
    └── evals/
        ├── atlantis_{task}/eval_data/evaluation_results.json
        ├── {task}/eval_data/evaluation_results.json
        └── multi_task/eval_data/evaluation_results.json
```

### Representation Extraction Results
```
data/experiments/revision/exp1/
└── pt1_seed{1,2,3}_ftwb2-{1..21}/
    └── analysis_higher/
        └── {task}_firstcity_last_and_trans_l5/
            ├── representations/
            ├── pca_results/
            └── probe_results/
```

## Notes

- All scripts should be run from project root
- Evaluation uses last checkpoint only (not all checkpoints like original pt1)
- Representation extraction uses layer 5 only
- Each script has `--overwrite` flag to allow re-running
- Monitor GPU memory when running multiple scripts in parallel

## Documentation

See detailed documentation in:
- `docs/logs/2025-11-20/revision_exp1_eval_setup.md`
- `docs/logs/2025-11-20/revision_exp1_repr_setup.md`
