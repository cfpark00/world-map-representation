# Exp4: PT1 Single-Task Seed Robustness

## Overview

This experiment tests the robustness of PT1 single-task pretraining across different random seeds.

**Experiments**:
- 7 tasks (distance, trianglearea, angle, compass, inside, perimeter, crossing)
- 2 seeds per task (seed1, seed2) + original (seed 42)
- Total: 21 experiments

## Directory Structure

```
exp4/
├── pt1_single_task_seed/       # Training scripts
│   ├── pt1-1/
│   │   ├── pt1-1_seed1.sh
│   │   └── pt1-1_seed2.sh
│   └── ... (pt1-2 through pt1-7)
│
├── representation_extraction/   # Representation analysis
│   ├── extract_seed1_representations.sh
│   └── extract_seed2_representations.sh
│
├── cka_analysis/               # CKA similarity analysis
│   └── run_all_cka_cross_seed.sh
│
└── run_pt1-X_all.sh           # Run both seeds for task X
```

## Workflow

### 1. Train Experiments (Already Done for Seed1)

```bash
# Train seed1 (already done)
bash scripts/revision/exp4/run_pt1-1_all.sh  # Trains seed1 and seed2 for distance
bash scripts/revision/exp4/run_pt1-2_all.sh  # Trains seed1 and seed2 for trianglearea
# ... etc

# Or individually
bash scripts/revision/exp4/pt1_single_task_seed/pt1-1/pt1-1_seed1.sh
```

### 2. Extract Representations

**For Seed1** (ready now):
```bash
bash scripts/revision/exp4/representation_extraction/extract_seed1_representations.sh
```

**For Seed2** (after training):
```bash
bash scripts/revision/exp4/representation_extraction/extract_seed2_representations.sh
```

### 3. Compute 21×21 CKA Matrix

```bash
bash scripts/revision/exp4/cka_analysis/run_all_cka_cross_seed.sh
```

This computes CKA between all 21 experiments (7 tasks × 3 seeds).

## Results

**Training Results**: `data/experiments/revision/exp4/pt1-X_seedY/`
**Representations**: `data/experiments/revision/exp4/pt1-X_seedY/analysis_higher/`
**CKA Results**: `data/analysis_v2/cka/pt1_cross_seed/`

## Analysis

The 21×21 CKA matrix will show:
- **Same task, different seeds**: How robust are representations to seed changes?
- **Different tasks, same seed**: How task-specific are representations?
- **Different tasks, different seeds**: Overall dissimilarity

See `docs/cross_seed_cka_21x21.md` for full details.
