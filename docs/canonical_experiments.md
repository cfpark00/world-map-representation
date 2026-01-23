# Canonical Experiment Set

**Last Updated**: 2025-11-21

This document describes the canonical set of 179 experiments used for the paper revision. These experiments provide seed robustness across all training regimes.

## Overview

| Group | Description | Seeds | Experiments |
|-------|-------------|-------|-------------|
| PT1-X | Single-task pretraining (7 tasks) | 3 | 21 |
| PT2 | Two-task pretraining (7 variants) | 3 | 21 |
| PT3 | Three-task pretraining (7 variants) | 3 | 21 |
| PT1 | Multi-task pretraining (7 tasks) | 4 | 4 |
| PT1+FTWB1 | Multi-task → single-task fine-tuning | 4 | 28 |
| PT1+FTWB2 | Multi-task → two-task fine-tuning | 4 | 84 |
| **Total** | | | **179** |

---

## Group 1: Pretraining Experiments (63 total)

These experiments study representation formation during pretraining with 1, 2, or 3 tasks.

### PT1-X: Single-Task Pretraining (21 experiments)

Each model is trained on exactly 1 task. 7 tasks × 3 seeds = 21 experiments.

**Tasks:**
- pt1-1: distance
- pt1-2: trianglearea
- pt1-3: angle
- pt1-4: compass
- pt1-5: inside
- pt1-6: perimeter
- pt1-7: crossing

**Locations:**

| Seed | Path | Notes |
|------|------|-------|
| Original (seed42) | `data/experiments/pt1-{1-7}/` | |
| Seed1 | `data/experiments/revision/exp4/pt1-{1-7}_seed1/` | |
| Seed2 | `data/experiments/revision/exp4/pt1-{1-4,6-7}_seed2/` | |
| Seed3 (pt1-5 only) | `data/experiments/revision/exp4/pt1-5_seed3/` | **Replaces seed2 for pt1-5** |

**IMPORTANT:** pt1-5 (inside task) seed2 training failed. Seed3 is used instead.

### PT2: Two-Task Pretraining (21 experiments)

Each model is trained on exactly 2 tasks. 7 variants × 3 seeds = 21 experiments.

**Task Combinations:**
- pt2-1: distance + trianglearea
- pt2-2: angle + compass
- pt2-3: inside + perimeter
- pt2-4: crossing + distance
- pt2-5: trianglearea + angle
- pt2-6: compass + inside
- pt2-7: perimeter + crossing

**Locations:**

| Seed | Path |
|------|------|
| Original (seed42) | `data/experiments/pt2-{1-7}/` |
| Seed1 | `data/experiments/revision/exp2/pt2-{1-7}_seed1/` |
| Seed2 | `data/experiments/revision/exp2/pt2-{1-7}_seed2/` |

### PT3: Three-Task Pretraining (21 experiments)

Each model is trained on exactly 3 tasks. 7 variants × 3 seeds = 21 experiments.

**Task Combinations:**
- pt3-1: distance + trianglearea + angle
- pt3-2: compass + inside + perimeter
- pt3-3: crossing + distance + trianglearea
- pt3-4: angle + compass + inside
- pt3-5: perimeter + crossing + distance
- pt3-6: trianglearea + angle + compass
- pt3-7: inside + perimeter + crossing

**Note:** pt3-8 exists but is excluded from the canonical 3-seed set.

**Locations:**

| Seed | Path |
|------|------|
| Original (seed42) | `data/experiments/pt3-{1-7}/` |
| Seed1 | `data/experiments/revision/exp2/pt3-{1-7}_seed1/` |
| Seed2 | `data/experiments/revision/exp2/pt3-{1-7}_seed2/` |

---

## Group 2: Multi-Task + Fine-Tuning Experiments (116 total)

These experiments study fine-tuning dynamics from a multi-task pretrained model.

### PT1: Multi-Task Pretraining Base (4 experiments)

Each model is pretrained on all 7 tasks simultaneously.

**Locations:**

| Seed | Path |
|------|------|
| Original (seed42) | `data/experiments/pt1/` |
| Seed1 | `data/experiments/revision/exp1/pt1_seed1/` |
| Seed2 | `data/experiments/revision/exp1/pt1_seed2/` |
| Seed3 | `data/experiments/revision/exp1/pt1_seed3/` |

### PT1 + FTWB1: Single-Task Fine-Tuning (28 experiments)

Fine-tune each PT1 base model on 1 task. 4 seeds × 7 tasks = 28 experiments.

**FTWB1 Tasks:**
- ftwb1-1: distance
- ftwb1-2: trianglearea
- ftwb1-3: angle
- ftwb1-4: compass
- ftwb1-5: inside
- ftwb1-6: perimeter
- ftwb1-7: crossing

**Locations:**

| Seed | Path |
|------|------|
| Original | `data/experiments/pt1_ftwb1-{1-7}/` |
| Seed1 | `data/experiments/revision/exp1/pt1_seed1_ftwb1-{1-7}/` |
| Seed2 | `data/experiments/revision/exp1/pt1_seed2_ftwb1-{1-7}/` |
| Seed3 | `data/experiments/revision/exp1/pt1_seed3_ftwb1-{1-7}/` |

### PT1 + FTWB2: Two-Task Fine-Tuning (84 experiments)

Fine-tune each PT1 base model on 2 tasks. 4 seeds × 21 combinations = 84 experiments.

**FTWB2 Task Combinations (21 total):**
All unique pairs of 7 tasks: C(7,2) = 21

| ID | Tasks |
|----|-------|
| ftwb2-1 | distance + trianglearea |
| ftwb2-2 | angle + compass |
| ftwb2-3 | inside + perimeter |
| ftwb2-4 | crossing + distance |
| ftwb2-5 | trianglearea + angle |
| ftwb2-6 | compass + inside |
| ftwb2-7 | perimeter + crossing |
| ftwb2-8 | distance + angle |
| ftwb2-9 | trianglearea + compass |
| ftwb2-10 | angle + inside |
| ftwb2-11 | compass + perimeter |
| ftwb2-12 | inside + crossing |
| ftwb2-13 | perimeter + distance |
| ftwb2-14 | crossing + trianglearea |
| ftwb2-15 | distance + compass |
| ftwb2-16 | trianglearea + inside |
| ftwb2-17 | angle + perimeter |
| ftwb2-18 | compass + crossing |
| ftwb2-19 | inside + distance |
| ftwb2-20 | perimeter + trianglearea |
| ftwb2-21 | crossing + angle |

**Locations:**

| Seed | Path |
|------|------|
| Original | `data/experiments/pt1_ftwb2-{1-21}/` |
| Seed1 | `data/experiments/revision/exp1/pt1_seed1_ftwb2-{1-21}/` |
| Seed2 | `data/experiments/revision/exp1/pt1_seed2_ftwb2-{1-21}/` |
| Seed3 | `data/experiments/revision/exp1/pt1_seed3_ftwb2-{1-21}/` |

---

## Directory Structure Summary

```
data/experiments/
├── pt1/                          # Multi-task base (original)
├── pt1-{1-7}/                    # Single-task PT1-X (original)
├── pt2-{1-7}/                    # Two-task PT2 (original)
├── pt3-{1-7}/                    # Three-task PT3 (original)
├── pt1_ftwb1-{1-7}/              # FTWB1 (original)
├── pt1_ftwb2-{1-21}/             # FTWB2 (original)
│
└── revision/
    ├── exp1/                     # Multi-task seeds + fine-tuning
    │   ├── pt1_seed{1,2,3}/
    │   ├── pt1_seed{1,2,3}_ftwb1-{1-7}/
    │   └── pt1_seed{1,2,3}_ftwb2-{1-21}/
    │
    ├── exp2/                     # PT2/PT3 seeds
    │   ├── pt2-{1-7}_seed{1,2}/
    │   └── pt3-{1-7}_seed{1,2}/
    │
    └── exp4/                     # PT1-X seeds
        └── pt1-{1-7}_seed{1,2,3}/  # Note: pt1-5 uses seed3 instead of seed2
```

---

## Quick Reference

**To iterate over all PT1-X experiments (21):**
```python
experiments = []
for task in range(1, 8):
    experiments.append(f"pt1-{task}")  # original
    experiments.append(f"pt1-{task}_seed1")
    if task == 5:
        experiments.append(f"pt1-{task}_seed3")  # exception!
    else:
        experiments.append(f"pt1-{task}_seed2")
```

**To iterate over all PT2 experiments (21):**
```python
experiments = []
for var in range(1, 8):
    experiments.append(f"pt2-{var}")
    experiments.append(f"pt2-{var}_seed1")
    experiments.append(f"pt2-{var}_seed2")
```

**To iterate over all PT1+FTWB2 experiments (84):**
```python
experiments = []
seeds = ["", "_seed1", "_seed2", "_seed3"]
for seed in seeds:
    for ft in range(1, 22):
        if seed == "":
            experiments.append(f"pt1_ftwb2-{ft}")
        else:
            experiments.append(f"pt1{seed}_ftwb2-{ft}")
```
