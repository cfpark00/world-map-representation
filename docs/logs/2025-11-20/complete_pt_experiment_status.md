# Complete PT Experiment Status

**Date**: 2025-11-20

## Overview

Full matrix of PT experiments across all task combinations and seeds:
- **PT1-X**: Single-task pretraining (7 tasks)
- **PT2**: Two-task pretraining (8 variants)
- **PT3**: Three-task pretraining (8 variants)
- **Seeds**: Original (42), Seed 1, Seed 2

## Training Status

### PT1-X (Single-task, 7 tasks)

| Task | Seed 42 | Seed 1 | Seed 2 | Total |
|------|---------|--------|--------|-------|
| pt1-1 (distance) | ✓ | ✓ | ✓ | 3/3 |
| pt1-2 (trianglearea) | ✓ | ✓ | ✓ | 3/3 |
| pt1-3 (angle) | ✓ | ✓ | ✓ | 3/3 |
| pt1-4 (compass) | ✓ | ✓ | ✓ | 3/3 |
| pt1-5 (inside) | ✓ | ✓ | ✓ | 3/3 |
| pt1-6 (perimeter) | ✓ | ✓ | ✓ | 3/3 |
| pt1-7 (crossing) | ✓ | ✓ | ✓ | 3/3 |

**PT1-X Total: 21/21 ✓ Complete**

### PT2 (Two-task, 8 variants)

| Variant | Tasks | Seed 42 | Seed 1 | Seed 2 | Total |
|---------|-------|---------|--------|--------|-------|
| pt2-1 | distance+trianglearea | ✓ | ✓ | ✓ | 3/3 |
| pt2-2 | angle+compass | ✓ | ✓ | ✓ | 3/3 |
| pt2-3 | inside+perimeter | ✓ | ✓ | ✓ | 3/3 |
| pt2-4 | crossing+distance | ✓ | ✓ | ✓ | 3/3 |
| pt2-5 | trianglearea+angle | ✓ | ✓ | ✓ | 3/3 |
| pt2-6 | compass+inside | ✓ | ✓ | ✓ | 3/3 |
| pt2-7 | perimeter+crossing | ✓ | ✓ | ✓ | 3/3 |
| pt2-8 | ? | ✓ | ✗ | ✗ | 1/3 |

**PT2 Total: 22/24 (92% complete)**
**Missing: pt2-8_seed1, pt2-8_seed2**

### PT3 (Three-task, 8 variants)

| Variant | Tasks | Seed 42 | Seed 1 | Seed 2 | Total |
|---------|-------|---------|--------|--------|-------|
| pt3-1 | distance+trianglearea+angle | ✓ | ✓ | ✓ | 3/3 |
| pt3-2 | compass+inside+perimeter | ✓ | ✓ | ✓ | 3/3 |
| pt3-3 | crossing+distance+trianglearea | ✓ | ✓ | ✓ | 3/3 |
| pt3-4 | angle+compass+inside | ✓ | ✓ | ✓ | 3/3 |
| pt3-5 | perimeter+crossing+distance | ✓ | ✓ | ✓ | 3/3 |
| pt3-6 | trianglearea+angle+compass | ✓ | ✓ | ✓ | 3/3 |
| pt3-7 | inside+perimeter+crossing | ✓ | ✓ | ✓ | 3/3 |
| pt3-8 | distance+trianglearea+compass | ✓ | ✗ | ✗ | 1/3 |

**PT3 Total: 22/24 (92% complete)**
**Missing: pt3-8_seed1, pt3-8_seed2**

## Grand Total

**Trained: 65/69 experiments (94% complete)**

By type:
- PT1-X: 21/21 ✓ (100%)
- PT2: 22/24 (92%)
- PT3: 22/24 (92%)

**Missing: 4 experiments**
- pt2-8_seed1
- pt2-8_seed2
- pt3-8_seed1
- pt3-8_seed2

## Representation Extraction Infrastructure

### PT1-X (Exp4)
✓ **Complete** - Has repr extraction + PCA for all 21 experiments
- Original (seed 42): 7 × 4 layers = 28 configs
- Seed 1: 7 × 4 layers = 28 configs
- Seed 2: 7 × 4 layers = 28 configs (where applicable)

### PT2 (Exp2)
✓ **Complete** - Has repr extraction + PCA for trained experiments
- 22 experiments with layer 5 extraction
- 3 PCA types: mixed, raw, na
- Missing configs only for pt2-8_seed{1,2} (not trained)

### PT3 (Exp2)
✓ **Just created** - Has repr extraction + PCA for trained experiments
- 21 experiments (7 variants × 3 seeds) with layer 5 extraction
- 3 PCA types: mixed, raw, na
- Missing configs only for pt3-8_seed{1,2} (not trained)

## Task Extraction Strategy

For multi-task experiments, we extract representations using the **first task** from each combination:

**PT2:**
- pt2-1: distance (from distance+trianglearea)
- pt2-2: angle (from angle+compass)
- pt2-3: inside (from inside+perimeter)
- pt2-4: crossing (from crossing+distance)
- pt2-5: trianglearea (from trianglearea+angle)
- pt2-6: compass (from compass+inside)
- pt2-7: perimeter (from perimeter+crossing)

**PT3:**
- pt3-1: distance (from distance+trianglearea+angle)
- pt3-2: compass (from compass+inside+perimeter)
- pt3-3: crossing (from crossing+distance+trianglearea)
- pt3-4: angle (from angle+compass+inside)
- pt3-5: perimeter (from perimeter+crossing+distance)
- pt3-6: trianglearea (from trianglearea+angle+compass)
- pt3-7: inside (from inside+perimeter+crossing)

## Next Steps

1. **Train missing experiments** (if needed):
   - pt2-8_seed1, pt2-8_seed2
   - pt3-8_seed1, pt3-8_seed2

2. **Generate missing infrastructure** (when trained):
   - Re-run generator scripts to create configs for pt2-8 and pt3-8 seeds

3. **Analysis ready**: All trained experiments have complete repr extraction + PCA infrastructure

## File Locations

**Original experiments:**
- PT1-X: `data/experiments/pt1-{1-7}/`
- PT2: `data/experiments/pt2-{1-8}/`
- PT3: `data/experiments/pt3-{1-8}/`

**Seed experiments:**
- PT1-X: `data/experiments/revision/exp4/pt1-{1-7}_seed{1,2}/`
- PT2: `data/experiments/revision/exp2/pt2-{1-7}_seed{1,2}/`
- PT3: `data/experiments/revision/exp2/pt3-{1-7}_seed{1,2}/`

**Note**: PT2-8 and PT3-8 seeds are missing, everything else is trained and has infrastructure.
