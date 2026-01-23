# Cross-Seed CKA Analysis Setup

## Overview

This setup computes a **14×14 CKA similarity matrix** comparing:
- **7 original PT1 experiments** (pt1-1 through pt1-7, seed 42)
- **7 seed1 PT1 experiments** (pt1-1_seed1 through pt1-7_seed1)

Each experiment uses the correct **task-specific prompt** (e.g., `distance_last_and_trans`, `trianglearea_last_and_trans`, not all using `distance_firstcity_last_and_trans`).

## Task Mappings

| Experiment | Task | Prompt Format |
|------------|------|---------------|
| pt1-1 | distance | `distance_last_and_trans` |
| pt1-2 | trianglearea | `trianglearea_last_and_trans` |
| pt1-3 | angle | `angle_last_and_trans` |
| pt1-4 | compass | `compass_last_and_trans` |
| pt1-5 | inside | `inside_last_and_trans` |
| pt1-6 | perimeter | `perimeter_last_and_trans` |
| pt1-7 | crossing | `crossing_last_and_trans` |

## Directory Structure

```
configs/analysis_v2/
├── representation_extraction/
│   ├── original/
│   │   ├── pt1-1/
│   │   │   ├── distance_last_and_trans_l3.yaml
│   │   │   ├── distance_last_and_trans_l4.yaml
│   │   │   ├── distance_last_and_trans_l5.yaml
│   │   │   └── distance_last_and_trans_l6.yaml
│   │   ├── pt1-2/
│   │   │   ├── trianglearea_last_and_trans_l3.yaml
│   │   │   └── ...
│   │   └── ... (pt1-3 through pt1-7)
│   └── seed1/
│       ├── pt1-1_seed1/
│       ├── pt1-2_seed1/
│       └── ... (pt1-3_seed1 through pt1-7_seed1)
│
└── cka_cross_seed/
    ├── pt1-1_vs_pt1-1/
    │   ├── layer3.yaml
    │   ├── layer4.yaml
    │   ├── layer5.yaml
    │   └── layer6.yaml
    ├── pt1-1_vs_pt1-2/
    ├── pt1-1_vs_pt1-1_seed1/
    ├── pt1-1_seed1_vs_pt1-1_seed1/
    └── ... (105 unique pairs total)

scripts/analysis_v2/
└── repr_extraction/
    ├── original/
    │   ├── extract_pt1-1_distance_l3.sh
    │   ├── extract_pt1-1_distance_l4.sh
    │   └── ... (28 scripts total: 7 tasks × 4 layers)
    ├── seed1/
    │   ├── extract_pt1-1_seed1_distance_l3.sh
    │   └── ... (28 scripts total)
    └── run_all_extractions.sh  # Master script
```

## Workflow

### Step 1: Extract Representations

Extract representations for all experiments using task-specific prompts:

```bash
# Option 1: Run all extractions at once
bash scripts/analysis_v2/repr_extraction/run_all_extractions.sh

# Option 2: Run individual extractions
bash scripts/analysis_v2/repr_extraction/original/extract_pt1-1_distance_l5.sh
bash scripts/analysis_v2/repr_extraction/seed1/extract_pt1-1_seed1_distance_l5.sh

# Option 3: Run by group
for script in scripts/analysis_v2/repr_extraction/original/*.sh; do
    bash "$script"
done

for script in scripts/analysis_v2/repr_extraction/seed1/*.sh; do
    bash "$script"
done
```

**Expected Output**: Representations saved in:
- Original: `data/experiments/pt1-X/analysis_higher/{task}_last_and_trans_lY/representations/`
- Seed1: `data/experiments/revision/exp4/pt1-X_seed1/analysis_higher/{task}_last_and_trans_lY/representations/`

### Step 2: Compute CKA Matrix

Compute CKA for all 105 unique pairs (14×14 upper triangle + diagonal):

```bash
# Run specific pair
uv run python src/scripts/analyze_cka_pair.py \
    configs/analysis_v2/cka_cross_seed/pt1-1_vs_pt1-1_seed1/layer5.yaml \
    --overwrite

# Run all pairs for a specific layer (can parallelize)
for config in configs/analysis_v2/cka_cross_seed/*/layer5.yaml; do
    echo "Processing $config"
    uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
done
```

**Expected Output**: CKA results in:
```
data/analysis_v2/cka/pt1_cross_seed/
├── pt1-1_vs_pt1-1/
│   └── layer5/
│       ├── config.yaml
│       ├── cka_timeline.csv
│       ├── cka_timeline.png
│       └── summary.json
├── pt1-1_vs_pt1-1_seed1/
│   └── layer5/
│       └── ...
└── ...
```

### Step 3: Aggregate Results

Collect all CKA final values into a 14×14 matrix:

```python
import pandas as pd
import json
from pathlib import Path

# Collect all summary.json files
results = []
for summary_file in Path('data/analysis_v2/cka/pt1_cross_seed').glob('*/layer5/summary.json'):
    with open(summary_file) as f:
        data = json.load(f)
        results.append({
            'exp1': data['exp1'],
            'exp2': data['exp2'],
            'final_cka': data['final_cka'],
        })

df = pd.DataFrame(results)

# Pivot to matrix form
matrix = df.pivot(index='exp1', columns='exp2', values='final_cka')

# Fill lower triangle (CKA is symmetric)
matrix = matrix.combine_first(matrix.T)

print(matrix)
```

## Generated Files Summary

### Representation Extraction
- **Configs**: 56 total (7 tasks × 2 seeds × 4 layers)
  - 28 for original experiments
  - 28 for seed1 experiments
- **Scripts**: 56 bash scripts + 1 master script

### CKA Analysis
- **Configs**: 420 total (105 pairs × 4 layers)
  - Pairs include:
    - Original vs Original: 28 pairs (7×7 upper triangle + diagonal)
    - Original vs Seed1: 49 pairs (7×7)
    - Seed1 vs Seed1: 28 pairs (7×7 upper triangle + diagonal)
  - **Total unique pairs**: 105

## Key Differences from Old Infrastructure

1. **Task-Specific Prompts**: Each task uses its own prompt format, not all using `distance`
2. **Organized Structure**: Clear separation of original vs seed1 experiments
3. **Reproducible**: All configs saved with results
4. **Scalable**: Easy to add more seeds or tasks

## Matrix Interpretation

The 14×14 CKA matrix will show:
- **Diagonal**: Self-similarity (should be 1.0)
- **Within-seed blocks**: How similar different tasks are within same seed
- **Across-seed blocks**: How similar same task is across different seeds
- **Cross-task cross-seed**: How different tasks compare across seeds

Expected patterns:
- Same task, different seeds → High CKA (if training is robust)
- Different tasks, same seed → Lower CKA (task-specific representations)
- Different tasks, different seeds → Lowest CKA

## Notes

- All experiments use seed 42 for representation extraction (PCA probes)
- Original experiments were trained with seed 42
- Seed1 experiments were trained with seed 1
- City filter excludes Atlantis and low-ID cities: `region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$`
