# CKA Analysis Infrastructure v2

## Overview

Clean, maintainable CKA (Centered Kernel Alignment) analysis infrastructure following the design from `world-representation`. This implementation does NOT interact with the old CKA infrastructure.

## Key Improvements

1. **Hierarchical Organization**: Group → Pair → Layer structure
2. **Consistent Naming**: Uses actual experiment names consistently
3. **Standard Output**: Each analysis produces `config.yaml`, `cka_timeline.csv`, `cka_timeline.png`, `summary.json`
4. **Single Source of Truth**: Experiment metadata centralized in `experiment_registry.py`
5. **Automatic Config Generation**: Generate all configs with one command
6. **Clean Code**: Modular, well-documented, easy to maintain

## Directory Structure

```
data/analysis_v2/cka/
├── pt1_all_pairs/                    # PT1 single-task comparisons
│   ├── pt1-1_vs_pt1-2/
│   │   ├── layer3/
│   │   │   ├── config.yaml           # Config used for this analysis
│   │   │   ├── cka_timeline.csv      # Per-checkpoint CKA values
│   │   │   ├── cka_timeline.png      # Timeline plot
│   │   │   └── summary.json          # Statistics (final, mean, std, etc.)
│   │   ├── layer4/
│   │   ├── layer5/
│   │   └── layer6/
│   ├── pt1-1_vs_pt1-3/
│   └── ... (28 pairs total: 7 choose 2 + 7 diagonal)
│
├── pt2_all_pairs/                    # PT2 two-task comparisons
│   └── ... (similar structure)
│
└── pt3_all_pairs/                    # PT3 multi-task comparisons
    └── ... (similar structure)

configs/analysis_v2/cka/
├── pt1_all_pairs/
│   ├── pt1-1_vs_pt1-2/
│   │   ├── layer3.yaml
│   │   ├── layer4.yaml
│   │   ├── layer5.yaml
│   │   └── layer6.yaml
│   └── ...
└── ...
```

## Code Structure

```
src/analysis/cka_v2/
├── __init__.py
├── compute_cka.py             # Core CKA math (CPU/GPU)
├── load_representations.py    # Load & align representations
└── experiment_registry.py     # Experiment metadata (single source of truth)

src/scripts/
├── analyze_cka_pair.py        # Main analysis script (one pair, one layer)
└── generate_cka_configs.py    # Auto-generate configs for all pairs

scripts/analysis_v2/
├── generate_all_cka_configs.sh        # Generate configs for pt1, pt2, pt3
└── run_cka_pt1_layer5_example.sh      # Example: run one analysis
```

## Usage

### Step 1: Generate Configs

Generate all CKA configs for all experiment types:

```bash
bash scripts/analysis_v2/generate_all_cka_configs.sh
```

Or generate configs for specific experiment type:

```bash
uv run python src/scripts/generate_cka_configs.py --type pt1 --layers 3,4,5,6
uv run python src/scripts/generate_cka_configs.py --type pt2 --layers 5
```

This creates YAML config files in `configs/analysis_v2/cka/{group_name}/`.

### Step 2: Run CKA Analysis

Run analysis for a single pair and layer:

```bash
uv run python src/scripts/analyze_cka_pair.py \
    configs/analysis_v2/cka/pt1_all_pairs/pt1-1_vs_pt1-2/layer5.yaml \
    --overwrite
```

Or use the example script:

```bash
bash scripts/analysis_v2/run_cka_pt1_layer5_example.sh
```

### Step 3: View Results

Results are saved in the `output_dir` specified in the config:

```bash
# View summary statistics
cat data/analysis_v2/cka/pt1_all_pairs/pt1-1_vs_pt1-2/layer5/summary.json

# View timeline data
head data/analysis_v2/cka/pt1_all_pairs/pt1-1_vs_pt1-2/layer5/cka_timeline.csv

# View plot
open data/analysis_v2/cka/pt1_all_pairs/pt1-1_vs_pt1-2/layer5/cka_timeline.png
```

## Config File Format

```yaml
output_dir: data/analysis_v2/cka/pt1_all_pairs/pt1-1_vs_pt1-2/layer5

exp1:
  name: pt1-1
  repr_dir: data/experiments/pt1-1/analysis_higher/distance_firstcity_last_and_trans_l5/representations
  task: distance

exp2:
  name: pt1-2
  repr_dir: data/experiments/pt1-2/analysis_higher/trianglearea_firstcity_last_and_trans_l5/representations
  task: trianglearea

layer: 5
checkpoint_steps: null              # null = all checkpoints
city_filter: region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$
kernel_type: linear
center_kernels: true
use_gpu: true
save_timeline_plot: true
```

## Output Files

Each analysis produces:

1. **`config.yaml`**: Copy of input config for reproducibility
2. **`cka_timeline.csv`**: Per-checkpoint CKA values
   ```csv
   step,cka
   0,0.7120
   8204,0.2165
   ...
   ```
3. **`cka_timeline.png`**: Plot of CKA over training
4. **`summary.json`**: Summary statistics
   ```json
   {
     "exp1": "pt1-1",
     "exp2": "pt1-2",
     "layer": 5,
     "n_checkpoints": 41,
     "n_cities": 4413,
     "final_cka": 0.3924,
     "mean_cka": 0.3923,
     "std_cka": 0.0715,
     "min_cka": 0.1977,
     "max_cka": 0.7120
   }
   ```

## Experiment Registry

The `experiment_registry.py` module provides a single source of truth for experiment metadata:

```python
from src.analysis.cka_v2.experiment_registry import (
    get_pt1_experiments,
    get_pt2_experiments,
    get_repr_path,
    TASK_NAMES,
)

# Get all PT1 experiments
experiments = get_pt1_experiments()
# Returns: {'pt1-1': {...}, 'pt1-2': {...}, ...}

# Get representation path
repr_path = get_repr_path('pt1-1', 'distance', layer=5)
# Returns: Path to representations directory
```

## Comparison with Old Infrastructure

| Aspect | Old (WM_1) | New (v2) |
|--------|------------|----------|
| **Structure** | Flat, scattered | Hierarchical: group/pair/layer |
| **Configs** | 21+ manual files | Auto-generated from registry |
| **Output** | 4+ different directories | Single `analysis_v2/` tree |
| **Results** | Scattered .json files | Standard 4-file format |
| **Naming** | Inconsistent | Consistent experiment names |
| **Code** | Duplicated logic | Modular, DRY |
| **Maintenance** | Hard to update | Easy to maintain |

## Running Multiple Analyses

To analyze all PT1 pairs at layer 5, you can create a loop script:

```bash
#!/bin/bash
for config in configs/analysis_v2/cka/pt1_all_pairs/*/layer5.yaml; do
    echo "Processing $config"
    uv run python src/scripts/analyze_cka_pair.py "$config" --overwrite
done
```

Or submit as parallel SLURM jobs (see SLURM job scripts in `scripts/analysis_v2/`).

## Testing

Tested successfully on:
- ✅ PT1-1 vs PT1-2, layer 5
- ✅ 41 checkpoints processed
- ✅ 4413 cities (after filtering Atlantis and low-ID cities)
- ✅ GPU-accelerated CKA computation
- ✅ Timeline plot and summary statistics

## Future Enhancements

Planned additions:
1. Result aggregation script to create master CSV
2. Matrix visualization (heatmaps for all pairs)
3. Cross-seed CKA analysis support
4. Trend analysis across layers
5. Statistical significance testing

## Notes

- This infrastructure is completely independent of the old CKA code
- Old results are in `data/experiments/cka_analysis/` and `scratch/cka_analysis_clean/`
- New results are in `data/analysis_v2/cka/`
- The two can coexist without conflicts
