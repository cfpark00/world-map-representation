# Codebase Reorganization Plan

This document tracks the migration from flat structure to track-based organization.

## Target Track Structure

```
src/<track_name>/           # Track-specific implementation
configs/<track_name>/       # Track-specific configs
scripts/<track_name>/       # Track-specific bash scripts
data/<track_name>/          # Track-specific outputs (gitignored)
docs/tracks/<track_name>/   # Track-specific documentation
```

## Planned Tracks

| Track | Purpose | Status |
|-------|---------|--------|
| `data_generation_v1` | Data generation + tokenizer creation | ✅ DONE |
| `pretraining_v1` | All pretraining (PT1, PT1-X, PT2, PT3, all seeds) | ⏳ TODO |
| `finetuning_v1` | All fine-tuning (FT, FTWB, all combos/seeds) | ⏳ TODO |
| `cka_v1` | CKA analysis infrastructure | ⏳ TODO |

**Not tracks** (downstream of training):
- Evaluation (performance metrics) - downstream of pretraining/finetuning
- Representation extraction - downstream of pretraining/finetuning

---

## Completed: data_generation_v1

### Code Structure
```
src/data_generation_v1/
├── __init__.py
├── tasks/                    # 7 task implementations
│   ├── angle.py
│   ├── compass.py
│   ├── crossing.py
│   ├── distance.py
│   ├── inside.py
│   ├── perimeter.py
│   └── trianglearea.py
├── create_city_dataset.py
├── combine_datasets.py
├── append_cities_to_dataset.py
├── create_tokenizer.py
└── utils.py                  # Track-specific utilities (pair generation)
```

### Config Structure
```
configs/data_generation_v1/
├── cities/
│   └── city_dataset_default.yaml
├── tokenizers/
│   └── default_tokenizer.yaml
├── single_tasks/                    # 21 yamls (7 tasks × 3 variants)
│   ├── {task}_1M_no_atlantis.yaml
│   ├── {task}_1M_with_atlantis.yaml
│   └── {task}_100k_atlantis_required.yaml
└── derived/
    ├── pretraining/                 # 18 yamls
    │   ├── pt7.yaml                 # All 7 tasks (was multitask_pt1)
    │   ├── pt7_with_atlantis.yaml
    │   ├── pt2-{1..8}.yaml          # 2-task combos
    │   └── pt3-{1..8}.yaml          # 3-task combos
    └── finetuning/                  # 70 yamls
        ├── ft1-{1..7}.yaml          # 1-task fine-tuning
        ├── ft2-{1..21}.yaml         # 2-task fine-tuning
        ├── ft3-{1..7}.yaml          # 3-task fine-tuning
        ├── ftwb1-{1..7}.yaml        # 1-task FT with warmup+baseline
        ├── ftwb2-{1..21}.yaml       # 2-task FT with warmup+baseline
        └── ftwb3-{1..7}.yaml        # 3-task FT with warmup+baseline
```

### Output Structure
```
data/data_generation_v1/
├── cities/                   # City dataset
│   ├── cities.csv
│   ├── config.yaml
│   └── metadata.json
├── tokenizers/               # Tokenizers
│   └── default_tokenizer/
├── single_datasets/          # Individual task datasets
│   ├── distance_1M_no_atlantis/
│   ├── distance_1M_with_atlantis/
│   └── ...
└── derived_datasets/         # Combined/mixed datasets
    ├── multitask_pt1/
    ├── ftwb1-1/
    └── ...
```

### Changes Made

1. **Moved files:**
   - `src/data_processing/` → `src/data_generation_v1/`
   - `src/tasks/` → `src/data_generation_v1/tasks/`
   - `src/create_tokenizer.py` → `src/data_generation_v1/`
   - `configs/data_generation/` → `configs/data_generation_v1/`
   - `configs/tokenizers/` → `configs/data_generation_v1/tokenizers/`
   - `scripts/data_generation/` → `scripts/data_generation_v1/`
   - `scripts/tokenizers/` → `scripts/data_generation_v1/tokenizers/`

2. **Updated all config paths** (171 files):
   - `data/datasets/cities` → `data/data_generation_v1/cities`
   - `data/datasets/{task}_*` → `data/data_generation_v1/single_datasets/{task}_*`
   - `data/datasets/multitask_*` → `data/data_generation_v1/derived_datasets/multitask_*`
   - `data/datasets/ft*` → `data/data_generation_v1/derived_datasets/ft*`
   - `data/tokenizers/` → `data/data_generation_v1/tokenizers/`

3. **Fixed imports:**
   - `from src.data_processing.data_utils` → `from src.data_generation_v1.utils`

4. **Added --debug flags** to scripts missing them:
   - `create_city_dataset.py` - limits to 50 cities
   - `combine_datasets.py` - limits to 100 samples per split

5. **Removed non-core tasks:**
   - randomwalk, randring, center, circlecount, nearest_neighbor
   - Now only 7 core tasks: distance, trianglearea, angle, compass, inside, perimeter, crossing

6. **Replaced hardcoded cluster paths** (8,804 files):
   - `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1` → ``
   - `/n/home12/cfpark00/WM_1` → ``
   - `/n/home12/cfpark00/datadir/WM_1` → ``

### Tested

✅ City generation: `uv run python src/data_generation_v1/create_city_dataset.py configs/data_generation_v1/city_dataset_default.yaml`
- Output: `data/data_generation_v1/cities/` (5,175 cities)

✅ Tokenizer generation: `uv run python src/data_generation_v1/create_tokenizer.py configs/data_generation_v1/tokenizers/default_tokenizer.yaml`
- Output: `data/data_generation_v1/tokenizers/default_tokenizer/` (98 tokens)

### Not Yet Tested
- Individual task dataset generation (need cities first ✅)
- Combined dataset generation (need single datasets first)

---

## Naming Convention Plan

### The 7 Tasks (canonical order)
```
1. distance
2. trianglearea
3. angle
4. compass
5. inside
6. perimeter
7. crossing
```

### Combinatorics
- 1-task: 7 combinations (C(7,1) = 7)
- 2-task: 21 combinations (C(7,2) = 21)
- 3-task: 35 combinations (C(7,3) = 35)
- 7-task: 1 (all tasks)

### Data Generation Naming (`configs/data_generation_v1/`)

**Single task datasets** (`single_tasks/`):
```
{task}_1M_no_atlantis.yaml          # 1M samples, no Atlantis cities
{task}_1M_with_atlantis.yaml        # 1M samples, includes Atlantis
{task}_100k_atlantis_required.yaml  # 100k samples, Atlantis required (for elicitation)
```

**Derived/combined datasets** (`derived/`):

| Config Name | Description | Count | Status |
|-------------|-------------|-------|--------|
| `pt7.yaml` | All 7 tasks combined (pretraining) | 1 | Rename from `multitask_pt1` |
| `pt2-{1..21}.yaml` | 2-task pretraining combos | 8 of 21 | Partial (compute limited) |
| `pt3-{1..35}.yaml` | 3-task pretraining combos | 8 of 35 | Partial (compute limited) |
| `ft1-{1..7}.yaml` | 1-task fine-tuning + elicitation | 7 | ✓ Complete |
| `ft2-{1..21}.yaml` | 2-task fine-tuning + elicitation | 21 | ✓ Complete |
| `ft3-{1..35}.yaml` | 3-task fine-tuning + elicitation | 7 of 35 | Partial |
| `ftwb1-{1..7}.yaml` | 1-task FT with warmup+baseline | 7 | ✓ Complete |
| `ftwb2-{1..21}.yaml` | 2-task FT with warmup+baseline | 21 | ✓ Complete |
| `ftwb3-{1..35}.yaml` | 3-task FT with warmup+baseline | 7 of 35 | Partial |

**Note:** pt1-X (single-task pretraining) uses `single_tasks/` directly, no combining needed.

### Training Naming (`configs/pretraining_v1/`, `configs/finetuning_v1/`)

**Pretraining:**
```
pt1-{1..7}.yaml     # Single-task pretraining (task 1-7)
pt2-{1..21}.yaml    # Two-task pretraining
pt3-{1..35}.yaml    # Three-task pretraining
pt7.yaml            # All 7 tasks
pt7_seed{N}.yaml    # Seed variations
```

**Fine-tuning:**
```
ft1-{1..7}.yaml     # Single-task fine-tuning
ft2-{1..21}.yaml    # Two-task fine-tuning
ft3-{1..35}.yaml    # Three-task fine-tuning
ftwb1-{1..7}.yaml   # With warmup+baseline (elicitation)
ftwb2-{1..21}.yaml
ftwb3-{1..35}.yaml
```

### Config Rename ✅ COMPLETE (2026-01-23)

**Location:** `configs/data_generation_v1/derived/pretraining/` and `derived/finetuning/`

**What was done:**
1. Dropped `combine_` prefix from all 88 filenames
2. Renamed `multitask_pt1` → `pt7` (both filename and output_dir)
3. Updated `output_dir` in pt7.yaml and pt7_with_atlantis.yaml

**Pretraining (18 files):**
- `pt7.yaml` (was `combine_multitask_pt1.yaml`)
- `pt7_with_atlantis.yaml` (was `combine_multitask_pt1_with_atlantis.yaml`)
- `pt2-{1..8}.yaml` (was `combine_pt2-{1..8}.yaml`)
- `pt3-{1..8}.yaml` (was `combine_pt3-{1..8}.yaml`)

**Finetuning (70 files):**
- `ft1-{1..7}.yaml`, `ft2-{1..21}.yaml`, `ft3-{1..7}.yaml`
- `ftwb1-{1..7}.yaml`, `ftwb2-{1..21}.yaml`, `ftwb3-{1..7}.yaml`

### Output Path Convention

Data outputs follow the config name:
```
data/data_generation_v1/derived_datasets/pt7/
data/data_generation_v1/derived_datasets/pt2-1/
data/data_generation_v1/derived_datasets/ft1-1/
data/data_generation_v1/derived_datasets/ftwb2-15/
```

---

## TODO: pretraining_v1

### Planned Structure
```
src/pretraining_v1/
├── train.py                  # Main training script
├── scripts/
└── utils.py

configs/pretraining_v1/
├── pt1/                      # Multi-task pretraining
├── pt1_single_task/          # PT1-X single task
├── pt2/                      # Two-task combinations
├── pt3/                      # Three-task combinations
└── revision/                 # Seed variations

scripts/pretraining_v1/
└── ...

data/pretraining_v1/
└── experiments/              # Model checkpoints
```

### Key Changes Needed
- Update `dataset_path` in training configs to point to `data/data_generation_v1/derived_datasets/`
- Update `tokenizer_path` to point to `data/data_generation_v1/tokenizers/`

---

## TODO: finetuning_v1

### Planned Structure
```
src/finetuning_v1/
├── train.py                  # Fine-tuning script (may share with pretraining)
└── ...

configs/finetuning_v1/
├── ft1/                      # Single-task fine-tuning
├── ft2/                      # Two-task fine-tuning
├── ftwb1/                    # Fine-tuning with warmup+baseline
├── ftwb2/
└── ...
```

### Key Consideration
- Fine-tuning depends on pretrained checkpoints (upstream)
- May need `upstream_dir` pattern for checkpoint paths

---

## TODO: cka_v1

### Planned Structure
```
src/cka_v1/
├── compute_cka.py
├── visualization/
└── ...

configs/cka_v1/
└── ...
```

---

## Notes

### What stays in src/ root
- `src/utils.py` - Cross-track utilities (dataset loading, init_directory, etc.)
- `src/evaluation.py` - Unified evaluation module
- `src/metrics.py` - Task metric calculations

### Downstream Pattern
Eval and representation extraction are **downstream** of training runs:
- They read from `upstream_dir` (a training output)
- They write to `output_dir` (typically `upstream_dir/downstream/<name>/`)
- They don't need their own tracks
