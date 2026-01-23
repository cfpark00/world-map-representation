# Cross-Seed CKA Infrastructure Setup

**Date**: 2025-11-19 23:03
**Task**: Build infrastructure for 21×21 cross-seed CKA analysis

## Overview

Created complete infrastructure to compute CKA similarity matrices comparing representations across different training seeds for PT1 single-task experiments.

## What Was Built

### 1. New CKA v2 Infrastructure (Clean Implementation)

**Core Modules** (`src/analysis/cka_v2/`):
- `compute_cka.py` - CPU/GPU accelerated CKA computation with centered kernel alignment
- `load_representations.py` - Loads and aligns representations from checkpoint directories
- `experiment_registry.py` - Single source of truth for experiment metadata

**Main Script** (`src/scripts/`):
- `analyze_cka_pair.py` - Analyzes one experiment pair at one layer, produces:
  - `config.yaml` - Reproducible config
  - `cka_timeline.csv` - Per-checkpoint CKA values
  - `cka_timeline.png` - Timeline visualization
  - `summary.json` - Statistics (final, mean, std, min, max)

**Key Design Principles**:
- Hierarchical organization: `{group}/{pair}/layer{X}/`
- No interaction with old CKA code
- Uses existing representation format (checkpoint directories with metadata.json + representations.pt)
- Task-specific prompts: `distance_firstcity_last_and_trans`, `trianglearea_firstcity_last_and_trans`, etc.

### 2. Representation Extraction Configs

**Generated**: 56 configs in `configs/analysis_representation_higher/`
- `seed1/` - 28 configs (7 tasks × 4 layers) for pt1-X_seed1 experiments
- `seed2/` - 28 configs (7 tasks × 4 layers) for pt1-X_seed2 experiments

**Format**: Uses existing `analyze_representations_higher.py` script (not a new one)

### 3. CKA Analysis Configs

**Generated**: 924 configs in `configs/analysis_v2/cka_cross_seed/`
- 231 unique pairs (21 experiments choose 2, upper triangle + diagonal)
- 4 layers each (3, 4, 5, 6)
- 231 × 4 = 924 total configs

**Comparisons**:
- Original (seed 42) vs Original: 28 pairs
- Original vs Seed1: 49 pairs
- Original vs Seed2: 49 pairs
- Seed1 vs Seed1: 28 pairs
- Seed1 vs Seed2: 49 pairs
- Seed2 vs Seed2: 28 pairs

### 4. Execution Scripts

**Location**: `scripts/revision/exp4/` (following existing structure)

**Representation Extraction**:
- `representation_extraction/extract_seed1_representations.sh` - Extract all seed1 (28 jobs)
- `representation_extraction/extract_seed2_representations.sh` - Extract all seed2 (28 jobs)

**CKA Analysis**:
- `cka_analysis/run_all_cka_cross_seed.sh` - Compute all 231 pairs at layer 5

**Documentation**:
- `README.md` - Exp4 overview and workflow

### 5. Config Generators

**Scripts** (`src/scripts/`):
- `generate_repr_configs_seed1_simple.py --seeds 1,2` - Generate representation extraction configs
- `generate_cka_configs_cross_seed.py --seeds 1,2` - Generate CKA comparison configs

## Key Decisions

### 1. Reuse Existing Infrastructure
- Used `src/analysis/analyze_representations_higher.py` (not a new script)
- Used existing representation format (checkpoint dirs with .pt files)
- Followed existing config structure in `configs/analysis_representation_higher/`

### 2. Task-Specific Prompts
- Each task uses its own prompt: `{task}_firstcity_last_and_trans`
- NOT using `distance_firstcity_last_and_trans` for everything
- Matches what original experiments already have

### 3. Organization
- Put scripts in `scripts/revision/exp4/` (not `scripts/analysis_v2/`)
- Removed temporary `scripts/analysis_v2/` directory after moving needed files
- Kept clean CKA v2 library in `src/analysis/cka_v2/`

## Current Status

### ✅ Ready Now
- Original PT1 experiments (pt1-1 through pt1-7) already have representations
- Seed1 experiments (pt1-1_seed1 through pt1-7_seed1) are trained
- All 56 representation extraction configs generated
- All 924 CKA configs generated
- All execution scripts ready

### ⏳ Waiting
- Seed2 experiments need to be trained first
- Training configs exist: `configs/revision/exp4/pt1_single_task_seed/pt1-X/pt1-X_seed2.yaml`

## Next Steps

### Immediate (Seed1 ready)
```bash
# Extract seed1 representations (28 jobs)
bash scripts/revision/exp4/representation_extraction/extract_seed1_representations.sh

# Compute 14×14 or partial 21×21 CKA matrix
bash scripts/revision/exp4/cka_analysis/run_all_cka_cross_seed.sh
```

### After Seed2 Training
```bash
# Extract seed2 representations (28 jobs)
bash scripts/revision/exp4/representation_extraction/extract_seed2_representations.sh

# Re-run to get full 21×21 matrix
bash scripts/revision/exp4/cka_analysis/run_all_cka_cross_seed.sh
```

## Expected Output

### Representation Locations
- Original: `data/experiments/pt1-X/analysis_higher/{task}_firstcity_last_and_trans_lY/`
- Seed1: `data/experiments/revision/exp4/pt1-X_seed1/analysis_higher/{task}_firstcity_last_and_trans_lY/`
- Seed2: `data/experiments/revision/exp4/pt1-X_seed2/analysis_higher/{task}_firstcity_last_and_trans_lY/`

### CKA Results
`data/analysis_v2/cka/pt1_cross_seed/{exp1}_vs_{exp2}/layer{X}/`
- `config.yaml`
- `cka_timeline.csv`
- `cka_timeline.png`
- `summary.json`

### Final Matrix
21×21 symmetric matrix showing CKA similarity between all experiment pairs:
- Diagonal: Self-similarity (1.0)
- Same task, different seeds: Robustness measure
- Different tasks, same seed: Task specificity
- Different tasks, different seeds: Overall dissimilarity

## Files Modified/Created

### Created
- `src/analysis/cka_v2/__init__.py`
- `src/analysis/cka_v2/compute_cka.py`
- `src/analysis/cka_v2/load_representations.py`
- `src/analysis/cka_v2/experiment_registry.py`
- `src/scripts/analyze_cka_pair.py`
- `src/scripts/generate_cka_configs_cross_seed.py`
- `src/scripts/generate_repr_configs_seed1_simple.py`
- `scripts/revision/exp4/representation_extraction/extract_seed1_representations.sh`
- `scripts/revision/exp4/representation_extraction/extract_seed2_representations.sh`
- `scripts/revision/exp4/cka_analysis/run_all_cka_cross_seed.sh`
- `scripts/revision/exp4/README.md`
- `configs/analysis_representation_higher/seed1/` (28 configs)
- `configs/analysis_representation_higher/seed2/` (28 configs)
- `configs/analysis_v2/cka_cross_seed/` (924 configs)
- `docs/cka_v2_infrastructure.md`
- `docs/cross_seed_cka_setup.md`
- `docs/cross_seed_cka_21x21.md`

### Removed
- `scripts/analysis_v2/` (entire directory - superseded by revision/exp4 scripts)

## Technical Details

### CKA Computation
- Uses linear kernel: K = X @ X.T
- Centered kernel alignment (centered=True)
- GPU-accelerated when available
- Computes: CKA = <K, L>_F / (||K||_F * ||L||_F)

### Representation Format
- Loaded from checkpoint directories
- Format: `metadata.json` + `representations.pt`
- Handles flattened representations automatically
- Aligns cities across experiments
- Applies city filter: `region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$`

### Testing
- Tested CKA v2 infrastructure on pt1-1 vs pt1-2, layer 5
- Processed 41 checkpoints, 4413 cities
- Final CKA: 0.3924, Mean: 0.3923 ± 0.0715
- Confirmed output format matches design

## Documentation

See:
- `docs/cka_v2_infrastructure.md` - Clean CKA v2 design and usage
- `docs/cross_seed_cka_21x21.md` - Cross-seed analysis overview
- `scripts/revision/exp4/README.md` - Exp4 workflow
