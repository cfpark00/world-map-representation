# Session Log: Atlantis Evaluation and Analysis Script Enhancements
**Date**: 2025-09-01
**Time**: 22:31 EDT
**Topic**: Enhanced representation analysis for evaluating Atlantis and probe generalization

## Summary
Major enhancements to the representation analysis script to properly evaluate how well models learn Atlantis representations. Added controls to test probe generalization and discovered that Atlantis representations are poorly positioned despite extensive training.

## Key Accomplishments

### 1. Analysis Script Enhancements (`src/analysis/analyze_representations.py`)

#### New Command-Line Arguments
- **`--additional-cities`** (renamed from `--additional-eval`): Path to additional cities CSV file
- **`--concat-additional`**: Flag to concatenate additional cities to main pool (can appear in training) vs test-only
- **`--additional-labels`**: JSON file with additional country-to-region mappings (e.g., `{"XX0": "Atlantis"}`)
- **`--remove-label-from-train`**: Exclude specific region from probe training while maintaining 3000 samples

#### Visual Improvements
- Changed Atlantis color from indigo to hot pink (#FF1493) for better visibility
- Fixed gray dots to only show training set cities (not test set)
- Directory naming includes suffixes: `_plus100eval`, `_plus100concat`, `_noAfrica`

#### Implementation Details
- Smart sampling when excluding regions: maintains 3000 training samples from non-excluded regions
- Proper handling of additional cities in both concatenation and test-only modes
- Fixed multiple redundant `import json` statements (was importing 3 times!)
- Added checkpoint-0 saving in training script for initial model state

### 2. Configuration Files Created

#### Low Learning Rate Config
- Created `configs/rw200_100k_1m_20epochs_pt1_lowlr.yaml`
- 10x lower learning rate (3e-5 vs 3e-4) to prevent catastrophic forgetting
- Based on findings from catastrophic forgetting report

#### Mixed Dataset Configs
- Created `configs/mixed_dist20k_cross100k_finetune.yaml`
- 20k original distance + 100k Atlantis cross samples
- Created mixing script `src/data_processing/mix_datasets.py`

#### Supporting Files
- `configs/atlantis_region_mapping.json`: Maps XX0 to "Atlantis" region

### 3. Dataset Management

#### Fixed Naming Issue
- Renamed `atlantis_inter_100k_42` to `atlantis_inter_4k_42` (actually has 4k samples, not 100k)
- Updated all config references

#### Created Mixed Dataset
- `mixed_dist20k_cross100k_42`: 20k original + 100k Atlantis cross
- 120k total training samples to prevent catastrophic forgetting

### 4. Experiments and Analysis

#### Ran Multiple Evaluations on `mixed_dist20k_cross100k_finetune`
1. **Atlantis test-only**: R² = 0.856/0.847, error = 1497 km
2. **Atlantis in full pool**: R² = 0.905/0.871, error = 1427 km  
3. **Africa excluded control**: R² = 0.881/0.796, error = 1583 km

#### Key Finding
Atlantis representations are poorly positioned despite 100k training samples. The Africa control proved it's not a probe generalization issue - the model genuinely learned weak representations for the isolated fictional continent.

### 5. Documentation
- Created `reports/2025-09-02-atlantis-evaluation-analysis.md`
- Internal documentation of experiments and findings
- Explanation of why isolated fictional locations are harder to embed

## Code Changes Summary

### Modified Files
- `src/analysis/analyze_representations.py`: Major enhancements for flexible evaluation
- `src/training/train.py`: Added checkpoint-0 saving
- `configs/atlantis_inter_finetune.yaml`: Fixed dataset path
- `configs/atlantis_inter_finetune_bs64.yaml`: Fixed dataset path

### New Files
- `src/data_processing/mix_datasets.py`: Dataset mixing utility
- `configs/rw200_100k_1m_20epochs_pt1_lowlr.yaml`: Low LR config
- `configs/mixed_dist20k_cross100k_finetune.yaml`: Mixed dataset training
- `configs/atlantis_region_mapping.json`: Region mapping for Atlantis
- `reports/2025-09-02-atlantis-evaluation-analysis.md`: Analysis documentation

### Datasets Created
- `atlantis_inter_4k_42`: Renamed from incorrect 100k name
- `mixed_dist20k_cross100k_42`: Mixed training dataset
- `mixed_dist20k_rw100k_42`: Initially created but wrong mix (deleted)

## Git Commits
- "Add low learning rate config for random walk fine-tuning"

## Next Steps
- The Atlantis evaluation revealed that isolated fictional locations are challenging for the model
- Consider experiments with fictional continents that have more geographic context
- Investigate whether adding intermediate waypoint cities could help