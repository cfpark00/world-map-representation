# Session Log: Atlantis Fine-tuning Datasets and Training Setup
**Date**: 2025-09-01
**Time**: 21:10 EDT  
**Topic**: Creating Atlantis fine-tuning infrastructure and analysis

## Summary
Created dataset generation scripts and training configurations for fine-tuning experiments with virtual "Atlantis" cities, focusing on studying how models adapt to new geographic regions.

## Key Accomplishments

### 1. Dataset Creation Scripts
Created two specialized scripts for Atlantis fine-tuning datasets:

#### `src/data_processing/create_atlantis_inter_dataset.py`
- Generates pairs with **only inter-Atlantis cities** (both cities from Atlantis)
- Maximum 4,950 unique pairs possible (100 cities)
- Default splits: 4,000 train, 128 val, 500 test
- Auto-detects offset from world cities (continues from row_id 5075)

#### `src/data_processing/create_atlantis_cross_dataset.py`
- Generates pairs that **include at least one Atlantis city**
- Mix of inter-Atlantis and Atlantis-World pairs
- Default 10% inter-Atlantis ratio (adjustable)
- Default splits: 100,000 train, 128 val, 10,000 test

### 2. Row ID Correction
- Fixed Atlantis city IDs to be **contiguous** with world cities
- World cities: row_ids 0-5074
- Atlantis cities: row_ids 5075-5174
- Avoids inconsistent digit counts (was 10000+, now proper continuation)

### 3. Training Configurations
Created 4 fine-tuning configs:
- `atlantis_inter_finetune.yaml` - batch size 512, 50 epochs
- `atlantis_inter_finetune_bs64.yaml` - batch size 64 variant
- `atlantis_cross_finetune.yaml` - batch size 512, 10 epochs  
- `atlantis_cross_finetune_bs64.yaml` - batch size 64 variant

All configs:
- Load from final checkpoint of `dist_100k_1M_20epochs`
- Use lower learning rates for fine-tuning (1e-4 or 5e-5)
- Distance prediction task type

### 4. Dataset Generation
Successfully generated both datasets:
- `outputs/datasets/atlantis_inter_100k_42/` - 4,628 total pairs
- `outputs/datasets/atlantis_cross_100k_42/` - 110,128 total pairs

### 5. Training and Analysis
- Ran `atlantis_cross_finetune` experiment
- Analysis showed significant degradation in geographic representations:
  - Initial (pre-fine-tuning): R² 0.956/0.923, error 993 km
  - After fine-tuning: R² 0.824/0.831, error 2230 km
  - Drop of 0.132 in longitude R², 0.092 in latitude R²
- Shows that virtual regions disrupt learned geography

### 6. Training Script Enhancement
Modified `src/training/train.py` to:
- **Always evaluate at step 0** (both from scratch and checkpoints)
- Save initial model and metrics to `checkpoint-0/`
- Shows chance level for new models or starting point for fine-tuning
- Provides complete evaluation history

## Files Created/Modified

### Created:
- `src/data_processing/create_atlantis_inter_dataset.py`
- `src/data_processing/create_atlantis_cross_dataset.py`
- `configs/atlantis_inter_finetune.yaml`
- `configs/atlantis_inter_finetune_bs64.yaml`
- `configs/atlantis_cross_finetune.yaml`
- `configs/atlantis_cross_finetune_bs64.yaml`
- `outputs/datasets/atlantis_inter_100k_42/` (dataset)
- `outputs/datasets/atlantis_cross_100k_42/` (dataset)

### Modified:
- `src/training/train.py` - Added initial evaluation at step 0

## Technical Details

### Dataset Statistics
- Inter-Atlantis: 4,950 maximum unique pairs
- Cross-dataset actual distribution:
  - Train: 4.5% inter-Atlantis, 95.5% cross pairs
  - Validation: 7.8% inter-Atlantis, 92.2% cross pairs
  - Test: 4.2% inter-Atlantis, 95.8% cross pairs

### Key Findings
1. Fine-tuning on virtual geographic regions causes **catastrophic forgetting**
2. Model quickly loses accurate world geography representations
3. Performance degrades most in first 200 steps, then stabilizes
4. Even mixing with real world pairs doesn't prevent degradation

## Next Steps
- Run inter-Atlantis only fine-tuning experiment
- Compare batch size effects (64 vs 512)
- Test recovery strategies or regularization approaches
- Consider creating multiple Atlantis regions in different oceans