# Session Log: 2025-09-02 00:13 - Checkpoint-0 Evaluation and Representation Analysis

## Summary
Fixed checkpoint-0 evaluation/plotting issue in training script and ran comprehensive representation analyses on mixed distance/cross-entropy finetuned model with various Atlantis configurations.

## Tasks Completed

### 1. Fixed Checkpoint-0 Evaluation and Plotting
**Problem:** Checkpoint-0 was being saved and evaluated but not appearing in training plots generated during training.

**Root Cause:** The `trainer.state.log_history` gets reset when training starts, so the manually inserted checkpoint-0 entry was lost.

**Solution Implemented:**
- Modified `src/training/train.py` to:
  - Run generation-based evaluation for checkpoint-0 (lines 122-128)
  - Add checkpoint-0 metrics to log_history (lines 134-142)
  - Save initial plot after checkpoint-0 evaluation (line 172)
  
- Modified `src/utils.py` `save_training_plots()` function to:
  - Always read checkpoint-0 metrics from saved JSON file (lines 869-881)
  - Prepend checkpoint-0 data to plots regardless of log_history state
  - Added json import (line 9)

**Files Modified:**
- `/src/training/train.py`
- `/src/utils.py`

### 2. Ran Comprehensive Representation Analyses

Executed four different representation analyses on the mixed_dist20k_cross100k_finetune model to understand how the model represents Atlantis cities vs real geographic knowledge.

**Model Analyzed:** `/outputs/experiments/mixed_dist20k_cross100k_finetune_/`

**Analysis Configurations (all with 5000 probe cities, 3000 for training):**

1. **Baseline (no modifications)**
   - Directory: `dist_layers3_4_probe5000_train3000/`
   - Final R²: Lon 0.938, Lat 0.912
   - Distance error: 1165 km
   - Best overall performance

2. **Atlantis as eval only**
   - Directory: `dist_layers3_4_probe5000_train3000_plus100eval/`
   - 100 Atlantis cities added to test set only
   - Final R²: Lon 0.856, Lat 0.847
   - Distance error: 1497 km

3. **Africa and Atlantis concatenated**
   - Directory: `dist_layers3_4_probe5000_train3000_plus100concat/`
   - Atlantis cities included in training pool
   - Final R²: Lon 0.905, Lat 0.871
   - Distance error: 1427 km

4. **Africa removed (control)**
   - Directory: `dist_layers3_4_probe5000_train3000_noAfrica/`
   - Africa excluded from probe training
   - Final R²: Lon 0.881, Lat 0.796
   - Distance error: 1583 km
   - Worst latitude R²

**Key Findings:**
- Model maintains strong geographic representations for real world cities (baseline R² > 0.9)
- Including Atlantis in training pool improves probe performance for Atlantis cities
- Removing Africa significantly hurts latitude predictions
- The model has learned separate representation spaces for real vs fictional geography

## Technical Notes

### Checkpoint-0 Evaluation Issue
The HuggingFace Trainer doesn't have a built-in option for initial evaluation that triggers callbacks. The fix ensures checkpoint-0 metrics are always loaded from disk when generating plots, making it robust against log_history resets.

### Analysis Script Parameters Used
```bash
--exp_dir /path/to/experiment
--cities_csv outputs/datasets/cities_100k_plus_seed42.csv
--task-type distance
--n_probe_cities 5000
--n_train_cities 3000
--layers 3 4  # default
```

Additional flags for different configurations:
- `--additional-cities` for Atlantis CSV
- `--additional-labels` for region mapping JSON
- `--concat-additional` to include in training pool
- `--remove-label-from-train` to exclude regions

## Files/Directories Created
- Analysis outputs in `/outputs/experiments/mixed_dist20k_cross100k_finetune_/analysis/`:
  - `dist_layers3_4_probe5000_train3000/`
  - `dist_layers3_4_probe5000_train3000_plus100eval/`
  - `dist_layers3_4_probe5000_train3000_plus100concat/`
  - `dist_layers3_4_probe5000_train3000_noAfrica/`
- Each contains: `representation_dynamics.csv`, `dynamics_plot.png`, `final_world_map.png`, `world_map_evolution.gif`

## Next Steps
The representation analysis results show clear evidence of separate geographic spaces for real vs fictional locations, supporting the catastrophic forgetting hypothesis when models are fine-tuned on fictional geography.