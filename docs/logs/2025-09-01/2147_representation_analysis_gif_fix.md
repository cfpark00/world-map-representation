# Representation Analysis GIF Fix
Date: 2025-09-01 21:47

## Summary
Fixed a critical bug in the representation analysis script that was causing GIF animations to only show 2 frames instead of all checkpoints, making it impossible to see the actual dynamics of the learned representations.

## Tasks Completed

### 1. Ran Representation Analysis on Multiple Experiments
- **atlantis_inter_finetune**: Analyzed with 50 probe cities and 40 training cities using Atlantis dataset
  - Results showed negative R² values throughout training (longitude: -1.22, latitude: -0.50)
  - Indicates model doesn't learn meaningful geographic representations for fictional cities
  
- **atlantis_cross_finetune_bs64**: Similar analysis with Atlantis cities
  - Also showed negative R² values (longitude: -1.27, latitude: -1.23)
  - Minimal improvement over training
  
- **dist_100k_1M_200epochs_bs128**: Analysis with real world cities (100k+ population)
  - Strong positive R² values (longitude: 0.977, latitude: 0.979)
  - Distance error improved from 1701 km to 598 km
  - Model converged to stable representation by ~20% through training

### 2. Discovered and Fixed GIF Animation Bug
- **Problem Identified**: User noticed GIF showed no movement despite R² values oscillating between 0.973-0.984
- **Root Cause**: Found filtering logic in `analyze_representations.py` lines 579-581:
  ```python
  return_preds = (len(checkpoint_dirs) <= 20 or 
                step % max(1, checkpoint_dirs[-1][0] // 10) == 0 or 
                step == checkpoint_dirs[0][0] or 
                step == checkpoint_dirs[-1][0])
  ```
  - For experiments with >20 checkpoints, only saved first and last checkpoint predictions
  - This resulted in 2-frame GIFs showing no dynamics

- **Fix Applied**: Changed to always save predictions for all checkpoints:
  ```python
  return_preds = True
  ```

- **Impact**: 
  - CSV and dynamics plots were never affected (always used all checkpoints)
  - GIF now properly shows all 24-25 frames with visible oscillations and dynamics
  - Can now see both initial dramatic learning phase and later subtle fine-tuning

### 3. Key Insights from Analysis
- Real world models achieve stable geographic representations early in training (~20% through)
- After convergence, representations oscillate slightly but remain stable
- Atlantis (fictional) cities never achieve positive R² values, suggesting model can't generalize to arbitrary coordinate systems

## Files Modified
- `/src/analysis/analyze_representations.py`: Removed restrictive frame filtering

## Output Locations
Analysis results saved to:
- `/outputs/experiments/atlantis_inter_finetune/analysis/`
- `/outputs/experiments/atlantis_cross_finetune_bs64/analysis/`
- `/outputs/experiments/dist_100k_1M_200epochs_bs128/analysis/`

Each contains:
- `representation_dynamics.csv`: Full metrics for all checkpoints
- `dynamics_plot.png`: R² evolution plot
- `world_map_evolution.gif`: Animated visualization (now with all frames)
- `final_world_map.png`: Final state visualization