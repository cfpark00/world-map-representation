# PT1 Training Dynamics Visualization - 2025-11-21 02:16

## Summary
Created comprehensive vertical training dynamics plots for all PT1 experiments (PT1-1 through PT1-7), showing loss curves, task metrics, coordinate prediction R², and distance error across training.

## Context
Building on previous work to visualize training dynamics, extended from 2-panel to 3-panel layout and refined formatting for publication-quality figures.

## Work Completed

### 1. Three-Panel Vertical Plot Structure
Created `scratch/formation_dynamics/plot_pt1_all.py` to generate training dynamics plots with:
- **Top panel**: Training and validation loss curves (log-log scale)
- **Middle panel**: Task-specific metric (left y-axis) and Mean Coordinate R² (right y-axis)
- **Bottom panel**: Mean distance error to ground truth coordinates (log-log scale)

### 2. Data Sources
- Training/validation loss: `trainer_state.json` from checkpoints
- Task metrics: Evaluation data from `data/experiments/pt1/evals/{task}/eval_data/*.jsonl`
- Coordinate R² and distance error: `representation_dynamics.csv` from analysis_higher

### 3. Per-Experiment Customization
Implemented adaptive y-axis ranges and ticks for each PT1 experiment:
- **PT1-1 (distance)**: Loss ylim [0.7, 1.1], task metric ticks [3, 10, 30, 100, 300, 1000]
- **PT1-2 (trianglearea)**: Loss ylim [0.9, 1.15], task metric up to 300,000 with custom ticks
- **PT1-3 (angle)**: Loss ylim [0.8, 1.1], task metric up to 180 degrees
- **PT1-4 (compass)**: Loss ylim [0.6, 0.9], accuracy task (linear scale 0-1)
- **PT1-5 (inside)**: Loss ylim [0.85, 1.0], accuracy task
- **PT1-6 (perimeter)**: Loss ylim [0.8, 1.1], task metric up to 2000
- **PT1-7 (crossing)**: Loss ylim [0.8, 1.0], accuracy task (failed to learn, R²≈-0.075)

### 4. Formatting Refinements
Applied multiple rounds of formatting improvements:
- Thick black spines (3px width)
- Large tick labels (size 28)
- No grid lines (changed from `seaborn-v0_8-whitegrid` to `seaborn-v0_8-white`)
- Removed x-axis ticks/labels from top and middle panels (only show on bottom)
- Thicker plot lines (linewidth=5.0) for all metrics
- Fixed distance error yticks to [50, 100, 500] for consistency across experiments
- Color-coded y-axis labels (green for task metric, red for R²)

### 5. Key Debugging
- **PT1-2 ylim issue**: Data extended to 2.9M but requested ylim was 40K, matplotlib silently ignored. Fixed by setting ylim to 300K to capture visible data range.
- **Data filtering**: Ensured all representation analysis data is filtered to training step range (max 328,146 steps).

## Output
Generated 7 plots saved to `scratch/formation_dynamics/figures/`:
- `pt1-1_vertical.png` through `pt1-7_vertical.png`

## Files Modified
- `scratch/formation_dynamics/plot_pt1_all.py` - Main plotting script

## Technical Details
- Uses matplotlib GridSpec with height ratios [2, 1, 1]
- Twin axes (twinx) for dual y-axis on middle panel
- Log scale on both axes for most plots (except accuracy tasks use linear y-scale)
- ScalarFormatter to avoid scientific notation
- Data loaded from multiple sources (trainer_state.json, eval jsonl files, representation_dynamics.csv)

## Key Findings
All PT1 experiments successfully converged except PT1-7 (crossing):
- PT1-1 to PT1-6: Final R² ranging from 0.61 (compass) to 0.99 (angle, perimeter)
- PT1-7 (crossing): Failed to learn coordinate representations (R² = -0.075)
- Distance error stabilizes around 50-100 units for successful experiments
