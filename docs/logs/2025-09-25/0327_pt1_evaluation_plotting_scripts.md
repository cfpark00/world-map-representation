# Development Log - 2025-09-25 03:27 - PT1 Evaluation Plotting Scripts

## Summary
Created visualization scripts for PT1 experiment evaluation metrics, focusing on task performance across training checkpoints.

## Tasks Completed

### 1. PT1 Experiment Directory Location
- Found PT1 experiment directory at `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/data/experiments/pt1/`
- Explored structure of evals/ directory containing evaluation results for all tasks
- Identified data format: JSONL files with metrics for each checkpoint

### 2. Individual Task Performance Plotting Script
**File**: `scratch/pt1_eval_plot/plot_task_performances.py`

Initial issues addressed:
- Incorrectly treating all metrics as accuracy (0-1 range)
- Not sorting checkpoints, causing zigzag lines in plots
- Including Atlantis tasks when not needed
- Misunderstanding metric types

Corrections made:
- Properly distinguished between:
  - **Accuracy tasks** (crossing, compass, inside): Calculate percentage of correct predictions
  - **Error tasks** (distance, trianglearea, angle, perimeter): Calculate mean absolute error (lower is better)
- Added logarithmic x-axis for checkpoint numbers
- Added logarithmic y-axis for error metrics
- Sorted checkpoints before plotting to avoid line artifacts
- Excluded Atlantis tasks as requested
- Created 2x4 subplot grid for 7 core tasks

### 3. Combined Metrics Plotting Script
**File**: `scratch/pt1_eval_plot/plot_all_metrics_combined.py`

Features implemented:
- Single plot showing all 7 metrics together
- Twin y-axes:
  - Left axis: Error metrics with log scale
  - Right axis: Accuracy metrics (0-100%)
- Shared logarithmic x-axis for training checkpoints
- Distinct color schemes for better visibility
- Legend positioned outside plot area

Iterative improvements made:
- Changed colors multiple times to ensure distinctiveness
- Increased font sizes: labels (18pt), tick labels (18pt), legend (14pt)
- Changed all axis labels and tick labels to black
- Removed title for cleaner appearance
- Fixed legend to show all task names
- Changed all lines to solid style

## Key Findings from PT1 Evaluation

### High-Performance Tasks (Accuracy)
- Compass: 99.10%
- Inside: 97.97%
- Crossing: 97.66%

### Low-Error Tasks (Mean Absolute Error)
- Angle: 0.96
- Distance: 2.38

### Higher-Error Tasks
- Perimeter: 21.52
- Trianglearea: 1189.93

## Technical Details

### Metric Calculation Understanding
- Read `src/evaluation.py` and `src/metrics.py` to understand proper metric computation
- Distance, trianglearea, angle, perimeter use absolute error metrics from `src/metrics.py`
- Crossing, compass, inside are binary accuracy tasks (1.0 = correct, 0.0 = incorrect)

### Data Processing
- Parsed JSONL files from `evals/*/eval_data/detailed_results_checkpoint-*.jsonl`
- Extracted metric values and computed appropriate statistics per task type
- Sorted checkpoints numerically to ensure proper temporal ordering

## Files Created/Modified
- `/scratch/pt1_eval_plot/plot_task_performances.py` - Individual subplot for each task
- `/scratch/pt1_eval_plot/plot_all_metrics_combined.py` - All metrics on single plot with twin axes
- `/scratch/pt1_eval_plot/task_performances.png` - Output visualization (individual)
- `/scratch/pt1_eval_plot/all_metrics_combined.png` - Output visualization (combined)

## Notes
- Scripts are in scratch/ directory as temporary analysis tools
- Both scripts properly handle the different metric types and scales
- Visualizations clearly show training dynamics and final performance across all tasks