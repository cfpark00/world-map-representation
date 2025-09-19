# Development Log - 2025-09-18 17:20
## Multi-task Fine-tuning Setup and Evaluation Fixes

### Summary
Created multi-task fine-tuning infrastructure (ft2) combining distance, angle, and compass tasks with Atlantis cities. Fixed critical evaluation plotting bugs including missing checkpoint-0 and implemented configurable plot scaling.

### Major Changes

#### 1. Atlantis Required Task Analysis
- Identified missing atlantis_required support for circlecount, randring, and randomwalk
- Removed incomplete atlantis_required implementations for these three complex tasks
- Clarified task behavior with Atlantis cities:
  - 8 tasks unchanged when Atlantis not in prompt (distance, compass, angle, trianglearea, crossing, perimeter, center, inside)
  - 4 tasks changed even without Atlantis in prompt (nearest_neighbor, circlecount, randomwalk, randring)

#### 2. Multi-task Fine-tuning Dataset (ft2)
**Created combine_multitask_ft2 infrastructure:**
- Config: `configs/data_generation/combine_multitask_ft2.yaml`
  - 20,000 samples from distance_1M_no_atlantis
  - 100,000 samples from distance_100k_atlantis_required
  - 100,000 samples from angle_100k_atlantis_required
  - 100,000 samples from compass_100k_atlantis_required
  - Total: 320,000 training samples
- Script: `scripts/data_generation/merge/combine_multitask_ft2.sh`
- Properly maintains train/val/test split structure with proportional sampling

#### 3. Training Configuration for ft2
**Created ft2 training setup:**
- Config: `configs/training/ft_m1_10M_ft2.yaml`
  - Uses multitask_ft2 dataset
  - 10 epochs (reduced from 30)
  - Checkpoint saves every 5% of training
  - Same model architecture and base checkpoint as ft1
- Script: `scripts/training/ft_m1_10M_ft2.sh`

#### 4. Evaluation Configuration for ft2
**Created comprehensive evaluation configs in `configs/eval/m1_10M_ft2/`:**
- Atlantis-required tasks: atlantis_distance, atlantis_angle, atlantis_compass, atlantis_trianglearea
- No-Atlantis tasks: distance, angle, compass, trianglearea
- Multi-task evaluation: multi_task
- Script: `scripts/eval/eval_m1_10M_ft2_all.sh` runs all 9 evaluations

#### 5. Plot Scaling Configuration
**Implemented configurable plot scaling:**
- Added `plot_log_scale` parameter to evaluation configs (default: true)
- Set `plot_log_scale: false` for all ft2 configs to use linear x-axis
- Modified `save_training_plots()` in `src/utils.py` to accept config parameter
- Updated plotting logic to conditionally use log or linear scale based on config

#### 6. Fixed Checkpoint-0 Evaluation Bug
**Fixed critical bug where checkpoint-0 wasn't being plotted:**
- Root cause: Function was skipping step=0 entries assuming they were loaded from file
- Added `loaded_checkpoint_0` flag to track if checkpoint-0 was loaded from file
- Only skip step=0 entries in log_history if actually loaded from file
- Now checkpoint-0 properly appears in evaluation plots

### Technical Details

#### Dataset Merging
- `combine_datasets.py` properly handles multi-split datasets
- Proportional sampling maintains train/val/test ratios
- Shuffling applied to each split independently

#### Nearest Neighbor Task Clarification
- `must_include` strategy: Query cities from Atlantis, pool includes all cities
- `all_pairs` strategy: Both query and pool from same filtered set
- Atlantis cities can appear in results even when not in query

#### Removed Implementations
- Deleted `configs/data_generation/randomwalk_100k_atlantis_required.yaml`
- Removed atlantis_required command from `create_randomwalk_datasets.sh`
- Confirmed circlecount and randring never had atlantis_required configs

### Files Modified
- Created 9 eval configs in `configs/eval/m1_10M_ft2/`
- Created `configs/data_generation/combine_multitask_ft2.yaml`
- Created `configs/training/ft_m1_10M_ft2.yaml`
- Created `scripts/data_generation/merge/combine_multitask_ft2.sh`
- Created `scripts/training/ft_m1_10M_ft2.sh`
- Created `scripts/eval/eval_m1_10M_ft2_all.sh`
- Modified `src/utils.py` - Added config parameter and plot scaling logic
- Modified `src/eval/evaluate_checkpoints.py` - Pass config to plotting function

### Next Steps
- Run ft2 training and evaluation
- Consider implementing atlantis_required for circlecount, randring, randomwalk
- Analyze ft2 performance vs ft1 (single-task vs multi-task fine-tuning)