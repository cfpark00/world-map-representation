# Multi-task Geometry Datasets and Evaluation System

## Date: 2025-09-14 19:14

## Summary
Created comprehensive multi-task training infrastructure for 4 geometry tasks (distance, random walk, triangle area, angle), including dataset generation scripts, evaluation metrics, and visualization systems.

## Major Accomplishments

### 1. Created Three New Dataset Types
- **Random Walk Dataset** (`create_randomwalk_dataset.py`)
  - Format: `rw(max_dist,chain_len)=c_XXXX,c_XXXX,...`
  - Samples max_distance from [min_max_distance, max_max_distance] per row
  - Samples chain_length from [min_chain_length, max_chain_length] per row
  - Always allows city revisiting (no restrictions)

- **Triangle Area Dataset** (`create_trianglearea_dataset.py`)
  - Format: `triarea(c_XXXX,c_XXXX,c_XXXX)=AREA`
  - Calculates area using Shoelace formula
  - Generates random triples with degenerate filtering

- **Angle Dataset** (`create_angle_dataset.py`)
  - Format: `angle(c_XXXX,c_XXXX,c_XXXX)=DEGREES`
  - Calculates angle at center city in degrees (0-180)
  - Handles edge cases with proper validation

### 2. Configuration Files
Created 6 config files for new tasks:
- `randomwalk_1M_no_atlantis_pad.yaml` / `randomwalk_1M_with_atlantis_pad.yaml`
- `trianglearea_1M_no_atlantis_pad.yaml` / `trianglearea_1M_with_atlantis_pad.yaml`
- `angle_1M_no_atlantis_pad.yaml` / `angle_1M_with_atlantis_pad.yaml`

Key parameters for random walk:
- `min_max_distance: 50, max_max_distance: 500`
- `min_chain_length: 5, max_chain_length: 20`

### 3. Multi-task Dataset Combination
- Created `combine_multitask_4M_no_atlantis_pad.yaml` to merge all 4 tasks
- Total 4M samples (1M each from distance, randomwalk, trianglearea, angle)
- Full concatenation with shuffling enabled

### 4. Evaluation System Overhaul

#### Metric Standardization
Fixed inconsistent metrics - all now use **error metrics** (lower is better):

- **Distance**: Absolute error in km (0-4025 km)
  - Max possible: √(3600² + 1800²) ≈ 4025 km (with 10x coordinate scaling)
  - Format errors get 4025 km

- **Random Walk**: Combined error (0-1)
  - Formula: `(chain_length_error + invalid_transition_ratio) / 2`
  - Format errors get 1.0

- **Triangle Area**: Absolute error in square units (0-3,240,000)
  - Max possible: 3600 × 1800 / 2 = 3,240,000
  - Format errors get maximum

- **Angle**: Absolute error in degrees (0-180)
  - Format errors get 180°

#### Plotting Improvements
- **X-axis**: Log scale for all plots (min=100 to avoid log(0))
- **Y-axis**:
  - Log scale for distance & triangle area (large ranges)
  - Linear scale for random walk & angle (bounded 0-1 and 0-180)
- Separate plots for each task in `output_dir/summary/`
- Proper reference lines and boundaries for each metric

### 5. Bug Fixes
- Fixed `city_id` vs `row_id` column mismatch in evaluation
- Fixed euclidean_distance array handling in random walk generation
- Added proper error handling for unparseable outputs
- Fixed multi-task metric key access in training script

### 6. Documentation Updates
- Strengthened `repo_usage.md` bash script guidelines:
  - ONLY shebang and commands allowed
  - NO comments, echo statements, or formatting
  - Scripts exist SOLELY for reproducibility

### 7. Training Configuration
Created `train_multitask_4M_no_atlantis_15epochs_lowerlr_pad.yaml`:
- Same model architecture as single-task (per user request)
- `max_sequence_length: 224` to handle longer random walks
- Added `randomwalk` section for evaluation config

## Key Design Decisions

1. **Metric Consistency**: All tasks now report errors (0=perfect, higher=worse) instead of mixing scores and errors

2. **Format Error Handling**: Format errors always get maximum penalty for their task type

3. **Coordinate Scaling**: Remembered that coordinates are scaled 10x in data generation, affecting all distance calculations

4. **RoPE Position Encoding**: Confirmed Qwen2 uses RoPE (relative position encoding), not absolute, so variable-length sequences work well

5. **Bash Script Minimalism**: Enforced strict no-comments, no-echo policy for reproducibility scripts

## Files Created/Modified

### New Python Scripts
- `src/data_processing/create_randomwalk_dataset.py`
- `src/data_processing/create_trianglearea_dataset.py`
- `src/data_processing/create_angle_dataset.py`

### New Configs
- 6 dataset generation configs (2 per task type)
- `combine_multitask_4M_no_atlantis_pad.yaml`
- `train_multitask_4M_no_atlantis_15epochs_lowerlr_pad.yaml`

### New Bash Scripts
- `scripts/data_generation/create_randomwalk_datasets_pad.sh`
- `scripts/data_generation/create_trianglearea_datasets_pad.sh`
- `scripts/data_generation/create_angle_datasets_pad.sh`
- `scripts/data_generation/create_all_geometry_datasets_pad.sh`
- `scripts/data_generation/combine_multitask_4M_no_atlantis.sh`
- `scripts/training/train_multitask_4M_no_atlantis_15epochs_lowerlr_pad.sh`

### Modified Files
- `src/utils.py` - Extensive evaluation and plotting updates
- `src/training/train.py` - Multi-task metric printing
- `docs/repo_usage.md` - Stronger bash script guidelines

## Next Steps
- Run full multi-task training with 4M dataset
- Monitor convergence across all 4 task types
- Analyze if model can handle multiple geometry tasks simultaneously
- Consider task-specific loss weighting if needed

## Notes
User was particularly insistent about:
- Keeping model architecture identical to single-task
- Removing ALL comments/echoes from bash scripts
- Using consistent error metrics (not mixing scores/errors)
- Proper maximum values for format errors based on actual coordinate ranges