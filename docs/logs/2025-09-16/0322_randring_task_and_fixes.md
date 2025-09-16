# Development Log - 2025-09-16 03:22 - RandRing Task and System Fixes

## Summary
Created a new stochastic geometric task (randring), fixed multiple system issues, improved analyze_representations.py, and verified unit systems across all data generation processes.

## Major Tasks Completed

### 1. Created RandRing Stochastic Task
- **File**: `src/data_processing/create_randring_dataset.py`
- **Grammar**: `randring(c_ID,r=MIN,R=MAX,n=NUM)=c_ID1,c_ID2,...`
- **Config**: `configs/data_generation/randring_1M_no_atlantis.yaml`
- **Features**:
  - Samples n cities randomly from an annulus (ring) around a center city
  - Inner radius r: 10-450 units
  - Outer radius R: 50-500 units (R > r enforced)
  - Sample count n: 1-10 cities
  - Returns all available cities if fewer than n exist in the ring
  - Second stochastic task after randomwalk
- **Script**: `scripts/data_generation/single_tasks/create_randring_datasets.sh`

### 2. Fixed analyze_representations.py
- **Changed**: Removed `n_probe_cities` requirement
- **New**: Direct specification of `n_train_cities` and `n_test_cities`
- **Logic**: Simplified 4-step sampling:
  1. Get train candidates by pattern
  2. Sample train cities (error if insufficient)
  3. Get test candidates excluding train
  4. Sample test cities (error if insufficient)
- No train-test overlap enforced

### 3. Script Generation Improvements
- Created `scripts/analysis_representations/run_all_multi_pretrain_30ep.sh`
- Hardcoded all 17 YAML config paths for batch analysis
- Follows minimal bash script convention (no comments/echo)

### 4. System-wide Fixes

#### Seed Management
- **Added**: `seed_all()` function in `src/utils.py`
- **Seeds**: Python random, NumPy, PyTorch, CUDA, CuDNN
- **Sets**: Deterministic mode and PYTHONHASHSEED
- **Updated**: `train.py` to use `seed_all()` instead of partial seeding

#### Dataset Generation Fixes
- **Fixed**: CircleCount JSON serialization (numpy int64 → Python int/float)
- **Fixed**: Center dataset 0D array issue with `np.atleast_1d()`
- **Fixed**: Print order in center dataset (now prints "Loaded X cities" immediately)

#### Training Config Fixes
- **Fixed**: RandomWalk training config missing eval section
- **Added**: Required `cities_csv` path for randomwalk evaluation

### 5. Unit System Verification
Verified all 12 tasks use consistent units:

**Raw coordinate units**:
- Distance, Center, Perimeter, CircleCount (radius), RandomWalk (max_dist), RandRing (r/R)
- Triangle Area (units squared)

**Meaningful units**:
- Angle (degrees 0-180)
- Compass (categorical: N/NE/E/SE/S/SW/W/NW)

**Boolean outputs**:
- Crossing, Inside

**ID outputs**:
- Nearest Neighbor, Center (city IDs)

### 6. Local Bias Analysis
Identified tasks with true local sampling bias:
- **Constrained sampling**: Nearest Neighbor, Random Walk, RandRing, CircleCount
- **Random sampling**: All others (Distance, Triangle Area, Angle, Compass, Crossing, Inside, Perimeter, Center)

Note: Center samples cities randomly but finds locally-biased result (city near center of mass).

### 7. Standardization
- Ensured all datasets use `leading_zeros: true` and `n_id_digits: 4`
- Fixed randring config to match standard pattern
- Confirmed 1M samples as standard dataset size

## Files Created/Modified

### Created
- `/src/data_processing/create_randring_dataset.py`
- `/configs/data_generation/randring_1M_no_atlantis.yaml`
- `/scripts/data_generation/single_tasks/create_randring_datasets.sh`
- `/scripts/analysis_representations/run_all_multi_pretrain_30ep.sh`

### Modified
- `/src/utils.py` - Added seed_all() function
- `/src/training/train.py` - Use seed_all()
- `/src/analysis/analyze_representations.py` - Direct n_train/n_test specification
- `/src/data_processing/create_center_dataset.py` - Fixed 0D array and print order
- `/src/data_processing/create_circlecount_dataset.py` - Fixed JSON serialization
- `/configs/training/train_randomwalk_1M.yaml` - Added eval section

## Next Steps
- Test randring dataset generation
- Consider additional stochastic tasks if needed
- Verify multi-task training with new randring task