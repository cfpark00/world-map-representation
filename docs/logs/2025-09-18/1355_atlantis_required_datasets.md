# Development Log - 2025-09-18 13:55
## Atlantis Required Dataset Variants Creation

### Summary
Extended dataset generation infrastructure to create "atlantis_required" variants for multiple tasks, ensuring at least one Atlantis city appears in each sample. Created shared utility module for pair generation strategies and updated multiple dataset creation scripts.

### Major Changes

#### 1. Created Shared Data Utilities
**File**: `src/data_processing/data_utils.py`
- Extracted pair generation functions from `create_distance_dataset.py`
- Functions: `generate_pairs`, `generate_all_pairs`, `generate_within_group_pairs`, `generate_between_group_pairs`, `generate_mixed_pairs`, `generate_must_include_pairs`
- `generate_must_include_pairs`: Ensures ~10% have both cities from Atlantis, ~90% have one Atlantis + one other
- Now reusable across all dataset creation scripts

#### 2. Updated Dataset Creation Scripts

**Modified to use data_utils**:
- `create_distance_dataset.py` - Imports from data_utils instead of local functions
- `create_compass_dataset.py` - Now uses `generate_pairs` from data_utils

**Added must_include strategy support**:
- `create_crossing_dataset.py` - Ensures at least 1 of 4 cities is Atlantis
- `create_perimeter_dataset.py` - Ensures at least 1 vertex is Atlantis
- `create_center_dataset.py` - Ensures at least 1 city in group is Atlantis (partially updated)
- `create_inside_dataset.py` - Ensures at least 1 hull vertex is Atlantis

**Special handling for nearest_neighbor**:
- Modified `create_nearest_neighbor_dataset.py` to handle deterministic generation for must_include
- Generates ALL possible (city, k) combinations first
- Splits deterministically to ensure test set has no overlap with train
- Created `nearest_400_atlantis_required` (not 100k) due to limited Atlantis cities

#### 3. Created Atlantis Required Configs

**100k sample configs created**:
- `distance_100k_atlantis_required.yaml` (already existed)
- `compass_100k_atlantis_required.yaml`
- `angle_100k_atlantis_required.yaml`
- `trianglearea_100k_atlantis_required.yaml`
- `crossing_100k_atlantis_required.yaml`
- `perimeter_100k_atlantis_required.yaml`
- `center_100k_atlantis_required.yaml`
- `inside_100k_atlantis_required.yaml`
- `randomwalk_100k_atlantis_required.yaml` (config only, script doesn't support)

**Special cases**:
- `nearest_400_atlantis_required.yaml` - Only 400 train samples due to ~500 total possible queries

#### 4. Updated Bash Scripts
Added atlantis_required commands to:
- `create_distance_datasets.sh`
- `create_compass_datasets.sh`
- `create_angle_datasets.sh`
- `create_trianglearea_datasets.sh`
- `create_nearest_datasets.sh`
- `create_randomwalk_datasets.sh`
- `create_crossing_datasets.sh`
- `create_perimeter_datasets.sh`
- `create_center_datasets.sh`
- `create_inside_datasets.sh`

### Key Insights

#### Task Classification by Atlantis Impact

**Answer changes** (Atlantis affects computation):
- `nearest_neighbor` - Atlantis cities could be among k nearest
- `circlecount` - Atlantis cities get counted if within radius
- `center` - Center changes when in=FALSE searches all cities
- `inside` - Test point distribution changes

**Answer unchanged** (Atlantis only in prompt):
- `distance`, `compass`, `angle`, `trianglearea` - Geometric properties unchanged
- `crossing`, `perimeter` - Properties of specific cities unchanged
- `randomwalk` - Path through specific cities unchanged

### Technical Notes

1. **Must Include Strategy**: Typically ensures ~10% of pairs/groups have multiple Atlantis cities, ~90% have exactly one

2. **Nearest Neighbor Determinism**: Special handling to avoid train/test overlap when total query space is small

3. **Random Walk Issue**: Config created but script doesn't actually support must_include - would need modification

4. **Visualization Update**: Added regex filtering to `visualize_cities.py` to support excluding Atlantis regions

### Files Modified
- 10 dataset creation Python scripts
- 10 bash scripts
- 10+ new YAML configs
- Created `src/data_processing/data_utils.py`
- Updated `src/visualization/visualize_cities.py`

### Next Steps
- Implement must_include for `circlecount` if needed
- Fix `randomwalk` to properly support must_include strategy
- Complete `center` script update (generate_center_samples function)