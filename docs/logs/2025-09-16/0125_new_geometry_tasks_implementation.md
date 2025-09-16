# Development Log - 2025-09-16 01:25 - New Geometry Tasks Implementation

## Summary
Implemented 7 new geometric reasoning tasks for the World Model project, standardized dataset sizes, and created comprehensive dataset generation infrastructure.

## Tasks Completed

### 1. New Dataset Generation Scripts Created

#### Nearest Neighbor Task
- **File**: `src/data_processing/create_nearest_neighbor_dataset.py`
- **Grammar**: `nearest(c_ID,k)=c_ID1,c_ID2,...,c_IDk`
- **Config**: 5k train, 10k test samples (limited by ~25k unique possibilities)
- Finds k nearest cities to a query city

#### Line Crossing Detection Task
- **File**: `src/data_processing/create_crossing_dataset.py`
- **Grammar**: `cross(c_ID1,c_ID2;c_ID3,c_ID4)=TRUE/FALSE`
- **Config**: 1M train, 10k test samples
- Determines if two line segments intersect within their bounds
- Uses proper segment intersection algorithm (not just infinite line crossing)

#### Inside Convex Hull Task
- **File**: `src/data_processing/create_inside_dataset.py`
- **Grammar**: `inside(c_ID1;c_ID2,c_ID3,...)=TRUE/FALSE`
- **Config**: 1M train, 10k test samples
- Uses scipy ConvexHull and Delaunay triangulation for point-in-hull testing
- Handles 3-6 hull points

#### Center of Mass Task
- **File**: `src/data_processing/create_center_dataset.py`
- **Grammar**: `center(c_ID1,c_ID2,...;in=TRUE/FALSE)=c_ID`
- **Config**: 1M train, 10k test samples
- When in=TRUE: finds closest city from given set to their center of mass
- When in=FALSE: finds closest city from all cities
- **Important fix**: Updated to handle ties deterministically (selects first in list)

#### Circle Count Task
- **File**: `src/data_processing/create_circlecount_dataset.py`
- **Grammar**: `circlecount(c_ID,r=RADIUS)=COUNT`
- **Config**: 100k train, 10k test samples
- Counts cities within radius r (5-200) from center city
- Returns integer count (no leading zeros)

#### Compass Direction Task
- **File**: `src/data_processing/create_compass_dataset.py`
- **Grammar**: `compass(c_ID1,c_ID2)=DIRECTION`
- **Config**: 1M train, 10k test samples
- Returns compass direction from city1 to city2
- Output: N, NE, E, SE, S, SW, W, NW

#### Perimeter Task
- **File**: `src/data_processing/create_perimeter_dataset.py`
- **Grammar**: `perimeter(c_ID1,c_ID2,...)=LENGTH`
- **Config**: 1M train, 10k test samples
- Calculates perimeter of polygon formed by 2-5 cities
- Returns integer perimeter value

### 2. Data Analysis Scripts

#### Density Analysis
- **File**: `scratch/dataproperties/analyze_density.py`
- Analyzed density patterns for different radii
- Generated comprehensive visualization plots

#### Count Analysis
- **File**: `scratch/dataproperties/analyze_counts.py`
- Analyzed integer city counts within various radii
- Created histograms, percentile plots, and statistics
- Key finding: counts range from 0-2329 for radius 5-500

### 3. Dataset Standardization

Standardized all dataset sizes based on unique data availability:
- **Small datasets** (limited unique data):
  - Nearest Neighbor: 5k train, 10k test
  - Circle Count: 100k train, 10k test
- **Large datasets** (billions+ unique combinations):
  - All others: 1M train, 10k test
- All use 128 validation samples

Updated configs and scripts:
- `nearest_1k` → `nearest_5k` (5k training)
- `circlecount_1M` → `circlecount_100k` (100k training)
- `center_100k` → `center_1M` (1M training)

### 4. Meta Script Creation

Created `scripts/data_generation/create_all_geometry_datasets_pad.sh`
- Single script to run all 11 geometry dataset generation tasks
- Clean format: shebang + 11 lines (one per task)

## Files Created/Modified

### New Python Scripts (7):
- `src/data_processing/create_nearest_neighbor_dataset.py`
- `src/data_processing/create_crossing_dataset.py`
- `src/data_processing/create_inside_dataset.py`
- `src/data_processing/create_center_dataset.py`
- `src/data_processing/create_circlecount_dataset.py`
- `src/data_processing/create_compass_dataset.py`
- `src/data_processing/create_perimeter_dataset.py`

### New Config Files (7):
- `configs/data_generation/nearest_5k_no_atlantis_pad.yaml`
- `configs/data_generation/crossing_1M_no_atlantis_pad.yaml`
- `configs/data_generation/inside_1M_no_atlantis_pad.yaml`
- `configs/data_generation/center_1M_no_atlantis_pad.yaml`
- `configs/data_generation/circlecount_100k_no_atlantis_pad.yaml`
- `configs/data_generation/compass_1M_no_atlantis_pad.yaml`
- `configs/data_generation/perimeter_1M_no_atlantis_pad.yaml`

### New Bash Scripts (7 + 1 meta):
- `scripts/data_generation/create_nearest_datasets_pad.sh`
- `scripts/data_generation/create_crossing_datasets_pad.sh`
- `scripts/data_generation/create_inside_datasets_pad.sh`
- `scripts/data_generation/create_center_datasets_pad.sh`
- `scripts/data_generation/create_circlecount_datasets_pad.sh`
- `scripts/data_generation/create_compass_datasets_pad.sh`
- `scripts/data_generation/create_perimeter_datasets_pad.sh`
- `scripts/data_generation/create_all_geometry_datasets_pad.sh` (meta script)

### Analysis Scripts (2):
- `scratch/dataproperties/analyze_density.py`
- `scratch/dataproperties/analyze_counts.py`
- `scratch/external/test_convex_hull.py` (testing scipy functionality)

### Modified:
- `src/data_processing/create_center_dataset.py` (tie-breaking fix)

## Key Technical Decisions

1. **Integer outputs only**: Avoided floating point by using counts instead of densities
2. **Tie-breaking**: Center task now deterministically selects first city when equidistant
3. **Padding convention**: City IDs padded to 4 digits, but numeric outputs (counts, distances, angles) are NOT padded
4. **Dataset sizing**: Based on analysis of unique data space for each task
5. **Loss masking**: All tasks generate loss masks for training (everything after '=' gets loss)

## Total New Tasks: 11 Geometry Tasks
1. Distance (existing)
2. Triangle Area (existing)
3. Angle (existing)
4. Random Walk (existing)
5. **Nearest Neighbor** (new)
6. **Line Crossing** (new)
7. **Inside Convex Hull** (new)
8. **Center of Mass** (new)
9. **Circle Count** (new)
10. **Compass Direction** (new)
11. **Perimeter** (new)

All tasks test different aspects of geometric reasoning and spatial understanding.