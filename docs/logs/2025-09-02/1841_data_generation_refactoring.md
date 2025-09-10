# Data Generation Refactoring and Visualization Updates
## Date: 2025-09-02 18:41

## Summary
Major refactoring of the data generation pipeline to create a cleaner, more modular system with Atlantis integration, safety checks, and improved visualization capabilities.

## Key Accomplishments

### 1. Data Generation Pipeline Overhaul
- **Merged Atlantis generation into main pipeline**: Removed standalone `generate_atlantis.py`, integrated functionality into `create_city_dataset.py`
- **Renamed for clarity**: `generate_filtered_dataset.py` → `create_city_dataset.py` (better reflects its role as main data generator)
- **Added Cartesian coordinates**: Now generates x,y coordinates directly from lon/lat using equirectangular projection
- **Removed longitude/latitude columns**: Simplified to only use x,y coordinates throughout the system
- **Random city IDs**: Added `--max-id` option to randomly assign IDs from [0, max_id-1], preventing geographic leakage patterns

### 2. Atlantis Configuration System
- **YAML-based configuration**: Created `atlantis_default.yaml` for defining synthetic regions
- **Region mapping**: Added `atlantis_region_mapping.json` to map country codes to region names
- **No hardcoded assumptions**: All Atlantis properties (names, locations, mappings) now configurable
- **Gaussian distribution**: Atlantis cities generated as Gaussian blobs in x,y space with configurable center and std_dev

### 3. Safety Infrastructure
- **Generic `init_directory()` function**: Replaced complex backwards-compatible `init_experiment_directory()` with simple, safe directory initialization
- **Single environment variable**: Uses `DATA_DIR_PREFIX` for all safety checks (removed confusing dual-prefix system)
- **Overwrite protection**: Only allows `--overwrite` within designated prefix to prevent accidental deletion of system files
- **Updated `.env`**: Added `DATA_DIR_PREFIX=/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/outputs`

### 4. Dataset Structure Improvements
- **Folder-based output**: Datasets now saved in folders with metadata
  - `cities.csv`: Main dataset
  - `metadata.json`: Creation parameters and timestamp
  - `atlantis_config.yaml`: Copy of config used (if Atlantis included)
- **Region column**: Added region mapping for all cities (world regions + Atlantis)
- **Consistent IDs**: Atlantis cities continue numbering from world cities, maintaining single ID space

### 5. Visualization Enhancements
- **New `plot_cities.py` script**: Flexible visualization with region highlighting
  - `--highlight-region` option to color specific regions differently
  - Absolute marker sizes in legend using empty scatter plot hack
  - Customizable figure size and output path
- **Shell scripts for automation**:
  - `scripts/create_dataset.sh`: Creates city dataset with Atlantis
  - `scripts/plot_cities.sh`: Generates both highlighted and non-highlighted maps
- **Improved aesthetics**: 
  - Larger fonts (title: 24pt, labels: 20pt, ticks: 18pt)
  - Better spacing with labelpad and title padding
  - Atlantis cities shown 5x larger than world cities when highlighted

### 6. Code Cleanup
- **Removed obsolete files**:
  - `generate_atlantis.py` (merged into main pipeline)
  - `plot_atlantis_cities.py` (replaced by better `plot_cities.py`)
- **Simplified create_dataset.sh**: Just calls Python with parameters, no complex bash logic
- **Fixed region naming**: Changed "Atlantis_0" to "Atlantis" for consistency

## Technical Details

### Coordinate System
- Using equirectangular projection (plate carrée)
- x = longitude (-180 to 180)
- y = latitude (-90 to 90)
- Simple Euclidean distance in degree space

### Dataset Statistics
- World cities: 5,075 (population ≥ 100k)
- Atlantis cities: 100
- Total: 5,175 cities
- 15 regions total (14 world regions + Atlantis)

### File Changes
- Created: `create_city_dataset.py`, `plot_cities.py`, `create_dataset.sh`, `plot_cities.sh`
- Modified: `utils.py` (new `init_directory()`), `train.py` (uses new function)
- Deleted: `generate_atlantis.py`, `plot_atlantis_cities.py`, `init_experiment_directory()` backwards compatibility

## Issues Resolved
- Fixed Atlantis position bias in cross-dataset pairs (now randomly swapped)
- Fixed legend marker size inconsistency (both now use '.' marker)
- Fixed environment variable loading issue in shell scripts
- Removed dangerous backwards compatibility that manipulated environment variables

## Next Steps
- Ready for training experiments with new datasets
- Consider adding more Atlantis regions for testing
- Possible future: different projections if distance accuracy needed

## Additional Updates
- Added `--exclude-region` parameter to `plot_cities.py` to allow filtering out specific regions
- Updated `scripts/plot_cities.sh` to include example of plotting world cities only (excluding Atlantis)
- Now supports three visualization modes:
  - Highlight specific region with different color
  - Exclude specific region from plot entirely
  - Plot all cities with same color
- Useful for comparing world-only vs world+Atlantis distributions