# Analysis Script Standardization and Regex-Based Filtering
**Date**: 2025-09-03 22:23
**Session**: Standardizing analyze_representations.py to use YAML configs and implementing regex-based city filtering

## Major Accomplishments

### 1. Standardized Analysis Script Interface
- **Problem**: `analyze_representations.py` used complex command-line arguments while other scripts use YAML configs
- **Solution**: Refactored to accept `config.yaml` and optional `--overwrite` flag
- **Pattern**: `python src/analysis/analyze_representations.py configs/analysis_dist_1M.yaml [--overwrite]`
- **Files Modified**: `src/analysis/analyze_representations.py`

### 2. Fixed Import Issues
- **Problem**: Script tried to import non-existent `load_cities_csv` and `haversine` functions
- **Solution**: 
  - Replaced `haversine` with `euclidean_distance` (matches new coordinate system)
  - Replaced `load_cities_csv` with direct `pd.read_csv()`
  - Fixed coordinate column names: `longitude/latitude` → `x/y` (scaled by 10)
- **Files Modified**: `src/analysis/analyze_representations.py`

### 3. Implemented Regex-Based City Filtering
- **Old Approach**: Complex file paths for additional cities, concatenation logic, region exclusions
- **New Approach**: Simple regex patterns in YAML:
  ```yaml
  probe_train: "region:^(?!Atlantis).*"  # Exclude Atlantis from training
  probe_test: ".*"                       # All cities for testing
  highlight: "region:Atlantis"           # Highlight Atlantis in visualizations
  ```
- **Key Feature**: Support for both name-based and region-based patterns with `region:` prefix
- **Files Modified**: `src/analysis/analyze_representations.py`

### 4. Required Analysis Name Configuration
- **Change**: Made `analysis_name` required in config (no auto-generated directory names)
- **Benefit**: Explicit control over where results are saved
- **Example**: `analysis_name: "standard_probe_3_4"`
- **Files Modified**: `src/analysis/analyze_representations.py`

### 5. Direct Region Column Usage
- **Discovery**: Cities CSV already has `region` column - no need for country→region mapping
- **Simplification**: Removed all `country_to_region` mapping logic
- **Result**: Atlantis cities correctly tagged as "Atlantis" region instead of "Unknown"
- **Files Modified**: `src/analysis/analyze_representations.py`

### 6. Created Analysis Config Files
- **Created**: Multiple analysis configs for different experiments:
  - `configs/analysis_dist_1M_no_atlantis.yaml` - Base model excluding Atlantis
  - `configs/analysis_dist_1M_with_atlantis.yaml` - Model trained with Atlantis
  - `configs/analysis_ft_atlantis_100k.yaml` - Fine-tuned model
  - `configs/analysis_ft_atlantis_120k_mixed.yaml` - Mixed dataset, exclude Atlantis from probe
  - `configs/analysis_ft_atlantis_120k_mixed_noafrica.yaml` - Exclude Africa from probe training
  - `configs/analysis_ft_atlantis_120k_mixed_trainall.yaml` - Train probe on all cities

### 7. Enhanced Filter Function for Region Support
- **Enhancement**: `filter_cities_by_pattern()` now supports:
  - Direct city name patterns: `"^Atlantis_"`
  - Region-based patterns: `"region:^(?!Africa).*"`
- **Implementation**: Checks for `region:` prefix and applies pattern to appropriate column
- **Files Modified**: `src/analysis/analyze_representations.py`

### 8. Created Batch Analysis Script
- **Created**: `scripts/run_analysis.sh` to run multiple analyses with `--overwrite`
- **Pattern**: Minimal bash script following project conventions
- **Files Created**: `scripts/run_analysis.sh`

## Key Design Decisions

### Regex Pattern Approach
- Clean separation of concerns: train set, test set, visualization highlighting
- No external files needed - all config in YAML
- Support for both city names and regions with `region:` prefix

### Direct Region Usage
- Eliminated unnecessary mapping complexity
- Uses region data already present in CSV
- Simplifies code and reduces potential errors

### Explicit Analysis Names
- No more auto-generated directory names with concatenated parameters
- Clear, intentional output paths
- Better organization of analysis results

## Technical Details

### Coordinate System Adaptation
- All coordinates scaled by 10 (longitude -180→180 becomes -1800→1800)
- Updated axis limits and tick marks in visualization
- Maintained consistency with data processing pipeline

### Backward Compatibility
- Script handles both old and new config formats
- Graceful defaults for missing parameters
- Clear error messages for required fields

## Files Modified
- `src/analysis/analyze_representations.py` - Complete refactor for YAML configs and regex filtering
- `configs/analysis_*.yaml` - 6 new analysis config files
- `scripts/run_analysis.sh` - New batch analysis script

## Next Steps (Future)
- Update visualization to use highlight patterns for special coloring (currently uses region_colors dict)
- Consider adding more complex filtering patterns (e.g., population-based, coordinate bounds)