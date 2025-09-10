# Representation Analysis Script Improvements & API Documentation
**Date**: 2025-08-31  
**Time**: 14:35  
**Session Focus**: Improving representation dynamics analysis script and documenting project APIs

## Summary
Enhanced the representation dynamics analysis script with better I/O interface, consistent file organization, improved visualizations, and created comprehensive API documentation for the project.

## Key Accomplishments

### 1. **Script Migration and Enhancement** (`src/analysis/analyze_representations.py`)
- Migrated `analysis/representation_dynamics.py` to `src/analysis/analyze_representations.py`
- Replaced old script that had inferior analysis logic
- Maintained the better analysis from representation_dynamics.py while adopting cleaner I/O interface

### 2. **Improved Command-Line Interface**
- Changed from positional to named arguments (`--exp_dir`, `--cities_csv`)
- Added configurable parameters:
  - `--n_probe_cities`: Number of cities to probe (default 5000)
  - `--n_train_cities`: Number for training linear probes (default 3000)  
  - `--seed`: Random seed for city sampling (default 42)
  - `--device`: Device selection (cuda/cpu)
- Better path handling - no hardcoded paths, relative to script location

### 3. **Organized Output Structure**
- Created subfolder system: `analysis/layers{X}_{Y}_probe{N}_train{M}/`
- Initially tried compact naming (`L3-4_P5000_T3000`) but reverted to clearer format
- All outputs in subfolder with consistent names:
  - `representation_dynamics.csv`
  - `dynamics_plot.png`
  - `world_map_evolution.gif`
- Prevents overwriting when running different parameter combinations

### 4. **Enhanced Visualization**
- **Redesigned dynamics plot**: 2x1 vertical layout instead of 1x3 horizontal
- **Top subplot**: Training loss (left y-axis) + Mean location error in km (right y-axis, log scale)
- **Bottom subplot**: R² scores + Haversine distance error (dual y-axes)
- Shared x-axis for easy visual correspondence
- Shows both correlation (R²) and absolute error (km) metrics together

### 5. **API Documentation** (`api.md`)
Created comprehensive API documentation covering:
- Data processing scripts (generate_filtered_dataset, create_distance_dataset, etc.)
- Tokenizer creation
- Training script
- Analysis tools
- Visualization utilities
- Complete usage examples and workflow

### 6. **Analysis Results**
Ran analysis on two experiments:
- **dist_100k_1M_20epochs**: Final R² 0.956/0.923, error 993 km
- **dist_100k_1M_80epochs_bs64**: Final R² 0.954/0.939, error 1193 km
- Discovered that extended training (80 epochs) doesn't improve and may degrade performance
- Peak performance for 80-epoch model at step 500k (989 km) before degradation

## Technical Details

### Key Bug Fixes
- Fixed inconsistent underscore formatting in folder names
- Removed redundant layer information from plot titles (already in folder name)
- Fixed path resolution for tokenizer when not absolute

### Clarifications Made
- Confirmed that the linear regression probes predict longitude/latitude for test cities
- The "mean_dist_error_km" is the haversine distance between predicted and actual locations
- This measures how well representations encode geographic information

## Files Modified
- `/src/analysis/analyze_representations.py` - Complete rewrite with improvements
- `/api.md` - New comprehensive API documentation
- Various experiment analysis outputs in respective folders

## Next Steps Potential
- Could add more probe metrics (e.g., country classification accuracy)
- Could analyze different layer combinations systematically
- Could create comparison plots across multiple experiments