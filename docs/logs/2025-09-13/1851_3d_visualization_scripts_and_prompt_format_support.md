# Development Log - 2025-09-13 18:51

## Summary
Created 3D visualization scripts for PCA and prediction error plots, fixed representation saving format, added support for dist_city_and_transition prompt format, and resolved various configuration and formatting issues.

## Major Tasks Completed

### 1. 3D Visualization Scripts Creation
- **Created two main visualization scripts**:
  - `src/analysis/visualize_pca_3d.py` - 3D PCA visualization of city representations colored by region
  - `src/analysis/visualize_prediction_error_3d.py` - 3D plot with predicted x/y coordinates and error magnitude as z-axis
- **Features added**:
  - Support for both old (2D) and new (4D) representation formats
  - Configurable axis mapping for PCA (choose which PCs to plot)
  - Log-transformed error for z-axis in prediction error plot
  - Removed white borders from scatter plot markers
  - Multiple plot outputs (all, test-only, train-only)

### 2. Representation Format Updates
- **Modified `analyze_representations.py`**:
  - Changed saved representation format from flat to `(batch, token, layer, dim)`
  - Keeps both `representations` (4D) and `representations_flat` (2D) for backward compatibility
  - Updated metadata to include shape information

### 3. Token/Layer Selection Enhancement
- **Implemented -1 as "concatenate all"**:
  - `token_index: -1` concatenates all tokens
  - `layer_index: -1` concatenates all layers
  - `-1, -1` concatenates all token-layer combinations
- **Added bounds checking** to prevent index errors when fewer tokens are available

### 4. dist_city_and_transition Prompt Format Support
- **Added new prompt format** that extracts representations from 9 positions:
  - Positions 6-14: `c`, `_`, `i1`, `i2`, `i3`, `i4`, `,`, `c`, `_`
  - Saves as `(batch, 9, n_layers, hidden_dim)` instead of `(batch, 3, n_layers, hidden_dim)`
- **Updated visualization scripts** to handle variable number of tokens

### 5. Configuration Management
- **Fixed output_dir handling**:
  - Scripts require explicit `output_dir` in config (fail-fast philosophy)
  - Use `init_directory` from utils for safe directory creation
  - Removed auto-generation of output paths
- **Standardized probe method configuration**:
  - Uses same format as `analyze_representations.py`
  - Supports linear, ridge, and lasso regression with configurable parameters

### 6. Script and Config Creation
- **Created bash scripts** in `scripts/analysis_3d_plots/`:
  - `visualize_pca_3d.sh`
  - `visualize_prediction_error_3d.sh`
- **Created config files** in `configs/analysis_3d_plots/`:
  - `pca_3d_regions.yaml`
  - `prediction_error_3d_atlantis.yaml`

## Key Code Changes

### analyze_representations.py
- Lines 100-167: Added conditional extraction logic for dist_city_and_transition format
- Lines 115-136: Extract from positions 6-14 for the new format
- Lines 69-73: Added prompt_format parameter to analyze_checkpoint function

### visualize_pca_3d.py
- Lines 33-62: Added axis_mapping support for flexible PC selection
- Lines 237-258: Implemented -1 token/layer concatenation logic
- Lines 120-127: Calculate variance for selected components

### visualize_prediction_error_3d.py
- Lines 133-136: Added log transformation for errors using np.log1p
- Lines 180-192: Removed white borders from scatter markers
- Lines 38-83: Unified probe method configuration

## Issues Resolved
1. ValueError with representation shape unpacking (expected 4, got 2)
2. Missing prompt_format support for dist_city_and_transition
3. Incorrect output_dir path generation
4. White borders on plot markers
5. Linear scale making error differences hard to see

## Files Modified
- `src/analysis/analyze_representations.py`
- `src/analysis/visualize_pca_3d.py`
- `src/analysis/visualize_prediction_error_3d.py`
- Multiple config files in `configs/analysis_3d_plots/`

## Next Steps
- Test the dist_city_and_transition format with actual model runs
- Consider adding more visualization options (e.g., t-SNE, UMAP)
- Add support for comparing multiple checkpoints in same plot