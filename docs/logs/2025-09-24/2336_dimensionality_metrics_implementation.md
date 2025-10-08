# Development Log: Dimensionality Metrics Implementation
**Date**: 2025-09-24
**Time**: 23:36
**Focus**: Implementing 2D manifold testing with TwoNN, Correlation Dimension, and Local PCA metrics

## Summary
Replaced existing dimensionality analysis code with cleaner implementation of three key metrics for testing 2D manifold hypothesis. Created centralized module and scripts for testing whether neural network representations lie on low-dimensional manifolds.

## Key Accomplishments

### 1. Created Core Dimensionality Module
- **File**: `src/dimensionality.py`
- Implemented three essential metrics from `scratch/dimensionality/test_manifold_metrics.py`:
  - **TwoNN dimension** (Facco et al. 2017): Uses ratio of nearest neighbor distances with empirical CDF regression
  - **Correlation dimension**: Log-log scaling of neighborhood counts
  - **Local PCA 2D energy**: Fraction of variance explained by first 2 PCs in local neighborhoods
- Added `test_for_2d_manifold()` function that runs all three metrics and determines if data is 2D

### 2. Enhanced Correlation Dimension for Locality
- Modified correlation dimension to focus on LOCAL structure:
  - Changed from global percentiles (5th-95th) to local neighborhoods
  - Now uses 30th nearest neighbor distance as r_max
  - Only measures scaling within local patches (1st to 30th neighbors)
  - Better captures local geometry vs global structure

### 3. Created Test Script with Filtering Support
- **File**: `src/scripts/test_2d_manifold.py`
- Features:
  - Loads representations from `.pt` or `.npy` files
  - Handles 4D tensors (cities, tokens, layers, hidden) from analysis_higher
  - Supports filtering based on metadata (same as PCA scripts)
  - Uses metadata.json from checkpoint directory for city info
  - Applies filters using `filter_dataframe_by_pattern` with `column_name='region'`

### 4. Config Structure
- Created configs in `configs/analysis_dimensionality/`
- Key parameters:
  - `representations_base_path`: Path to representations directory
  - `checkpoint`: Specific checkpoint or null for latest
  - `filter`: Optional pattern like `"region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$"`
  - `max_samples`: For computational efficiency
  - `local_pca_k`: Neighborhood size for local PCA

### 5. Fixed Multiple Issues
- Corrected numerical vs alphabetical sorting of checkpoints
- Handled dict format from torch.load with 'representations' key
- Fixed 4D tensor extraction (already at layer 5, just needed flattening)
- Matched PCA's exact filtering logic using metadata instead of external cities.csv
- Fixed JSON serialization of numpy bool types

## Results on pt1-1 (No Atlantis filter)
- **TwoNN dimension**: 12.56 (far from 2!)
- **Correlation dimension**: 2.84 (higher than 2 even locally)
- **Local PCA 2D energy**: 0.50 (only 50% variance in 2D, needs >90%)
- **Conclusion**: NOT a 2D manifold

## Technical Insights

### Correlation Dimension Interpretation
- Measures how number of neighbors scales with radius: N(r) ∝ r^d
- On 2D surface: doubling radius → 4x neighbors (area ∝ r²)
- On 3D manifold: doubling radius → 8x neighbors (volume ∝ r³)
- Log-log slope captures this scaling exponent
- Local version (up to 30 neighbors) better captures intrinsic dimension without global effects

### Why Representations Aren't 2D
- High TwoNN dimension (12+) suggests genuinely high-dimensional
- Even locally (30 neighbors), correlation dimension > 2.5
- Local PCA needs 50%+ of variance beyond 2 components
- Representations likely use many dimensions for city encoding

## Files Modified/Created
- Created: `src/dimensionality.py`
- Created: `src/scripts/test_2d_manifold.py`
- Created: Multiple configs in `configs/analysis_dimensionality/`
- Modified: `src/scripts/analyze_dimensionality.py` (updated imports)
- Modified: `src/scripts/analyze_manifold_dimension.py` (updated imports)

## Next Steps
- Test on different representation types (distance vs triangle area vs crossing)
- Compare filtered (no Atlantis) vs unfiltered
- Analyze evolution across training checkpoints
- Consider whether higher-dimensional manifolds (3D, 4D) fit better

## Commands to Run Analysis
```bash
# Test with filter (no Atlantis)
uv run python src/scripts/test_2d_manifold.py configs/analysis_dimensionality/pt1-1.yaml --overwrite

# Test other experiments
uv run python src/scripts/test_2d_manifold.py configs/analysis_dimensionality/pt1-2.yaml --overwrite
```

## Notes
- User emphasized importance of LOCAL structure for manifold testing
- Filtering uses metadata.json from checkpoints, not external cities.csv
- PCA scripts only need cities.csv for 'mixed' axis mapping mode (regression)
- Local correlation dimension (30 neighbors) still shows >2D structure