# Development Log - 2025-09-21 11:13
## PCA Projection Bug Fix and Evaluation Config Fixes

### Summary
Fixed a critical bug in the PCA 3D timeline visualization code where x and y regression directions were not being properly projected out when computing residual PCA. Also fixed configuration issues in evaluation scripts.

### Tasks Completed

#### 1. Fixed PCA Projection Bug in visualize_pca_3d_timeline.py
**Issue**: The mixed projection mode (which finds regression directions for x/y coordinates and then computes PCA on residuals) had a bug where projections were not properly removed from representations.

**Root Cause**:
- Line 235 incorrectly used `residual_repr @ x_direction` instead of `last_representations @ x_direction`
- The projection formula using `np.outer` was mathematically incorrect for projecting out directions from multiple data points

**Fix Applied**:
- Replaced the sequential projection approach with a proper simultaneous projection
- Stack both x and y directions into a matrix U
- Use the general projection formula: `projection = (X @ U.T) @ pinv(U @ U.T) @ U`
- Subtract this projection from the original representations to get proper residuals

**Code Changes** (src/analysis/visualize_pca_3d_timeline.py:233-242):
```python
# Old (buggy):
residual_repr -= np.outer(residual_repr @ x_direction, x_direction)
residual_repr -= np.outer(residual_repr @ y_direction, y_direction)

# New (fixed):
U = np.vstack([x_direction, y_direction])  # (2, d)
Ginv = np.linalg.pinv(U @ U.T)  # (2, 2) - handles non-orthonormal directions
coeffs = (last_representations @ U.T) @ Ginv  # (n, 2)
projection = coeffs @ U  # (n, d) - projection onto span(U)
residual_repr = last_representations - projection
```

#### 2. Fixed Evaluation Config Files
**Issue**: All evaluation configs in `/configs/eval/ftset/pt1_ft1-1/` had incorrect `experiment_dir` paths pointing to evaluation output directories instead of training experiment directories.

**Root Cause**: Configs had `experiment_dir: data/experiments/pt1_ft1-1/evals/<task>` instead of `experiment_dir: data/experiments/pt1_ft1-1`

**Fix Applied**:
- Updated all 14 config files in the directory to use the correct experiment_dir path
- This allows the evaluation script to properly find checkpoint-0 for tokenizer initialization

### Testing
- Successfully ran the fixed PCA timeline visualization on a test config
- Results showed:
  - X direction R²: 0.998 (high correlation with x-coordinates)
  - Y direction R²: 0.996 (high correlation with y-coordinates)
  - Residual PC: 54.7% of remaining variance after removing geographic information
- User confirmed evaluation scripts now run successfully after config fixes

### Impact
- The PCA projection bug would have caused spurious correlations between the residual PC and x-coordinates
- The fixed version properly removes both x and y directions before computing the residual PCA
- This ensures the third axis in mixed mode truly captures variation orthogonal to geographic information
- Evaluation scripts can now properly load tokenizers and evaluate model checkpoints

### Files Modified
1. `/src/analysis/visualize_pca_3d_timeline.py` - Fixed projection bug
2. `/configs/eval/ftset/pt1_ft1-1/*.yaml` (14 files) - Fixed experiment_dir paths

### Technical Notes
- The bug highlights the importance of careful linear algebra when projecting out multiple potentially non-orthogonal directions
- Using the pseudoinverse (pinv) handles the general case where directions may not be orthonormal
- The fix ensures mathematical correctness: projecting a point onto the span of multiple vectors requires considering their linear combinations