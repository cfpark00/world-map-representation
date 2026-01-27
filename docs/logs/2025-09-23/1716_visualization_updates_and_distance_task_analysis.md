# Development Log - 2025-09-23 17:16
## Topic: Visualization Updates and Distance Task Analysis

### Summary
Today's session focused on updating visualization scripts for PCA timeline analysis, creating heatmap scripts for fine-tuning experiments, and investigating why the distance task appears to harm other tasks during multi-task training.

### Tasks Completed

#### 1. PCA Timeline Visualization Enhancement
- **Added random projection type** to `/src/analysis/visualize_pca_3d_timeline.py`
  - Implemented `type: random` axis mapping that uses 3 random orthonormal directions
  - Uses QR decomposition to generate orthonormal basis
  - Computes variance captured for display

- **Refactored for better separation of concerns**
  - Created unified `compute_projection()` function that handles all projection types (PCA, mixed, random)
  - Returns consistent metadata structure for all projection types
  - Simplified plotting code by removing projection-specific conditionals
  - Fixed bug where `n_components` wasn't passed to compute_projection

#### 2. Probe Implementation Updates
- **Added 7 new probe variants** to `/src/analysis/analyze_representations_higher.py`
  - Implemented `{task}_firstcity_last` variants for all 7 tasks
  - These extract only the last digit token without the transition token
  - Tasks: distance, trianglearea, crossing, angle, compass, inside, perimeter

#### 3. Heatmap Visualization Scripts
- **Created FTwb1 heatmap script** (`/scratch/plots/evaluation/plot_ftwb1_heatmap.py`)
  - Single-task fine-tuning with warmup+bias
  - 7x7 heatmap matching FT1 structure
  - Includes comparison with FT1 results

- **Updated FT2 and FTwb2 heatmaps**
  - Changed difference plots to use symmetric coolwarm colormap
  - Updated to use `m = np.max(np.abs(matrix_diff))` for cleaner code

- **Fixed FTwb2 predictions**
  - Corrected to use FTwb1 (not FT1) as baseline for max predictions
  - Updated from 1x3 to 2x2 grid layout
  - Added scatter plot of true vs predicted values with correlation statistics
  - Differentiated training vs transfer tasks with colors/shapes

#### 4. Distance Task Investigation
Investigated why distance task appears to harm other tasks in multi-task training:

##### Findings:
1. **Training configurations are identical** across all FT1 tasks
   - Same learning rate (1e-5), batch size, epochs
   - Same dataset composition: 20k no-Atlantis + 100k Atlantis-required

2. **Loss computation is uniform**
   - All tasks use full sequence loss (not answer-only)
   - `use_loss_mask` is NOT set in configs, defaults to False
   - MultiTaskCollator applies same loss computation to all tasks

3. **Key differences identified**:
   - **Distance is 2-city task** while perimeter/trianglearea use 3 cities
   - **Output characteristics**:
     - Distance: continuous, range ~0-1000
     - Perimeter: continuous, range ~0-3000
     - TriangleArea: continuous, range ~0-500000
     - Crossing/Inside/Compass: categorical (2-8 classes)

4. **Hypothesis**: Distance task's simpler structure (2 cities, direct calculation) may lead to overly simplistic representations that don't transfer well to more complex tasks

### Code Organization
- Maintained project structure conventions
- All Python changes in `/src/`
- Configuration files in `/configs/`
- Visualization scripts in `/scratch/plots/`

### Next Steps for Investigation
- Compare performance patterns of all regression tasks (distance, perimeter, trianglearea)
- Analyze gradient norms during training for different task combinations
- Examine learned representations via PCA to identify structural differences
- Consider whether 2-city vs 3-city structure is the key differentiator

### Files Modified
1. `/src/analysis/visualize_pca_3d_timeline.py`
2. `/src/analysis/analyze_representations_higher.py`
3. `/scratch/plots/evaluation/plot_ftwb1_heatmap.py` (created)
4. `/scratch/plots/evaluation/plot_ft2_heatmap.py`
5. `/scratch/plots/evaluation/plot_ftwb2_heatmap.py`

### Notes
- Random projection implementation uses seed 42 for reproducibility
- All heatmap scripts maintain consistent normalization methods
- Distance task analysis revealed no special preprocessing, suggesting the issue is inherent to task characteristics