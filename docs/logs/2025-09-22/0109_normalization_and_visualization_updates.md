# Development Log - 2025-09-22 01:09
## Topic: Heatmap Normalization Updates and Visualization Scripts

### Summary
Implemented a new log-ratio normalization scheme for fine-tuning experiment heatmaps to better handle multi-scale error metrics, and fixed inconsistencies in visualization scripts.

### Key Changes

#### 1. New Normalization Scheme Implementation
- **Problem**: Previous normalization wasn't handling multi-scale error metrics well (ranging over many orders of magnitude)
- **Solution**: Implemented log-ratio normalization using both PT1's standard and Atlantis baseline performances
  - For error metrics: `log(baseline_atlantis/value_atlantis) / log(baseline_atlantis/baseline_standard)`
  - For accuracy metrics: `(value_atlantis - baseline_atlantis) / (baseline_standard - baseline_atlantis)`
  - Results in 0-1 scale where:
    - 0.0 = No improvement from Atlantis baseline
    - 1.0 = Reached standard task performance level
    - >1.0 = Super-generalization (better than standard)

#### 2. Updated Visualization Scripts
- **plot_ft1_heatmap.py**: Single-task fine-tuning (7x7 matrix)
  - Added new normalize_metric function with log-ratio calculation
  - Updated colorbar to show 0=no improvement, 1=standard level, >1=super-generalization
  - Set vmax=1.5 to allow visualization of super-generalization

- **plot_ft2_heatmap.py**: Two-task fine-tuning (21x7 matrix)
  - Implemented same normalization scheme
  - Creates 3 subplots: actual performance, predicted (max of FT1), and prediction error
  - Fixed vmax ranges for consistency

- **plot_ft3_heatmap.py**: Three-task fine-tuning (7x7 matrix)
  - Applied log-ratio normalization
  - Notes that FT3 trains on both standard and Atlantis versions

- **plot_ftwb2_heatmap.py**: Weak baseline two-task fine-tuning (7x7 matrix)
  - Fixed inconsistent colorbar labels and vmax values
  - Changed from "0=baseline or worse, 1=perfect" to proper normalization description
  - Updated difference plot vmin/vmax from ±1 to ±1.5

### Technical Details

#### Normalization Function
```python
def normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy=False):
    if is_accuracy:
        # Linear normalization for accuracy metrics
        normalized = (value_atlantis - baseline_atlantis) / (baseline_standard - baseline_atlantis)
    else:
        # Log-ratio for error metrics
        import math
        numerator = math.log(baseline_atlantis / value_atlantis)
        denominator = math.log(baseline_atlantis / baseline_standard)
        normalized = numerator / denominator
    # Clip to [0, 1.5] range
    return max(0.0, min(1.5, normalized))
```

#### Edge Case Handling
- When baseline_standard >= baseline_atlantis (no room for improvement): return 0.0
- When value <= 0 or other edge cases: fallback to linear normalization
- When denominator = 0: return 0.0

### Results from Running Scripts

#### FT1 (Single-task) Diagonal Analysis:
- distance: 0.630 (partial recovery)
- trianglearea: 0.615 (partial recovery)
- angle: 0.716 (near-complete transfer)
- compass: 0.999 (near-complete transfer)
- inside: 0.894 (near-complete transfer)
- perimeter: 0.952 (near-complete transfer)
- crossing: 0.941 (near-complete transfer)

All four heatmap visualizations successfully generated and saved to:
- `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/scratch/plots/evaluation/ft1_performance_heatmap.png`
- `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/scratch/plots/evaluation/ft2_performance_heatmap.png`
- `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/scratch/plots/evaluation/ft3_performance_heatmap.png`
- `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/scratch/plots/evaluation/ftwb2_performance_heatmap.png`

### Files Modified
1. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ft1_heatmap.py`
2. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ft2_heatmap.py`
3. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ft3_heatmap.py`
4. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ftwb2_heatmap.py`

### Notes
- The new normalization provides a much more interpretable scale for comparing performance across different tasks with vastly different error magnitudes
- The log-ratio approach naturally handles the exponential nature of error reduction during training
- All scripts now use consistent visualization parameters for better comparison across experiments