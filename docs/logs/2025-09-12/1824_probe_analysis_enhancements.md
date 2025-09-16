# Development Log - 2025-09-12 18:24

## Summary
Enhanced probe analysis pipeline with fit quality visualizations and configurable probe methods (linear, lasso, ridge regression).

## Major Tasks Completed

### 1. Added Fit Quality Visualizations
- **Created `create_fit_quality_plot()` function** in `analyze_representations.py`
  - Generates 2x2 scatter plots showing actual vs predicted coordinates
  - Separate plots for X/Y coordinates and train/test sets
  - Shows R² scores and perfect fit reference lines
- **Implemented configurable fit saving**:
  - Added `save_fits` parameter to config files (default: false)
  - When `save_fits: true`, generates fit quality plots for ALL checkpoints
  - Saves plots in `analysis/{probe}/fits/` subdirectory
  - File naming: `fit_quality_step{step:05d}.png`

### 2. Made Probe Method Configurable
- **Added `method` section to config files** allowing specification of:
  - Method name: `"linear"`, `"lasso"`, or `"ridge"`
  - Hyperparameters per method:
    - Ridge: `alpha`, `solver`, `max_iter`, `tol`
    - Lasso: `alpha`, `max_iter`, `tol`
    - Linear: no parameters (OLS)
- **Updated `analyze_checkpoint()` function**:
  - Takes `method_config` parameter
  - Creates appropriate sklearn model based on config
  - Falls back to Ridge(alpha=10.0) if no method specified
- **Display probe configuration** in analysis output

### 3. Created Method Variant Configs
- **Created example configs** for `dist_1M_no_atlantis_probe1`:
  - `dist_1M_no_atlantis_probe1_linear.yaml` - Linear regression variant
  - `dist_1M_no_atlantis_probe1_lasso.yaml` - Lasso regression variant
  - Each saves to separate output directory to avoid overwriting

### 4. Researched sklearn Diagnostics
- **Investigated sklearn Ridge capabilities**:
  - `n_iter_` attribute only available for 'sag'/'lsqr' solvers
  - No built-in loss history tracking
  - No convergence plots like SGDRegressor
- Confirmed fit quality plots are best approach for visualizing probe performance

## Files Changed

### Modified
- `/src/analysis/analyze_representations.py`:
  - Added `create_fit_quality_plot()` function
  - Added `method_config` parameter throughout
  - Added fit quality plot generation logic
  - Added probe method configuration display

### Created
- `/configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1_linear.yaml`
- `/configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1_lasso.yaml`

### Updated Configs
- Added `save_fits: true` to all `ft_atlantis_llr` probe configs
- Added `method` section example to `dist_1M_no_atlantis_probe1.yaml`

## Key Implementation Details

### Fit Quality Plot Generation
- Generates plots for checkpoints based on `save_fits` setting
- When true: saves for ALL checkpoints
- When false/missing: no fit quality plots generated
- All plots saved in `fits/` subdirectory only (no duplicate in root)

### Method Configuration Structure
```yaml
method:
  name: "ridge"  # Options: "linear", "lasso", "ridge"
  alpha: 10.0    # For ridge/lasso
  solver: "auto" # For ridge
  max_iter: 1000 # For iterative solvers
  tol: 0.0001    # Convergence tolerance
```

### Probe Creation Logic Location
- Lines 117-142 in `analyze_representations.py`
- Reads method name from config
- Applies appropriate hyperparameters
- Creates sklearn model accordingly

## Testing
- Verified fit quality plots generate correctly
- Confirmed plots save to `fits/` subdirectory
- Tested method configuration parsing and application

## Next Steps
- User can now run probes with different regression methods
- Fit quality plots provide visual diagnostics of probe performance
- Can compare ridge vs lasso vs linear regression effectiveness