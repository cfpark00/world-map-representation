# 2025-09-17 12:11 - Critical Evaluation Bug Fixes and 3D Visualization Development

## Summary
Fixed a critical bug in model evaluation that was causing massive discrepancy between training and post-training metrics. Also created a new 3D visualization tool for analyzing model representations.

## 1. Critical Evaluation Bug Discovery and Fix

### The Problem
- Training showed ~5 error on validation set during training
- Post-training evaluation showed ~130-140 error on the same validation set
- Senior developers suspected training was evaluating on wrong dataset

### Investigation Process
1. Traced through `src/training/train.py` and `src/eval/evaluate_checkpoints.py`
2. Confirmed both use validation set correctly (not train set)
3. Dataset inspection showed:
   - Train: 1,000,000 samples
   - Validation: 128 samples
   - Test: 10,000 samples
   - No overlap between splits

### Root Cause Identified
Found **TWO critical bugs** in `src/eval/evaluate_checkpoints.py`:

1. **Double-spacing bug**: Line 149 was doing `' '.join(p)` on already-spaced text
   - Dataset text: `"d i s t ( c _ 1 2 3 4 , c _ 5 6 7 8 ) ="`
   - After join: `"d   i   s   t   (   c   _   1   2   3   4   ,   c   _   5   6   7   8   )   ="`

2. **BOS token handling bug**:
   - Code was removing `<bos>` and `<eos>` from text
   - Then tokenizer was adding them back (default `add_special_tokens=True`)
   - Result: Double BOS tokens

### Fixes Applied
1. Removed the incorrect `' '.join()` operation
2. Keep `<bos>` and `<eos>` in text, use `add_special_tokens=False`
3. Also fixed similar issues in:
   - `src/analysis/analyze_representations.py`
   - `src/analysis/analyze_representations_higher.py`
   - `src/data_processing/create_randring_dataset.py`

### Verification
After fixes:
- Training final eval: **4.98** error
- Fixed post-training eval: **5.91** error
- Results now match as expected!

## 2. New 3D Visualization Tool: XY Residual PCA

### Purpose
Created `src/analysis/visualize_xy_residual_pca_3d.py` to visualize representations in a novel way:
- **X axis**: Best linear direction for predicting X coordinates
- **Y axis**: Best linear direction for predicting Y coordinates (orthogonalized)
- **Z axis**: First principal component AFTER projecting out X/Y information

### Key Features
- Shows geographic encoding quality (X/Y spread)
- Reveals non-geographic structure in representations (Z axis)
- Uses proper centering of coordinates (matching `analyze_representations.py`)
- Only shows test cities for cleaner visualization
- Cities colored by geographic region

### Technical Implementation
1. Centers coordinates by subtracting training mean
2. Finds optimal directions using Ridge/Linear regression
3. Orthogonalizes Y direction using Gram-Schmidt
4. Projects out both directions and performs PCA on residuals
5. Creates interactive Plotly visualization

### Config Files Created
- `xy_residual_pca_3d_distance_1M.yaml`
- `xy_residual_pca_3d_distance_1M_with_atlantis.yaml`
- `xy_residual_pca_3d_m1_10M.yaml`
- `xy_residual_pca_3d_m1_10M_ft1.yaml`

### Bug Fixes in Visualization
1. Fixed JSON serialization of numpy float32 values
2. Corrected linear regression to use centered coordinates
3. Removed training points and reference plane for cleaner plots

## 3. Other Fixes

### analyze_representations.py and analyze_representations_higher.py
- Added `add_special_tokens=False` when tokenizing prompts with `<bos>` already included

### create_randring_dataset.py
- Fixed tokenization to not add special tokens when already in text

## Files Modified
1. `/src/eval/evaluate_checkpoints.py` - Fixed double-spacing and BOS token bugs
2. `/src/analysis/analyze_representations.py` - Fixed special token handling
3. `/src/analysis/analyze_representations_higher.py` - Fixed special token handling
4. `/src/data_processing/create_randring_dataset.py` - Fixed tokenization
5. `/src/analysis/visualize_xy_residual_pca_3d.py` - Created new visualization tool

## Files Created
1. `/src/analysis/visualize_xy_residual_pca_3d.py` - New 3D visualization script
2. `/configs/analysis_3d_plots/xy_residual_pca_3d_*.yaml` - Config files for 4 models

## Impact
- **Critical**: Fixed evaluation pipeline producing wrong metrics
- Model evaluation now correctly shows true performance
- New visualization tool provides insights into what models learn beyond geography
- All tokenization bugs that could affect future analyses have been fixed

## Next Steps
- Re-run all post-training evaluations with fixed code
- Generate XY residual PCA visualizations for all models
- Compare residual structure across different training conditions