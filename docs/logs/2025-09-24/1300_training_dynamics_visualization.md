# Development Log - 2025-09-24 13:00

## Training Dynamics Visualization for PT1-3

### Summary
Created comprehensive plotting scripts for visualizing PT1-3 training dynamics, including loss curves, angle prediction error, and coordinate prediction R² values from representation analysis. Developed both clean minimal plots for presentation and verbose diagnostic plots for detailed analysis.

### Major Tasks Completed

#### 1. Explored PT1-3 Experiment Data
- Located training data in `data/experiments/pt1-3/`
- Found trainer state JSON with complete training history at checkpoint-328146
- Discovered representation analysis CSV files containing R² metrics for coordinate prediction
  - Located in `analysis_higher/angle_firstcity_last_and_trans_l5/representation_dynamics.csv`
  - Contains x/y coordinate R² for both train and test sets

#### 2. Key Findings from PT1-3
- **Training Progress**: 328,140 steps over 42 epochs
- **Loss Reduction**: 83% (4.83 → 0.82)
- **Angle Error**: Improved from 56.27° to 1.36° mean error
- **Coordinate R²**:
  - X-coordinate: -0.074 → 0.989 (test)
  - Y-coordinate: -0.043 → 0.993 (test)
  - Both coordinates achieve R² > 0 at step ~32,816
  - Both reach R² > 0.9 by steps 49,224-57,428

#### 3. Created Visualization Scripts

##### a. Clean Vertical Plot (`plot_pt1-3_vertical.py`)
- **Layout**: 2 vertically stacked subplots (15x10 inches)
- **Top plot**: Training and validation loss
  - Log-log scale with x starting at 500
  - Y-axis range: 0.8 to 1.05 for focused view
- **Bottom plot**: Twin y-axes
  - Left: Angle error (log scale, custom ticks: 1, 2, 10, 30, 80°)
  - Right: Mean coordinate R² (linear scale)
- **Styling**: No grids, no titles, minimal legends, clean spines

##### b. Verbose Diagnostic Plot (`plot_pt1-3_verbose.py`)
- **Layout**: 2x3 grid (16x6 inches)
- **Row 1**: Training loss, learning rate, gradient norms
- **Row 2**:
  - Angle prediction error (mean only, with std band)
  - X,Y coordinate R² (test solid, train dotted)
  - Euclidean distance error
- **Features**:
  - Horizontal grid lines only (no vertical)
  - Comprehensive metrics for detailed analysis
  - Train/test comparison for R² values

#### 4. Plot Customizations Implemented
- Log scales on both axes for loss and angle error
- Non-scientific notation for axis labels
- Custom angle error ticks for interpretability
- Removed unnecessary panels based on user feedback
- Clean minimal design for presentation quality

### File Structure Changes
Created new directory structure:
```
scratch/formation_dynamics/
├── figures/
│   ├── pt1-3_vertical.png (main presentation plot)
│   └── pt1-3_verbose.png (diagnostic plot)
├── plot_pt1-3_vertical.py
└── plot_pt1-3_verbose.py
```

Cleaned up intermediate files:
- Removed multiple draft plotting scripts
- Removed intermediate figure versions
- Kept only final two scripts and outputs

### Technical Details
- Used matplotlib with seaborn styling
- Implemented log-scale handling with offsets to avoid log(0)
- Created reusable plotting functions for future experiments
- Added comprehensive error handling for missing data
- Smooth curve interpolation for training loss visualization

### Key Insights
1. The model shows excellent convergence on both angle and coordinate prediction tasks
2. Coordinate prediction R² reaches near-perfect values (~0.99) by end of training
3. Angle error reduces to near-zero median with 1.36° mean error
4. Training and validation loss converge well, indicating good generalization

### Next Steps Potential
- Apply same visualization to other experiments (pt1-1, pt1-2, etc.)
- Create comparative plots across different training runs
- Add animation capabilities for training progression
- Extend to analyze layer-wise representation dynamics