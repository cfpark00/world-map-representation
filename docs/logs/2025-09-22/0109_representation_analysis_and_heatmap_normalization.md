# Development Log - 2025-09-22 01:09
## Topic: Representation Analysis Updates, PCA Timeline Fixes, and Heatmap Normalization

### Summary
Major updates to representation analysis pipeline, PCA visualization with proper train/test separation, prompt format standardization, and implementation of new log-ratio normalization for fine-tuning experiment heatmaps.

### Part 1: Representation Analysis and Prompt Format Updates

#### 1. Cleaned Up Prompt Formats in analyze_representations_higher.py
**Removed formats:**
- `dist` - Basic distance format
- `dist_city_and_transition` - Distance with city and transition
- `dist_firstcity_last_and_comma` - Renamed to `distance_firstcity_last_and_comma`
- `trianglearea_firstcity_last` - Triangle area with just last digit
- `randomwalk_firstcity_last_and_comma` - Random walk task format

**Implemented new formats with "_firstcity_last_and_trans" pattern:**
- `distance_firstcity_last_and_trans`
- `angle_firstcity_last_and_trans`
- `compass_firstcity_last_and_trans`
- `inside_firstcity_last_and_trans`
- `perimeter_firstcity_last_and_trans`
- `crossing_firstcity_last_and_trans`
- `trianglearea_firstcity_last_and_trans`

**Key change**: Renamed "comma" to "trans" (transition token) for generality since different tasks use different separators (e.g., inside uses ";")

#### 2. Fixed Data Generation Inconsistency
**Problem**: Inside task format inconsistency between data generation and tasks.json
- Data generation used: `inside(c_TEST,c_ID1,c_ID2,...)`
- tasks.json had: `inside(c_TEST;c_ID1,c_ID2,...)`

**Fix**: Updated tasks.json to use semicolon separator consistently:
```json
"format": "inside(c_TEST;c_ID1,c_ID2,...)=TRUE/FALSE"
```

### Part 2: PCA Timeline Visualization Major Refactoring

#### 1. Implemented Proper Train/Test Separation
**Problem**: Original visualize_pca_3d_timeline.py had no control over train/test sets
- Used `test_only` parameter which was unintuitive
- No regex pattern support for filtering cities
- Couldn't match analyze_representations_higher.py methodology

**Solution**: Complete refactoring with:
- Removed `test_only` parameter
- Added required `train_frac` parameter
- Added `probe_train` and `probe_test` regex pattern support
- Imported proper `filter_dataframe_by_pattern` from utils.py

**Implementation details:**
```python
# Filter cities by patterns
if probe_train:
    train_candidates = filter_dataframe_by_pattern(city_df, probe_train, column_name='region')
if probe_test:
    test_candidates = filter_dataframe_by_pattern(city_df, probe_test, column_name='region')
# Random split within filtered candidates
train_indices = np.random.choice(train_candidate_indices, train_size, replace=False)
```

#### 2. Fixed Regex Filtering Bugs
**Bug 1**: Initial naive regex implementation returned 0 cities
- Root cause: Didn't handle numeric columns or complex patterns properly
- Fix: Used existing `filter_dataframe_by_pattern` from utils

**Bug 2**: Atlantis filtering pattern `region:^(?!Atlantis_).*` wasn't working
- Root cause: Pattern had underscore but actual region name is "Atlantis" (no underscore)
- Fix: Corrected to `region:^(?!Atlantis).*`

#### 3. Clarified Coordinate Systems
**Investigation**: Plot showed -25 to 25 range vs actual coordinates -1579 to 1762
- Discovered plot shows projection values in representation space, not actual coordinates
- PCA/linear regression projects high-dimensional representations to 3D visualization space
- The -25 to 25 range represents learned representation distances, not geographic coordinates

### Part 3: Heatmap Normalization Implementation

#### 1. New Log-Ratio Normalization Scheme
**Problem**: Previous normalization couldn't handle multi-scale error metrics (spanning many orders of magnitude)

**Solution**: Implemented log-ratio normalization using both PT1's standard and Atlantis baseline performances
- For error metrics: `log(baseline_atlantis/value_atlantis) / log(baseline_atlantis/baseline_standard)`
- For accuracy metrics: `(value_atlantis - baseline_atlantis) / (baseline_standard - baseline_atlantis)`
- Results in interpretable 0-1+ scale:
  - 0.0 = No improvement from Atlantis baseline
  - 1.0 = Reached standard task performance level
  - >1.0 = Super-generalization (better than standard)

#### 2. Updated All Four Heatmap Scripts
**Files modified:**
1. `plot_ft1_heatmap.py` - Single-task fine-tuning (7x7 matrix)
2. `plot_ft2_heatmap.py` - Two-task fine-tuning (21x7 matrix, 3 subplots)
3. `plot_ft3_heatmap.py` - Three-task fine-tuning (7x7 matrix)
4. `plot_ftwb2_heatmap.py` - Weak baseline two-task (7x7 matrix, 3 subplots)

**Key changes:**
- Added `normalize_metric()` function with log-ratio calculation
- Updated colorbars: `vmax=1.5` to show super-generalization
- Fixed FTWB2 inconsistent labels and ranges
- Updated difference plots to use ±1.5 range

#### 3. Results from Running All Scripts
**FT1 Diagonal Performance** (trained on single task):
- distance: 0.630 (partial recovery)
- trianglearea: 0.615 (partial recovery)
- angle: 0.716 (near-complete transfer)
- compass: 0.999 (near-complete transfer)
- inside: 0.894 (near-complete transfer)
- perimeter: 0.952 (near-complete transfer)
- crossing: 0.941 (near-complete transfer)

All visualizations successfully generated to `/scratch/plots/evaluation/`

### Technical Implementation Details

#### Normalization Function with Edge Case Handling
```python
def normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy=False):
    if is_accuracy:
        # Linear for accuracy metrics
        if baseline_standard <= baseline_atlantis:
            return 0.0  # No improvement possible
        normalized = (value_atlantis - baseline_atlantis) / (baseline_standard - baseline_atlantis)
    else:
        # Log-ratio for error metrics
        if baseline_standard >= baseline_atlantis or baseline_atlantis <= 0 or value_atlantis <= 0:
            # Handle edge cases
            if value_atlantis >= baseline_atlantis:
                return 0.0  # No improvement
            elif value_atlantis <= baseline_standard:
                return 1.0  # Reached or exceeded standard
            else:
                # Linear fallback
                return (baseline_atlantis - value_atlantis) / (baseline_atlantis - baseline_standard)

        # Normal case: use log-ratio
        import math
        numerator = math.log(baseline_atlantis / value_atlantis)
        denominator = math.log(baseline_atlantis / baseline_standard)

        if denominator == 0:
            return 0.0

        normalized = numerator / denominator

    return max(0.0, min(1.5, normalized))
```

### Files Modified Today
1. `/n/home12/cfpark00/WM_1/src/analysis/analyze_representations_higher.py`
2. `/n/home12/cfpark00/WM_1/src/analysis/visualize_pca_3d_timeline.py`
3. `/n/home12/cfpark00/WM_1/configs/tasks.json`
4. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ft1_heatmap.py`
5. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ft2_heatmap.py`
6. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ft3_heatmap.py`
7. `/n/home12/cfpark00/WM_1/scratch/plots/evaluation/plot_ftwb2_heatmap.py`

### Key Insights
- The log-ratio normalization naturally handles exponential error reduction during training
- PCA timeline visualization now properly separates train/test sets matching the probe training methodology
- Standardized prompt formats with "_firstcity_last_and_trans" pattern provide consistency across all tasks
- Transition tokens vary by task (comma for most, semicolon for inside) but are now handled uniformly