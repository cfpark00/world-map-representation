# Gradient Analysis Implementation and Fixes
**Date**: 2025-09-21
**Time**: 00:16
**Main Topic**: Implementing activation gradient analysis for distance task

## Summary
Implemented and debugged activation gradient analysis to extract gradients flowing from loss back to intermediate token activations, with visualization of gradient norms across layers.

## Key Accomplishments

### 1. Created Gradient Analysis Script
- **File**: `src/analysis/analyze_activation_gradients.py`
- Successfully captures gradients using backward hooks (after discovering `retain_grad()` doesn't work on detached hidden states)
- Analyzes distance task sequences: `<bos> d i s t ( c _ X X X X , c _ X X X X ) = Y Y Y <eos>`
- Performs PCA analysis on gradient vectors
- Creates comprehensive visualizations

### 2. Fixed Critical Issues
- **Hidden try/except blocks**: Removed all error suppression per user feedback ("erroring out is wayyy better than hiding shit")
- **City ID format**: Fixed to use proper 4-digit zero-padded IDs from cities.csv
- **Layer indices**: Corrected to use valid layers (3, 4, 5, 6) instead of invalid ones (6, 8, 10)
- **Gradient capture method**: Switched from `retain_grad()` to backward hooks after extensive debugging

### 3. Created Gradient Norm Heatmap Visualization
- Shows gradient norms across all tokens (x-axis) and layers (y-axis)
- Properly displays tokens: `<bos> d i s t ( c _ X X X X , c _ X X X X )`
- Replaces digits with 'X' for aggregation across sequences
- Added layer 5 to visualization per user request
- Truncated display at closing parenthesis (input only, no output tokens)

### 4. Made Script Fully YAML-Driven
- Removed hardcoded path `/n/home12/cfpark00/WM_1`
- Added `task_type` field (currently only 'distance' supported)
- Made token detection dynamic (finds ')' automatically)
- Generic digit replacement (any digit → 'X')
- Clear error message for unsupported tasks

## Technical Discoveries

### Gradient Flow in HuggingFace Models
- Hidden states from `output_hidden_states=True` are detached from computation graph
- `retain_grad()` doesn't work on these tensors
- Solution: Use `register_full_backward_hook()` on model layers directly

### Backward Hooks Implementation
```python
def capture_all_gradients(layer_idx):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad = grad_output[0]
            captured_full_gradients[f"layer_{layer_idx}"] = grad[0].clone().detach()
```

## Key Observations from Analysis
1. Strong gradients at structural tokens (comma at position 12, closing paren at position 19)
2. Peak gradient activity in layers 3-4
3. Minimal gradients in layers 5-6 (computation mostly complete)
4. Model focuses on parsing structure in earlier layers

## Files Created/Modified
- `src/analysis/analyze_activation_gradients.py` - Main analysis script
- `configs/analysis_activation_gradients/distance_m1_10M.yaml` - Configuration
- `scratch/test_gradient_flow.py` - Test script for debugging
- `scratch/test_hooks_only.py` - Test script that proved hooks work
- `scratch/test_activation_access.py` - Test script for activation access

## User Feedback Incorporated
- Strong emphasis on not hiding errors
- Correct city ID format (4 digits, not 3)
- Include all relevant layers
- Show actual tokens on x-axis
- Stop display at input tokens only

## Next Steps (Future Work)
- Implement support for other task types (trianglearea, crossing, etc.)
- Each task needs different sequence generation logic
- Would require task-specific position detection

## Configuration Example
```yaml
task_type: distance  # Only 'distance' currently supported
output_dir: "data/experiments/m1_10M/analysis/activation_gradients_l3_4_6"
experiment_dir: "data/experiments/m1_10M"
cities_csv: "data/datasets/cities/cities.csv"
n_samples: 200
target_layers: [3, 4, 5, 6]
target_positions:
  - city1_last_digit
  - comma
```

## Lessons Learned
1. Always check actual data format (city IDs are 4 digits, not indices)
2. Don't hide errors - fail fast and loud
3. Test gradient flow mechanics in isolation before full implementation
4. Backward hooks are powerful for accessing gradients in complex models
5. Dynamic token detection makes scripts more flexible and reusable