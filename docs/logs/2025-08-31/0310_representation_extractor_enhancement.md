# Representation Study Notebook Enhancement
**Date**: 2025-08-31  
**Time**: 03:10  
**Focus**: Enhanced RepresentationExtractor class and notebook fixes

## Summary
Enhanced the representation study notebook to support multi-layer extraction and fixed various issues with the notebook cells.

## Work Completed

### 1. RepresentationExtractor Class Enhancement
- **Upgraded the extractor to support multiple layer indices**:
  - Can now pass a single int, list of ints, or None (defaults to layer 3)
  - Supports extracting from multiple layers simultaneously
  - Added `concatenate` parameter to control output format
  - Maintains backward compatibility with `layer_idx` property

- **Key features added**:
  - Flexible initialization: `RepresentationExtractor(model, [2, 3, 4])`
  - Concatenated output: Returns tensor of shape `(batch, seq_len, hidden_size * n_layers)`
  - Dictionary output: Returns dict mapping layer indices to tensors
  - Clean hook management for multiple layers

### 2. Notebook Fixes and Improvements

#### Fixed Variable Collision (cell 24)
- Changed `results` to `saved_results` to avoid overwriting the linear probe results dictionary
- This was causing the summary statistics cell to fail

#### Fixed Summary Statistics Cell (cell 27)
- Added checks for variable existence before using them
- Handles missing `pc1_corr` variable by calculating it if needed
- Added helpful messages when dependencies haven't been run
- Made the cell more robust to partial notebook execution

#### Fixed Model Loading (cell 6)
- Checkpoint path was incomplete/hardcoded
- Now derives checkpoint path from config path automatically
- When switching experiments in cell 4, model loads from correct checkpoint
- More flexible and maintainable approach

### 3. Testing and Validation
- Added test cell demonstrating multi-layer extraction capabilities
- Verified backward compatibility with existing code
- All cells below the enhancement remain functional
- Maintained expected tensor shapes and behavior

## Technical Details

### Multi-Layer Extraction Example
```python
# Extract from layers 3 and 4, concatenated
extractor = RepresentationExtractor(model, layer_indices=[3, 4])
reps = extractor.extract(input_ids, attention_mask)
# Output shape: (batch_size, seq_len, 256)  # 128 * 2 layers

# Extract all layers as dictionary
all_extractor = RepresentationExtractor(model, list(range(6)))
reps_dict = all_extractor.extract(input_ids, attention_mask, concatenate=False)
# Returns: {0: tensor(...), 1: tensor(...), ..., 5: tensor(...)}
```

### Key Design Decisions
1. **Backward Compatibility**: Preserved `layer_idx` property and single-layer behavior
2. **Flexible Output**: Support both concatenated and dictionary outputs
3. **Clean Architecture**: Separate hook functions for each layer
4. **Sorted Indices**: Ensure consistent ordering regardless of input order

## Files Modified
- `/n/home12/cfpark00/WM_1/notebooks/representation_study.ipynb`:
  - Cell 8: Enhanced RepresentationExtractor class
  - Cell 9: Added test cell for multi-layer extraction
  - Cell 24: Fixed variable name collision
  - Cell 27: Made summary statistics more robust
  - Cell 6: Fixed model loading from config

## Impact
- Researchers can now easily analyze how representations evolve across layers
- More robust notebook that handles partial execution better
- Cleaner separation between different results (probe results vs saved results)
- Easier to switch between different trained models for analysis

## Next Steps Potential
- Could extend to support attention weight extraction
- Could add layer-wise correlation analysis
- Could create comparative studies across different models
- Could add support for extracting from specific attention heads

## Notes
- The notebook now uses the randomwalk model (`rw200_100k_1m_20epochs`) 
- All visualizations and analyses work with the enhanced extractor
- The enhancement maintains full backward compatibility