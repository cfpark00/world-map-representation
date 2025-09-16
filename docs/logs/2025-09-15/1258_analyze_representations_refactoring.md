# Analyze Representations Script Refactoring

## Date: 2025-09-15 12:58

## Summary
Major refactoring of `analyze_representations.py` to centralize prompt format logic, remove unused code, and add support for new triangle area task formats. Also created multi-layer R² comparison visualization tool.

## Major Accomplishments

### 1. Centralized Prompt Format Logic
**Problem**: Prompt format logic was duplicated in two places (prompt creation and representation extraction), violating DRY principle.

**Solution**: Created `get_prompt_config()` function that returns:
- `prompt`: The formatted prompt string
- `extraction_indices`: List of token positions to extract
- `position_names`: Names for each extracted position

This makes adding new formats trivial - just add one elif branch instead of modifying multiple locations.

### 2. Supported Prompt Formats
After refactoring, the script now supports:
- `dist`: Distance task with last 3 tokens (comma, c, underscore)
- `dist_city_and_transition`: Distance task extracting 9 positions (full first city + transition)
- `dist_firstcity_last_and_comma`: Distance task extracting 2 positions (last digit, comma)
- `triarea_firstcity_last_and_comma`: Triangle area task extracting 2 positions (last digit, comma)
- `triarea_firstcity_last`: Triangle area task extracting only last digit

**Removed**: `rw200` format (random walk) as it wasn't being used

### 3. Fixed Causal Attention Issue
**Problem**: Some formats were including tokens after the extraction point (e.g., "c_" after comma), but due to causal attention, tokens can't see future positions.

**Solution**: Truncated prompts to end at the last extracted position:
- `dist_firstcity_last_and_comma`: Now ends at comma instead of including "c_"
- `triarea_firstcity_last_and_comma`: Similarly truncated
- `triarea_firstcity_last`: Ends at last digit

### 4. Removed Dead Code
- Removed `task_type` field which was required in configs but never used for any logic
- Only kept in metadata for reference, but made optional

### 5. Multi-Layer R² Visualization Tool
Created `/scratch/tempplots/plot_multi_layer_r2.py` that:
- Plots average R² scores from multiple layer analyses in one graph
- Supports layers 0-6
- Shows final R² values in text box
- Helps compare which layers encode location information best

### 6. Code Structure Improvements
- Renamed `dist_city_last_and_comma` → `dist_firstcity_last_and_comma` for consistency
- Better error handling with explicit ValueError for unknown formats
- Cleaner separation of concerns with extraction indices passed as parameters

## Technical Details

### Refactored Function Signature
```python
def analyze_checkpoint(..., extraction_indices=None, position_names=None):
    # Now receives extraction config instead of prompt_format
```

### New Prompt Config Structure
```python
{
    'prompt': prompt_string,
    'extraction_indices': [list of positions],
    'position_names': [list of names]
}
```

### Example Addition of New Format
To add a new format, only need to add one elif branch in `get_prompt_config()`:
```python
elif prompt_format == 'new_format':
    # Create prompt
    # Define extraction_indices
    # Define position_names
```

## Files Modified
- `/src/analysis/analyze_representations.py` - Major refactoring
- `/scratch/tempplots/plot_multi_layer_r2.py` - Created for visualization

## Next Steps Suggested
- Could further modularize by creating a PromptFormat class/enum
- Consider adding validation for city ID lengths vs extraction indices
- Could cache tokenization results since prompts are deterministic given city IDs