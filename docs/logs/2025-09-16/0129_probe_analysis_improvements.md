# Probe Analysis Improvements and Fixes

**Date**: 2025-09-16
**Time**: 01:29 EDT
**Focus**: Enhanced analyze_representations.py with new features and critical bug fixes

## Summary
Implemented support for randomwalk prompt format, added visualization features, fixed critical train/test overlap bug, and created regional analysis configs.

## Major Changes

### 1. Added Randomwalk Prompt Format Support
- Implemented `randomwalk_firstcity_last_and_comma` prompt format in `analyze_representations.py`
- Format: `rw(max_dist,chain_len)=c_XXXX,`
- Correctly handles character-level tokenization (spaces are delimiters, not tokens)
- Extracts representations at last digit of city ID (position 17) and comma (position 18)
- Uses average parameters: max_dist=275, chain_len=12 (based on typical dataset ranges)

### 2. Fixed Critical Train/Test Sampling Bug
**Problem**: When `probe_train` and `probe_test` patterns were identical, the code was sampling independently from the same pool, causing data leakage through overlap.

**Solution**: Implemented robust 3-step sampling:
1. Sample training set from cities matching train pattern
2. Remove training cities from test candidates
3. Sample test set from remaining cities
- Added proper error checking with descriptive messages
- Guarantees no overlap between train and test sets

### 3. New Visualization Features

#### plot_links Feature
- Added `plot_links` parameter (default: False)
- When True, draws lines connecting true to predicted city positions
- Helps visualize prediction errors as displacement vectors
- Different styling for highlighted vs regular cities

#### plot_autolim Feature
- Added `plot_autolim` (alias: `plot_autobox`) parameter (default: False)
- Automatically zooms map to show only the data region with 10% padding
- **Maintains equal aspect ratio** for geographic accuracy
- Uses larger dimension (width or height) for both axes to create square view

### 4. Probe Training Improvements
- Modified probes to predict deviations from training mean
- Calculates x_train_mean and y_train_mean
- Centers targets before training: `x_centered = x - x_mean`
- Adds means back for final predictions
- Improves probe stability and interpretability
- Stores training means in metadata and weights files

### 5. Regional Analysis Configs
Created YAML configs for regional probe analysis in `/configs/analysis_representation/randomwalk_pretrain_no_atlantis_15ep_llr_pad/regions/`:

- Individual regions: Africa, North America, South America, Western Europe, Eastern Europe, Middle East, India, China, Japan, Korea, Southeast Asia, Central Asia, Oceania, Antarctica
- Combined region: `asia_major` (China + India + Japan + Korea)

Each config includes:
- Regional filtering via regex patterns
- Adjusted city counts based on availability
- `plot_links: true` and `plot_autobox: true` for enhanced visualization

## Technical Details

### Tokenization Clarification
- Tokenizer splits on spaces: `"r w ( 2 7 5"` → `['r', 'w', '(', '2', '7', '5']`
- Spaces are delimiters, NOT tokens
- Total tokens for randomwalk prompt: 19 (not counting spaces)

### City Distribution
Analyzed city counts per region:
- Largest: Africa (705), North America (643), China (640), India (597)
- Smallest: Oceania (30), Central Asia (56), Korea (69)
- 100 unmapped cities with code "XX0" (likely Atlantis/test cities)
- Fixed NA (Namibia) being parsed as NaN

## Files Modified
- `/src/analysis/analyze_representations.py` - Major enhancements and bug fixes
- Created 15 new YAML config files for regional analysis

## Next Steps
- Run regional analyses to compare probe performance across different geographic areas
- Investigate whether the model learns different representation structures for different regions
- Consider adding more task-specific prompt formats