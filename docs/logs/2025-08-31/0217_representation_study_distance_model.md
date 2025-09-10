# Representation Study for Distance Model

**Date**: 2025-08-31
**Time**: 02:17
**Focus**: Creating comprehensive representation analysis notebook for dist_100k_1M_20epochs model

## Summary
Developed a detailed Jupyter notebook to analyze internal representations of the distance prediction model, implementing linear probes to understand what information is encoded at layer 4 of the transformer.

## Tasks Completed

### 1. Created Representation Study Notebook
Created `notebooks/representation_study.ipynb` with comprehensive analysis tools for studying model representations.

**Initial Setup:**
- Loaded dist_100k_1M_20epochs model checkpoint
- Set up representation extraction using PyTorch hooks at layer 4
- Fixed path issues for notebook execution from notebooks/ directory
- Fixed column name mismatches (asciiname vs City, no Population column)

### 2. Distance Prediction Linear Probe
Implemented linear probe training on distance prediction task representations.

**Setup:**
- Increased sample size from 100 to 500 city pairs
- Used 400 for training, 100 for testing
- Extracted representations at "=" token position (last token)

**Implementation:**
- Tested multiple regularization levels (Linear, Ridge with α=1.0, 10.0, 100.0)
- Added comprehensive visualizations (true vs predicted, error distributions)
- Compared linear probe performance with full model autoregressive generation

### 3. City Location Prediction from Partial Prompts
Developed novel analysis to predict city coordinates from partial prompt representations.

**Key Discovery - Tokenization Ambiguity:**
- Initial approach: `"dist(c_123"` - Poor performance due to ambiguity (is it c_123 or c_1234?)
- Improved: `"dist(c_123,"` - Better, comma signals end of city ID
- Further improved: `"dist(c_123,c"` - Even better, unambiguous parsing
- Final: `"dist(c_123,c_"` - Best results with full disambiguation

**Implementation Evolution:**
1. Started with single token representation (128 dims)
2. Evolved to concatenated representations (256 dims: comma + last token)
3. Final version uses 384 dims (128×3: comma + 'c' + '_' representations)

**Results:**
- Successfully trained linear probes for longitude and latitude prediction
- Added world map visualization showing prediction errors
- Lines connect true to predicted locations for visual error assessment

### 4. Key Insights

**Tokenization Boundaries Matter:**
- Model needs clear signals when a city ID is complete
- The sequence `c_123,c_` provides full disambiguation:
  - Comma ends first city ID definitively
  - 'c' signals second city context
  - '_' prepares for receiving digits

**Representation Quality:**
- Layer 4 representations contain strong linear signal for both:
  - Distance prediction between city pairs
  - Geographic location of individual cities
- Concatenating multiple token positions improves probe performance

### 5. Bug Fixes and Improvements

**Fixed Issues:**
- Column name references (City → asciiname)
- Path issues for notebook execution (added ../ prefix)
- Variable name collision (model → probe_models to avoid overriding transformer)
- Simplified representation extraction using left padding properties

**Code Optimizations:**
- Replaced loop-based last token extraction with simple indexing ([:, -1, :])
- Leveraged left padding for consistent token positions

## Files Modified
- Created: `notebooks/representation_study.ipynb`
- Modified: Structure reflected in `analysis/layer4_representations.pkl` (output from notebook)

## Technical Details

**Model Architecture:**
- Qwen2-based transformer
- 6 layers, 128 hidden dimensions, 4 attention heads
- Trained on 1M city pair distance predictions

**Representation Extraction:**
- Used forward hooks on transformer layers
- Captured outputs after layer 4 (5th layer, 0-indexed)
- Extracted at specific token positions based on task

**Linear Probe Training:**
- Ridge regression with cross-validation
- 750/250 train/test split for coordinate prediction
- 400/100 split for distance prediction analysis

## Next Steps Potential
- Analyze other layers (0-3, 5) for comparison
- Study attention patterns at disambiguation points
- Investigate how representations change during training
- Compare with location prediction model representations