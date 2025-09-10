# Representation Dynamics Analysis Development
**Date**: 2025-08-31  
**Time**: 03:49  
**Session Focus**: Creating comprehensive representation dynamics analysis script

## Summary
Developed a complete Python script to analyze how internal representations evolve during model training, tracking R² scores for geographic location prediction and generating visualizations.

## Key Accomplishments

### 1. **Script Development** (`analysis/representation_dynamics.py`)
- Created flexible command-line script taking experiment directory and cities CSV as arguments
- Implemented layer-wise representation extraction using transformer hooks
- Added support for analyzing multiple layers simultaneously (e.g., `--layers 3,4`)

### 2. **Representation Extraction Issues & Solutions**
- **Initial Problem**: Hook-based extraction was returning 2D tensors instead of 3D
- **Diagnosis**: Model hooks were somehow flattening batch dimensions
- **Solution**: Switched to using `output_hidden_states=True` directly instead of hooks
- **Result**: Successfully extracted and concatenated representations from multiple layers

### 3. **Location Probe Implementation**
- Extracted representations from partial prompts: `"dist(c_XXX,c_"`
- Concatenated last 3 token representations (comma, 'c', underscore)
- Trained Ridge regression probes for longitude/latitude prediction
- Achieved test R² of 0.956 for longitude, 0.923 for latitude at final checkpoint

### 4. **Loss Tracking Integration**
- Added extraction of training loss from `trainer_state.json` files
- Found loss values in checkpoint directories' log history
- Integrated loss tracking into results DataFrame

### 5. **Visualization Improvements**
- **Dynamics Plot**: Clean 3-panel layout showing:
  - Training loss vs steps
  - Test R² scores (longitude, latitude, average) vs steps
  - Mean distance error vs steps (log scale with reference lines)
  
- **World Map Animation**: 
  - Integrated detailed region/country mapping from notebook
  - Color-coded cities by geographic regions (North America, Europe, Asia, etc.)
  - Generated animated GIF showing prediction evolution across checkpoints

### 6. **Region Mapping**
- Implemented comprehensive country-to-region mapping covering:
  - Major continents and subcontinents
  - Detailed regional divisions (Western/Eastern Europe, Southeast Asia, etc.)
  - Special regions (India subcontinent, Greater China, Korea, Japan)
  - Total of 14 distinct regions with unique colors

## Technical Details

### Key Functions:
- `analyze_checkpoint()`: Analyzes single checkpoint, returns R² and predictions
- `create_world_map_frame()`: Creates single animation frame with region colors
- `RepresentationExtractor`: Class for extracting layer representations (though ultimately used direct outputs)

### Output Files Generated:
- `representation_dynamics.csv`: Full results table
- `representation_dynamics_layers{layers}.png`: Dynamics plot
- `world_map_evolution_layers{layers}.gif`: Animated world map

## Results for dist_100k_1M_20epochs

### Initial (Step 3908):
- Longitude R²: -0.134
- Latitude R²: -0.155  
- Distance Error: 6875 km
- Loss: 1.847

### Final (Step 39080):
- Longitude R²: 0.956
- Latitude R²: 0.923
- Distance Error: 993 km
- Loss: 1.087

### Improvements:
- Longitude R²: +1.090
- Latitude R²: +1.079
- Distance Error: -5882 km

## Files Created/Modified
- Created: `/analysis/representation_dynamics.py` (main analysis script)
- Modified: `/notebooks/representation_dynamics.ipynb` (copied from representation_study.ipynb)
- Generated: Multiple output files in experiment directories

## Challenges Overcome
1. Fixed tensor dimension issues with layer extraction
2. Resolved batching problems causing inconsistent sample counts
3. Integrated loss extraction from trainer state files
4. Properly handled country code mappings for visualization

## Next Steps Potential
- Could extend to analyze different task types (location, randomwalk)
- Could add more probe types (country classification, continent prediction)
- Could analyze attention patterns in addition to representations
- Could create comparative analysis across different experiments