# Analysis Script Fixes and Enhancements
**Date:** 2025-09-13
**Time:** 01:41

## Summary
Fixed critical bugs in the representation analysis script and added new features for saving probe weights and extracted representations.

## Key Changes

### 1. Fixed Layer Indexing Bug
- **Issue:** The code incorrectly used `outputs.hidden_states[idx + 1]` assuming layer N was at index N+1
- **Reality:** Testing confirmed `hidden_states[0]` = embeddings, `hidden_states[N]` = layer N output
- **Fix:** Removed the incorrect `+1` offset in `analyze_representations.py` line 91

### 2. Atlantis Visualization Improvements
- **Removed hardcoded Atlantis color** from `region_colors` dictionary
- **Fixed highlight logic** to properly skip Atlantis in regular plotting when highlighted
- **Fixed text label issue** - Atlantis now gets a text label on the map like other regions
- **Highlight feature is now fully configurable** via YAML config

### 3. Added Probe Weight Saving
- **New feature:** Saves linear probe weights when `save_fits: true`
- **Output structure:**
  - `weights/step{step}_x_weights.npy` - X coordinate probe weights
  - `weights/step{step}_y_weights.npy` - Y coordinate probe weights
  - `weights/step{step}_intercepts.npz` - Both intercepts
  - `weights/step{step}_weights.png` - Visualization heatmap

### 4. Added Representation Saving
- **New config option:** `save_repr_ckpts: [-1]` to save representations at specific checkpoints
- **Fixed critical bugs:**
  - `analysis_dir` was undefined - moved definition before the loop
  - Checkpoint detection logic for `-1` was broken - fixed with enumerate index
  - Removed try-except block per repo guidelines (fail fast!)
- **Output structure:**
  - `representations/checkpoint-{step}/representations.pt` - PyTorch tensor with extracted representations
  - `representations/checkpoint-{step}/metadata.json` - City info and configuration

### 5. CSV Cleanup
- Prevented representations from being saved to CSV (was causing corruption)
- Used separate return value for representations to avoid CSV bloat

### 6. Created Minimal Notebook
- `load_repr.ipynb` - Minimalistic notebook to load saved representations and metadata

## Files Modified
- `/src/analysis/analyze_representations.py` - Main analysis script with all fixes
- `/load_repr.ipynb` - New minimal notebook for loading representations

## Bugs Fixed
1. Layer indexing off-by-one error
2. Atlantis not showing text labels on maps
3. Representations not being saved due to undefined `analysis_dir`
4. CSV corruption from numpy arrays
5. Try-except hiding errors (removed per repo guidelines)

## Testing Notes
- Verified layer indexing with test script in `scratch/test_hidden_states_indexing.py`
- Confirmed `-1` in `save_repr_ckpts` now correctly saves final checkpoint
- Atlantis cities properly highlighted with configurable colors

## Next Steps
- The analysis pipeline is now fully functional with proper layer extraction
- Probe weights and representations are saved for further analysis
- All visualization features working correctly