# Development Log - 2025-09-12 01:39

## Session: Analysis Script Debugging and Tokenization Fixes

### Summary
Major debugging session focused on fixing the representation analysis pipeline. Discovered and fixed critical tokenization issues, updated coordinate system from lon/lat to x/y, integrated old notebook's RepresentationExtractor class, and removed problematic code.

### Key Issues Identified and Fixed

#### 1. Critical Tokenization Bug
- **Problem**: `create_space_delimited_prompt()` was destroying special tokens
  - Converting `<bos>` → `< b o s >` (5 separate tokens instead of 1)
  - This completely broke model's understanding of prompts
- **Solution**: 
  - Initially fixed the function to preserve special tokens
  - Later realized function was unnecessary and removed it entirely
  - Updated analysis script to match training data format exactly

#### 2. Coordinate System Mismatch
- **Problem**: Analysis script using longitude/latitude and haversine distance
- **Reality**: Project uses x/y coordinate system (scaled by 10)
- **Fixed**:
  - Replaced all lon/lat references with x/y
  - Removed haversine distance calculation
  - Used Euclidean distance instead
  - Updated visualization to scale coordinates properly (/10 for display)

#### 3. Region Mapping Improvements
- **Problem**: Huge hardcoded country_to_region dictionary in Python file
- **Solution**:
  - Removed 88+ lines of hardcoded mappings
  - Load from JSON file: `data/geographic_mappings/country_to_region.json`
  - Added config parameter `region_mapping_path`
  - Proper path resolution relative to project root

#### 4. RepresentationExtractor Class Integration
- **Found**: Clean hook-based class in old notebook
- **Integrated**: Created `src/representation_extractor.py`
- **Features**:
  - Proper hook lifecycle management
  - Supports single or multiple layer extraction
  - Context manager support
  - Validation and error handling
- **Updated**: `analyze_representations.py` to use the class

#### 5. Device Handling Issues
- **Problem**: Shape mismatch error `[1, 3000]` - tensors on wrong device
- **Solution**: Ensure tensors moved to correct device for each checkpoint

### Files Modified

#### Created:
- `/n/home12/cfpark00/WM_1/src/representation_extractor.py` - Hook-based representation extraction class
- `/n/home12/cfpark00/WM_1/old_commits/` - Directory for reference commits (gitignored)

#### Modified:
- `/n/home12/cfpark00/WM_1/src/analysis/analyze_representations.py`
  - Complete overhaul: x/y coordinates, removed haversine, fixed tokenization
  - Integrated RepresentationExtractor class
  - Fixed device handling
  - Removed `create_space_delimited_prompt` usage
  
- `/n/home12/cfpark00/WM_1/src/utils.py`
  - Deleted evil `create_space_delimited_prompt` function
  
- `/n/home12/cfpark00/WM_1/.gitignore`
  - Added `old_commits/` directory
  
- `/n/home12/cfpark00/WM_1/configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1.yaml`
  - Added `region_mapping_path` parameter

### Key Discoveries

1. **Training data is correct**: Uses format `<bos> d i s t ( c _ 1 2 3 ...` with space after `<bos>` but token preserved
2. **Analysis was broken**: Was creating `< b o s > d i s t ...` with broken special tokens
3. **Old notebook had better methodology**: Used hooks properly, correct tokenization
4. **Coordinate system**: Entire project uses x/y scaled by 10, not lon/lat degrees

### Impact
- Analysis script should now work correctly
- Results should match the old notebook's better performance
- No need to retrain models - training data was correct all along
- Only analysis scripts needed fixing

### Next Steps
- Run the fixed analysis script to verify it works
- Compare results with old notebook performance
- Consider cleaning up more legacy code

### Technical Notes
- Special tokens must NEVER be split by character-level tokenization
- Always verify tokenization matches between training and inference
- Hook-based extraction provides more control than `output_hidden_states`
- Device management crucial when passing tensors between functions

### Pulled Old Commit for Reference
- Commit: `292d758fb7fc8daff92e67cc45cb3d3108e022c8`
- Location: `/n/home12/cfpark00/WM_1/old_commits/`
- Purpose: Reference for better methodology, especially the notebook