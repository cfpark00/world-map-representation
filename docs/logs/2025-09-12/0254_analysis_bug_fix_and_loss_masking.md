# Development Log - 2025-09-12 02:54

## Summary
Fixed critical bug in representation analysis, reorganized old_commits directory, and implemented loss masking feature for distance datasets.

## Major Tasks Completed

### 1. Fixed Representation Analysis Bug
- **Issue**: Getting null regression results when extracting layers 3,4 residuals
- **Root Cause**: When using `output_hidden_states=True`, the hidden_states tuple includes embedding layer at index 0, requiring `idx+1` indexing
- **Fix**: Replaced hook-based `RepresentationExtractor` with direct use of `output_hidden_states` with proper indexing
- **Files Modified**:
  - `/src/analysis/analyze_representations.py` - Fixed layer extraction logic
  - Removed unused `RepresentationExtractor` import

### 2. Added Single Checkpoint Analysis Support
- **Feature**: Can now analyze a single checkpoint instead of all
- **Implementation**: Added `checkpoint` parameter to analysis configs
  - Can be a number (e.g., `52740`)
  - Can be `"final"` for final checkpoint
  - If omitted, analyzes all checkpoints (original behavior)
- **Files Modified**:
  - `/src/analysis/analyze_representations.py` - Added checkpoint parameter handling
  - Created example config: `/configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1_checkpoint52740.yaml`

### 3. Reorganized old_commits Directory
- **Issue**: Files were dumped directly in `/old_commits/` instead of organized by commit hash
- **Fix**: 
  - Moved existing files to `/old_commits/aug31_snapshot/`
  - Extracted commit b9c7d24 to `/old_commits/b9c7d24/`
- **New Structure**:
  ```
  old_commits/
  ├── aug31_snapshot/  (previously loose files)
  └── b9c7d24/        (specific commit extraction)
  ```

### 4. Implemented Loss Masking for Distance Datasets
- **Feature**: Added loss masking to train only on answer portion of distance tasks
- **Implementation**:
  - Dataset always generates `loss_mask` field (1:1 mapping to tokens)
    - `0` = mask token (no loss)
    - `1` = include token in loss
    - Masks prompt up to and including `=`
    - Trains on answer + `<eos>`
  - Training pipeline uses mask when `use_loss_mask: true` in config
- **Files Modified**:
  - `/src/data_processing/create_distance_dataset.py` - Always generates loss_mask
  - `/src/utils.py` - Updated `MultiTaskCollator` to support loss masking
  - Default behavior: `use_loss_mask: false` (uses old task-specific masking)

## Key Insights

### Representation Extraction Bug
- The bug was subtle: hooks capture layer outputs directly, but `output_hidden_states` includes embeddings at index 0
- This caused probing of wrong layers (2,3 instead of 3,4), resulting in poor regression

### Data Generation Observations
- Both old (b9c7d24) and current versions use random pair swapping (no c1<c2 bias)
- Major differences between versions:
  - Coordinate system: old used lon/lat, current uses scaled x/y (*10)
  - Tokenizer: old vocab_size=44, current vocab_size=98
  - Directory structure: old used `outputs/`, current uses `data/`

### Loss Masking Design
- Clean separation: dataset provides mask, training decides whether to use it
- Token-aligned: mask string has one character per token
- Flexible: can be extended to other masking patterns beyond answer-only

## Files Changed
- `/src/analysis/analyze_representations.py`
- `/src/data_processing/create_distance_dataset.py`
- `/src/utils.py`
- `/configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1_checkpoint52740.yaml`
- `/configs/analysis/dist_pretrain/dist_1M_no_atlantis_probe1_ckpt46880.yaml`
- `/old_commits/` directory restructuring

## Next Steps
- Regenerate distance datasets with loss_mask field
- Test loss masking in training runs
- Compare performance with and without answer-only masking