# Development Log: WM_1 Experiments Setup and Infrastructure
**Date:** 2025-08-30  
**Time:** 17:28  
**Main Topic:** World Model experiments infrastructure, tokenizer creation, and training preparation

## Summary
Conducted extensive data analysis, created new dataset generation scripts, established a unified tokenizer system, reorganized project structure, and prepared training infrastructure for location prediction experiments.

## Major Accomplishments

### 1. Spatial Data Analysis
- Analyzed 5,075 cities with population ≥100k using scipy's cKDTree
- Found that cities have mean 18.72 neighbors within 200km (median: 11)
- Identified clustering patterns: 33.5% have ≥20 neighbors, only 0.1% have ≥100
- Most connected city: Kōfu, Japan (105 neighbors)
- Created visualization script with histogram output in `/analysis/`

### 2. Random Walk Dataset Generator
- Created `create_randomwalk_dataset_hf.py` for sequential city traversal
- Format: `srd_200=c_1234,c_5678,c_9012,...` (sequential random distance)
- Features:
  - Random walk through cities within 200km distance constraint
  - Configurable max length (1-32 cities)
  - Sequence ends when no neighbors found
  - Visualization capability with world map plotting
  - Successfully tested with varied sequence lengths (mean ~13 cities)

### 3. Unified Tokenizer System
- Created character-level tokenizer with exactly 44 tokens:
  - 3 special tokens: `<bos>`, `<eos>`, `<pad>`
  - 5 grammar tokens: `(`, `)`, `,`, `=`, `_`
  - 26 lowercase letters: a-z
  - 10 digits: 0-9
- Fixed initial issue (removed quote character that made it 45)
- Created HuggingFace-compatible tokenizer saved at `outputs/tokenizer/wm1_tokenizer`
- Can be loaded with standard `AutoTokenizer.from_pretrained()`

### 4. Project Structure Reorganization
- Renamed `scripts/` → `src/` for better convention
- Moved tokenizer outputs to `outputs/tokenizer/`
- Updated README.md with comprehensive folder structure documentation:
  - `configs/`: Configuration files
  - `src/`: Main source code
  - `outputs/`: All generated artifacts (datasets, models, figures, tokenizer)
  - `data/`: Raw input data
  - `notebooks/`: Interactive Jupyter notebooks
  - `analysis/`: Analysis scripts (to be merged into src)
  - `claude_notes/`: Session tracking
  - `scratch/`: Temporary workspace
- Fixed all import paths after restructuring

### 5. Location Dataset Enhancement
- Modified `create_location_dataset_hf.py` to support validation splits
- Added `--n_val` argument for creating validation sets
- Fixed `--all` flag to properly set `n_train = n_cities`
- Changed sampling to `replace=False` for unique samples
- Created `loc_100kplus_all_42` with 5,075 train + 128 validation samples

### 6. Training Script Development
- Created `train_location.py` specifically for location prediction
- Features:
  - Loads HuggingFace tokenizer (44 tokens)
  - Handles location dataset format (`loc(c_XX)=Y,Z`)
  - Automatic validation split detection and usage
  - Answer-only loss masking (trains only on coordinates)
  - Checkpoint saving and training curves
  - Prepared for generation-based evaluation with haversine distance

### 7. Training Configuration
- Created `location_training.yaml` config:
  - Dataset: `/outputs/datasets/loc_100kplus_all_42`
  - Model: Qwen2.5-like (64 hidden, 4 layers, 4 heads)
  - Training: 5 epochs, batch size 512
  - Uses 44-token custom tokenizer
  - Saves checkpoints 5x per epoch, evaluates 10x per epoch

## Key Files Created/Modified

### New Files
- `/src/data_processing/create_randomwalk_dataset_hf.py`
- `/src/tokenizer/tokenizer_config.py`
- `/src/tokenizer/create_hf_tokenizer.py`
- `/src/training/train_location.py`
- `/configs/location_training.yaml`
- `/analysis/spatial_analysis_csv.py`

### Modified Files
- `/README.md` - Complete restructuring documentation
- `/src/data_processing/create_location_dataset_hf.py` - Added validation support
- `/src/data_processing/create_distance_dataset_hf.py` - Removed zero-padding
- `/notebooks/load_dataset.ipynb` - Added tokenizer testing cells

## Technical Decisions
1. Used character-level tokenization for precise control over vocabulary
2. Removed zero-padding from city IDs to handle arbitrary numbers of cities
3. Chose `replace=False` for dataset sampling to ensure unique samples
4. Implemented sequential random walks with distance constraints for spatial relationship learning
5. Separated training scripts by task type for better maintainability

## Next Steps (Prepared)
- Run location prediction training with: `python src/training/train_location.py configs/location_training.yaml`
- Implement generation-based evaluation with haversine distance metrics
- Explore different distance thresholds for random walk datasets
- Consider implementing location coordinate prediction evaluation

## Notes
- All dataset generation scripts now properly handle arbitrary numbers of cities
- Tokenizer is fully compatible with HuggingFace ecosystem
- Project structure follows standard ML project conventions
- Ready for large-scale experiments with proper train/validation splits