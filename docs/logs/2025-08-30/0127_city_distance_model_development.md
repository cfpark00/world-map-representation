# Development Log: City Distance Prediction Model
**Date:** 2025-08-30  
**Time:** 01:27  
**Main Topic:** Transformer model for geodesic distance prediction

## Summary
Built a complete pipeline for training a small Qwen2.5-like transformer model to predict geodesic distances between cities based on their IDs.

## Major Accomplishments

### 1. Data Pipeline
- Created filtered city datasets from GeoNames data (50k+ and 100k+ population thresholds)
- Built HuggingFace dataset creator with train/validation/test splits
- Updated to use argparse for clean CLI interface
- Added three columns to dataset: `text`, `prompt`, and `completion` for generation tasks
- Dataset format: `d(c_XXXX,c_XXXX)=YYYY` where XXXX are city IDs and YYYY is distance in km

### 2. Model Architecture
- Implemented small Qwen2.5 configuration:
  - 64 hidden dimensions
  - 4 layers
  - 4 attention heads  
  - RoPE positional embeddings
  - Custom tokenizer with 20 tokens (BOS, EOS, PAD, d, c, _, (, ), ,, =, 0-9)
  - Max sequence length: 32 tokens

### 3. Training Infrastructure
- Created comprehensive training script with:
  - Config-based training from YAML
  - tqdm progress bars
  - Automatic checkpointing at configurable intervals
  - Validation with MSE computation for distance predictions
  - Dual plot system: loss curves and MSE tracking
  - Warmup + cosine learning rate schedule
  - Error handling for experiment directory conflicts

### 4. Visualization
- Created city map plotting scripts (equirectangular projection)
- Population distribution histograms
- Training loss plots with log scale, capped at y=1
- MSE tracking with penalty (20,000²) for parsing errors

### 5. Project Organization
```
WM_1/
├── configs/               # Training configurations
├── scripts/
│   ├── data_processing/   # Dataset creation
│   ├── training/         # Training and testing
│   └── visualization/    # Plotting scripts
├── outputs/
│   ├── datasets/         # Processed data
│   ├── figures/          # Visualizations
│   └── experiments/      # Training runs
└── data/                 # Raw GeoNames data
```

## Key Design Decisions
1. **Removed try-except blocks** in MSE computation for better debugging
2. **Single loss plot file** that overwrites (not multiple timestamped files)
3. **Argparse everywhere** instead of positional arguments
4. **Config path required** as argument (no defaults)
5. **Separate eval batch size** for faster validation
6. **Dataset saves directly to specified path** without creating redundant subfolders

## Technical Details
- Fixed YAML parsing issues with proper type conversion at config load
- Handled tokenization properly with character-level encoding
- Used HuggingFace's generate() for inference during validation
- Implemented proper train/val/test split with guaranteed no overlap

## Files Created/Modified Today
- `scripts/data_processing/create_distance_dataset_hf.py` - Main dataset creator
- `scripts/data_processing/create_filtered_dataset.py` - City filtering
- `scripts/training/train.py` - Full training pipeline
- `scripts/training/batch_test.py` - Testing script with generation
- `scripts/visualization/create_city_map.py` - Map visualization
- `scripts/visualization/create_population_histogram.py` - Distribution plots
- `configs/training_config.yaml` - Training configuration
- `README.md` - Project documentation

## Next Steps
- Run full training experiments
- Tune hyperparameters based on MSE curves
- Potentially increase model size if needed
- Add more sophisticated evaluation metrics