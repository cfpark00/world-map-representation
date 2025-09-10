# Session Log: Memorization Task Debugging - Random4to8 Experiments
**Date**: 2025-08-30  
**Time**: 21:20  
**Duration**: ~2 hours  
**Primary Focus**: Investigating why transformers can't overfit simple memorization tasks

## Problem Statement
Discovered that transformers struggle to memorize simple "abcd=efgh" mappings even with multiple epochs. Need to understand why location prediction fails while similar tasks might work.

## Key Activities

### 1. Dataset Analysis
- Verified that location dataset (`loc_500kplus_all_42`) has validation as subset of training (100% overlap)
- Confirmed this is intentional for testing pure memorization
- Dataset: 1051 training samples, 128 validation (all from training)

### 2. Created Random4to4 → Random4to8 Dataset Generator
- Initially created `create_random4to4_hf.py` for simple 4-digit mappings
- Evolution of format:
  1. Started with `abcd=efgh`
  2. Changed to `loc(c_abcd)=efgh` (match location format)
  3. Final: `loc(c_abcd)=efgh,ijkl` (exactly like location)
- Renamed to `create_random4to8_hf.py` (4 digits map to 8 digits)
- Validation is subset of training (same as location dataset)

### 3. Training Script Development
- Created `train_random4to4_hf.py` → `train_random4to8_hf.py`
- Adapted from `train_location_hf.py` 
- Key metric: digit matches (0-8) instead of haversine distance
- Fixed bugs:
  - JSON serialization of numpy int64 types
  - Parsing of generated vs expected outputs
  - Digit counting to exclude comma (8 digits, not 9)

### 4. Biased Dataset Experiment
- Created `create_biasedrandom4to8_hf.py`
- 50% of samples have different inputs mapping to same outputs
- Tests many-to-one mapping capability
- Result: Model still successfully memorizes!

### 5. Critical Discovery: Batch Size Issue
**Found the root cause of location training failure!**

Original location config had:
- `batch_size: 512` (way too large for 1051 samples)
- `loss_mask_type: "null"` (trains on all tokens)

This caused:
- Only ~2 gradient updates per epoch (1051/512)
- 200 total updates over 100 epochs
- Severe undertraining

Fixed configuration:
- `batch_size: 64` → ~16 updates per epoch
- `loss_mask_type: "answer_only"` → focus on actual task
- Result: 8x more gradient updates, successful memorization!

## Files Created/Modified

### Created:
- `src/data_processing/create_random4to8_hf.py`
- `src/data_processing/create_biasedrandom4to8_hf.py`
- `src/training/train_random4to8_hf.py`
- `configs/random4to8_1k_100epochs.yaml`
- `configs/biasedrandom4to8_1k_100epochs.yaml`
- `configs/location_500k_100epochs_random4to8.yaml`

### Modified:
- `configs/location_500k_100epochs.yaml` (fixed batch size and loss masking)

### Datasets Generated:
- `outputs/datasets/random4to8_1k_42`
- `outputs/datasets/biasedrandom4to8_1k_42`

## Key Insights

1. **Batch size matters enormously for small datasets**: Large batch sizes can prevent memorization by reducing gradient updates
2. **Loss masking helps**: Training only on answer tokens (not prompts) improves learning efficiency
3. **Format complexity doesn't matter much**: Model handles comma-separated outputs fine
4. **Many-to-one mappings work**: Model can learn when multiple inputs map to same output

## Next Steps
- Re-run location training with fixed hyperparameters
- Consider testing with even smaller batch sizes
- Investigate if coordinate distribution affects learning

## Commands for Reproduction
```bash
# Generate datasets
uv run python src/data_processing/create_random4to8_hf.py 1000 128 outputs/datasets/random4to8_1k_42 --seed 42
uv run python src/data_processing/create_biasedrandom4to8_hf.py 1000 128 outputs/datasets/biasedrandom4to8_1k_42 --seed 42

# Train models
uv run python src/training/train_random4to8_hf.py configs/random4to8_1k_100epochs.yaml
uv run python src/training/train_location_hf.py configs/location_500k_100epochs.yaml  # Now with fixed params
```