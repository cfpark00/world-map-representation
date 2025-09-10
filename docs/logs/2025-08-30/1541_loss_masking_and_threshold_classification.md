# Development Log: Loss Masking and Distance Threshold Classification
**Date:** 2025-08-30  
**Time:** 15:41  
**Main Topic:** Implementing loss masking and binary classification for distance thresholds

## Summary
Extended the WM_1 project with selective loss masking for answer-only training and created a new binary classification task for distance thresholds.

## Major Accomplishments

### 1. Loss Masking Implementation
- Added `loss_mask_type` parameter to training configurations (test_1.yaml, test_2.yaml, test_3.yaml)
- Implemented two modes in train.py:
  - `None`: Standard next-token prediction on all tokens
  - `"answer_only"`: Loss computed only on answer tokens (YYYY<eos> in distance format)
- Modified DistanceDataset class to apply -100 masking to ignored tokens
- Updated batch_test.py to visualize masked labels with [MASKED] indicator

### 2. Metric Changes
- Changed evaluation metric from MSE to MAE (L1 distance)
- More interpretable: directly represents average error in kilometers
- Grammar/parsing errors counted as max_error (20,000 km) instead of squared

### 3. Distance Threshold Binary Classification Task
- Created `create_distancethres_dataset_hf.py` for new task format
- Format: `dt(c_XXXX,c_YYYY,ZZZZ)={0,1}` where:
  - XXXX, YYYY: 4-digit city IDs
  - ZZZZ: Distance threshold in km (no leading zeros)
  - Output: 1 if cities within threshold, 0 otherwise
- Default threshold: 2000 km
- Discovered significant class imbalance:
  - ~90.74% Class 0 (outside threshold)
  - ~9.26% Class 1 (within threshold)
  - Imbalance ratio: ~10:1

### 4. Training Script Adaptation
- Created `train_threshold.py` specifically for binary classification
- Updated tokenizer vocabulary to include 't' token (21 total tokens)
- Replaced MAE with F1 score, precision, recall metrics
- Changed from regression to classification evaluation
- Adjusted generation to produce only 0/1 + <eos> tokens

### 5. Data Analysis Tools
- Created analysis scripts in scratch/:
  - `calc_avg_distance.py`: Computes dataset statistics (avg distance: 7,945 km)
  - `analyze_threshold_balance.py`: Quantifies class imbalance

## Key Insights

### Model Behavior on Imbalanced Data
- Observed mode collapse: model predicts only majority class
- Achieves 91.41% accuracy but 0% F1 score
- Demonstrates why F1 is crucial for imbalanced classification

### Technical Details
- Token mappings for threshold task:
  - 'd'=3, 't'=4, 'c'=5, '='=10
  - '0'=11, '1'=12 (for binary output)
- Loss masking with answer_only:
  - Distance task: masks everything except YYYY<eos>
  - Threshold task: masks everything except {0,1}<eos>

## Files Created/Modified

### New Scripts
- `scripts/data_processing/create_distancethres_dataset_hf.py`
- `scripts/training/train_threshold.py`
- `scratch/calc_avg_distance.py`
- `scratch/analyze_threshold_balance.py`

### Modified Files
- `scripts/training/train.py` (MAE instead of MSE, loss masking)
- `scripts/training/batch_test.py` (loss masking visualization)
- All config files (added loss_mask_type parameter)

## Next Steps/Recommendations
1. Implement class weighting or balanced sampling to address mode collapse
2. Consider focal loss for better handling of class imbalance
3. Experiment with different distance thresholds for better class balance
4. Add early stopping based on F1 score rather than loss

## Dataset Locations
- Distance datasets: `outputs/datasets/dist_*`
- Threshold datasets: `outputs/datasets/distthres*`
- Test dataset with 1M samples: `/n/home12/cfpark00/WM_1/outputs/datasets/distthres2000_100kplus_1M_42`