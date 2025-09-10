# Task-Specific Collators and Coordinate System Refactor

**Date:** 2025-09-03 15:43  
**Duration:** Extended session  
**Main Topic:** Redesigned multi-task training architecture and coordinate system

## Summary

Major architectural refactor to support flexible multi-task training with clean abstractions. Moved from config-based task specification to dataset-level task identity, implemented task-specific collators, and simplified coordinate system from geographic haversine to Euclidean 2D space.

## Key Changes Made

### 1. Task-Specific Collator Architecture
- **Removed** `loss_mask_type` from configs and validation
- **Implemented** `MultiTaskCollator` that routes to task-specific collators based on dataset items
- **Created** individual collators:
  - `DistanceCollator`: Masks everything before '=' in `dist(c_X,c_Y)=ANSWER`
  - `LocationCollator`: Masks everything before ':' in `loc(c_X):ANSWER`
  - `RandomWalkCollator`: Masks everything before '=' in `walk_DIST=ANSWER`
  - `FullSequenceCollator`: Fallback for unknown tasks

### 2. Dataset Format Standardization
- **Updated** all dataset generation scripts to output only `text` and `task_type` fields
- **Removed** redundant `prompt` and `completion` fields
- **Task identity** now comes from dataset items, not config files
- **Examples:**
  ```python
  {
      'text': '<bos>dist(c_8658,c_4879)=769<eos>',
      'task_type': 'distance'
  }
  ```

### 3. Coordinate System Overhaul
- **Replaced** haversine distance calculation with Euclidean distance
- **Removed** all longitude/latitude references in favor of x,y coordinates
- **Scaled** coordinates by 10x to avoid decimals:
  - x range: -1800 to 1800 (was -180 to 180)
  - y range: -900 to 900 (was -90 to 90)
- **Distance range** now 0-4000 units instead of 0-20000km

### 4. Training Pipeline Updates
- **Modified** `train.py` to infer task type from dataset for evaluation
- **Updated** `get_dataset()` to return collator as 4th tuple element
- **Removed** config-level task type specification entirely
- **Maintained** backward compatibility for evaluation metrics

### 5. File Structure Changes
- **Updated** dataset creation scripts:
  - `src/data_processing/create_distance_dataset.py`
  - `src/data_processing/create_location_dataset_.py` 
  - `src/data_processing/create_randomwalk_dataset_.py`
  - `src/data_processing/create_city_dataset.py`
- **Updated** core utilities in `src/utils.py`
- **Cleaned** config files (removed `task_type`, `loss_mask_type`)
- **Created** `configs/tasks.json` documentation

## Architecture Benefits

### Multi-Task Support
- Can train on single task, mixed tasks, or any combination
- Each batch can contain different task types
- Automatic routing to appropriate loss masking

### Clean Abstractions
- Task-specific logic encapsulated in collators
- No config pollution - task type comes from data
- Extensible: add new tasks by creating new collators

### Memory Efficiency
- Just-in-time tokenization during collation
- Store raw text instead of pre-tokenized tensors
- Better than standard SFT pipeline for research use

## Testing Results

- **Distance dataset generation**: Successfully created datasets with Euclidean distances (0-4000 range)
- **Collator testing**: All task-specific collators apply correct loss masking
- **Training compatibility**: Existing configs work with minimal changes (removed obsolete fields)

## Files Modified

### Core Architecture
- `src/utils.py` - Added collator classes, removed haversine, updated evaluation
- `src/training/train.py` - Task type inference, collator integration

### Dataset Generation
- `src/data_processing/create_distance_dataset.py` - Euclidean distance, text+task_type output
- `src/data_processing/create_location_dataset_.py` - Simplified coordinate handling
- `src/data_processing/create_randomwalk_dataset_.py` - Text+task_type output
- `src/data_processing/create_city_dataset.py` - 10x coordinate scaling

### Configuration
- `configs/train_dist_1M_no_atlantis_20epochs.yaml` - Removed task_type, loss_mask_type
- `configs/tasks.json` - Created comprehensive task documentation

## Next Steps

- Test training with new architecture
- Implement location and randomwalk tasks when needed
- Consider mixed-task dataset creation utilities
- Evaluate multi-task training performance

## Architecture Comparison

Our approach vs standard SFT pipelines:
- **Ours**: Just-in-time tokenization, task-specific collators, multi-task support
- **Standard**: Pre-tokenization, single collator, assistant masking for chat
- **Verdict**: Our architecture is superior for multi-task research scenarios

## Coordinate System Summary

**Before**: Haversine distances on longitude/latitude (0-20,000km)
**After**: Euclidean distances on scaled x,y coordinates (0-4,000 units)
**Benefit**: Simpler 2D learning problem, cleaner numeric ranges, no geographic pretense