# Development Log - 2025-09-20 16:42
## Atlantis Clustering Investigation & Multi-Pattern Filter Implementation

### Summary
Investigated why Atlantis cities form two distinct clusters in neural network representations, enhanced the dataframe filtering utility to support multi-pattern composition, and added mixed projection mode to PCA visualization tool.

### Major Changes

#### 1. Enhanced filter_dataframe_by_pattern in utils.py
**File**: `src/utils.py`
- Created backup function `filter_dataframe_by_pattern_backup` for safety
- Added support for multi-pattern composition:
  - List of patterns (implicit AND)
  - Dict with AND/OR keys
  - Inline operators `&&` and `||` in strings
- Added automatic type detection for numeric columns (converts to string before regex matching)
- Fully backward compatible with existing single-pattern usage

**Examples of new syntax**:
```yaml
# Inline AND
probe_train: "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$"

# List (implicit AND)
probe_train:
  - "region:^(?!Atlantis).*"
  - "city_id:^[1-9][0-9]{3,}$"
```

#### 2. Extended PCA Timeline Visualization
**File**: `src/analysis/visualize_pca_3d_timeline.py`
- Added support for "mixed" projection mode alongside existing PCA mode
- Mixed mode uses:
  - X-axis: Linear regression direction for x-coordinate prediction
  - Y-axis: Linear regression direction for y-coordinate prediction
  - Z-axis: Top PC of residual after projecting out x/y directions
- Added validation requiring `cities_csv` in config when using mixed mode
- Added `auto_play=False` to prevent autoplay on HTML load
- Fixed syntax errors in dictionary comprehensions

#### 3. Atlantis Clustering Investigation
**Created multiple analysis scripts in scratch/**:

**investigate_atlantis_clusters.py**:
- Analyzed 90 Atlantis cities (after filtering IDs >= 1000)
- Confirmed two strong clusters (silhouette score: 0.366, Cohen's d: 7.72)
- Tested multiple hypotheses for clustering pattern

**test_atlantis_hypothesis.py**:
- Tested odd/even, threshold, prime, and digit sum patterns
- Found no simple mathematical rule explaining the clusters

**analyze_atlantis_in_training.py**:
- Analyzed frequency of Atlantis cities in training data
- Both clusters have similar average frequency (~210 appearances)
- No correlation between training frequency and clustering

**check_atlantis_position.py**:
- Checked if position in training pairs (first vs second) correlates with clustering
- Both clusters show ~72% first-position ratio
- No significant difference (p=0.111)

**check_position_by_dataset.py**:
- Discovered key difference between datasets:
  - Distance dataset: 50% first position (balanced)
  - Compass dataset: 95% first position (heavily biased)
- Explains overall 72% average but doesn't explain clustering

### Key Findings

#### Atlantis Clustering Mystery
The two Atlantis clusters appear to be an **emergent property** of the model's internal optimization that cannot be traced to:
1. Mathematical patterns in suffix numbers
2. Training data frequency
3. Position in training pairs
4. Geographic coordinates

The model appears to have created an arbitrary but consistent "filing system" for memorizing these synthetic cities during fine-tuning.

#### Data Pipeline Understanding
**Pretraining (m1_10M)**:
- 10M examples from 12 tasks, all excluding Atlantis
- Model learns geographic relationships on real cities only

**Fine-tuning (m1_10M_ft5)**:
- 240K examples: 40K without Atlantis, 200K with Atlantis required
- This is where model first encounters Atlantis cities
- Model must memorize 100 fake cities with no semantic meaning

### Technical Insights

1. **INLP (Iterative Nullspace Projection)** cannot remove city ID information even after 100 iterations, showing how deeply encoded these patterns are

2. **Compass dataset generation bias**: "atlantis_required" datasets show systematic bias with Atlantis appearing in first position 95% of the time in compass task

3. **Mixed projection visualization** allows viewing representations in geographic coordinate space rather than pure PCA space

### Files Modified
- `/src/utils.py` - Enhanced filter_dataframe_by_pattern
- `/src/analysis/visualize_pca_3d_timeline.py` - Added mixed projection mode
- Multiple configs in `/configs/analysis_representation_higher/` - Updated to use new filtering syntax

### Files Created
- `/scratch/investigate_atlantis_clusters.py`
- `/scratch/test_atlantis_hypothesis.py`
- `/scratch/analyze_atlantis_in_training.py`
- `/scratch/check_atlantis_position.py`
- `/scratch/check_position_by_dataset.py`
- `/scratch/inlp_id_removal.py`
- `/scratch/remove_id_directions.py`
- `/scratch/remove_id_directions_extended.py`

### Next Steps
- Further investigation into why models create these arbitrary but consistent internal organizations
- Consider analyzing intermediate checkpoints to see when clustering emerges
- Explore if similar patterns exist for other synthetic data