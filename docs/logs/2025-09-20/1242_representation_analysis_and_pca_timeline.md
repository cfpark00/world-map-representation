# Development Log - September 20, 2025, 12:42

## Session Overview
Enhanced representation analysis scripts with PCA analysis capabilities and created interactive timeline visualizations for tracking representation evolution during training. Discovered and documented a fascinating phenomenon where the model learns city ID digit patterns before geographic semantics.

## Major Accomplishments

### 1. Enhanced analyze_representations_higher.py
- Created copy of original analyze_representations.py stripped of world map/GIF generation
- Added PCA analysis functionality that tests linear regression on [2,4,6,8,10] principal components
- Implemented configurable behavior via YAML (perform_pca flag, defaults to True)
- Fixed NameError bug for perform_pca_analysis_flag variable
- Generates plots showing R² vs number of PCA components used
- Saves results to checkpoint-XX/pca/ directory structure

### 2. Implemented Representation Saving Features
- Added support for save_repr_ckpts: [-2] to save representations for ALL checkpoints
- Previous options: [-1] for last checkpoint, specific step numbers, or [] for none
- Updated print statements to clarify what -2 means ("Will save representations for ALL checkpoints")

### 3. Created Interactive PCA Timeline Visualization
**New Script**: `/src/analysis/visualize_pca_3d_timeline.py`
- Loads representations from all checkpoint directories automatically
- Creates interactive 3D scatter plot with time slider for navigating checkpoints
- Features play/pause buttons for animation
- Cities colored by geographic region with hover information
- **Important Update**: Modified to use fixed PCA coordinate system from final checkpoint
  - Provides consistent space to observe representation evolution
  - Shows how representations move from random to organized structure

**Configuration**: `/configs/analysis_pca_timeline/m1_10M_width=256_depth=10/world_distance/l8_all_checkpoints.yaml`
- Configurable marker_size parameter (default 4, set to 3 for smaller dots)
- Support for token/layer selection and axis remapping

### 4. Discovered City ID Digit Clustering Phenomenon

#### The Discovery
At step 17,763 (7.5% through training), found the model creates two distinct clusters based purely on city ID patterns:
- Small cluster (531 cities): All cities with city_id < 1000
- Large cluster (4,469 cities): All cities with city_id ≥ 1000
- Perfect 100% separation with Cohen's d = 6.48 (massive effect size)

#### Sub-clustering Revealed Hierarchy
- 3 cities: IDs 0-9 (single digit)
- 54 cities: IDs 10-99 (two digits)
- 474 cities: IDs 100-999 (three digits)
- 4,469 cities: IDs 1000+ (four digits)

#### Why This Happens
- City IDs appear in prompts (e.g., "dist(c_665,c_")
- Neural networks learn simple numerical patterns before complex semantics
- Digit count is easier to detect than geographic relationships

#### Key Finding
This ID-based clustering **persists as an independent PCA component in the final checkpoint**, suggesting models maintain parallel representation schemes throughout training.

### 5. Documentation and Visualization

#### Created Analysis Scripts in /scratch/
- `detect_clusters_step17763.py` - Initial cluster detection with multiple algorithms
- `analyze_small_cluster.py` - Deep dive into the smaller cluster
- `check_id_pattern.py` - Verification that pattern is city_id < 1000
- `plot_id_pattern_clusters.py` - Beautiful visualizations with color coding by digit count

#### Generated Compelling Visualizations
- Used pink gradient for IDs < 1000, blue for IDs ≥ 1000
- Larger markers for smaller ID numbers for visibility
- Both 2D and 3D views showing perfect separation

#### Documented Finding
Created `/docs/findings/city_id_digit_clustering.md` with:
- Complete technical description
- Statistical evidence (Cohen's d = 6.48)
- Explanation of learning dynamics
- Implications for understanding neural networks
- Evidence of persistence to final checkpoint

## File Structure Changes

### New Directories Created
- `/configs/analysis_representation_higher/` - Configs for stripped-down analysis script
- `/configs/analysis_pca_timeline/` - Configs for timeline visualization
- `/docs/findings/` - Directory for documenting discoveries

### New Scripts Added
- `/src/analysis/analyze_representations_higher.py` - Analysis without world map visualization
- `/src/analysis/visualize_pca_3d_timeline.py` - Interactive timeline visualization
- `/scripts/analysis_representations/pca_timeline_m1_10M_width=256_depth=10_l8.sh` - Runner script

### Scratch Work
Multiple analysis scripts in `/scratch/` for investigating the clustering phenomenon

## Technical Insights

1. **Models Learn in Stages**: Simple numerical patterns → complex semantic understanding
2. **Feature Persistence**: Early-learned features remain as latent dimensions
3. **PCA on Final Checkpoint**: Using fixed PCA from final checkpoint for all projections provides consistent coordinate system to observe learning dynamics
4. **Representation Richness**: Final models contain both task-relevant (geography) and task-irrelevant (ID numerics) features

## Next Steps Potential
- Investigate if ID pattern clustering occurs in other models/tasks
- Test if randomizing IDs prevents this artifact
- Explore whether this numerical scaffold helps or hinders learning
- Analyze more checkpoints to pinpoint exact transition from ID-based to geographic clustering

## Session Summary
Today's work significantly enhanced our ability to analyze and visualize how representations evolve during training. The discovery of ID-based clustering in early training provides valuable insights into how neural networks naturally decompose complex tasks, starting with the simplest available patterns. The tools created today will be valuable for future representation analysis work.