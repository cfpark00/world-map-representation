# Development Log - 2025-09-24 02:44

## CKA Analysis and Kernel Matrix Computation

### Summary
Developed comprehensive CKA (Centered Kernel Alignment) analysis pipeline for comparing representations between different models (pt1-1 through pt1-7). Created GPU-accelerated scripts for kernel matrix computation and CKA calculation, along with visualization tools.

### Major Tasks Completed

#### 1. Kernel Matrix Computation Pipeline
- Created `src/analysis/compute_kernel_matrices.py` with GPU acceleration
  - Processes representations from `analyze_representations_higher.py` output
  - Orders cities by city_id for consistent comparison across models
  - Supports regex-based city filtering
  - GPU-accelerated matrix operations for speed
  - Saves both combined and individual kernel matrices

- Created config: `configs/analysis_kernel/ftset/pt1-1/distance_firstcity_last_and_trans_l5.yaml`
  - Configurable token/layer selection
  - Linear/RBF kernel options
  - Optional kernel centering
  - City filtering with regex patterns

#### 2. CKA Computation from Representations
- Created `src/analysis/compute_cka_from_representations.py`
  - Loads representations directly from checkpoint directories
  - **Asserts same city IDs** between models for valid comparison
  - GPU-accelerated CKA computation
  - Saves results as JSON and CSV

- Generated 21 YAML configs for all unique model pairs (pt1-1 through pt1-7)
  - Used script to auto-generate configs: `scripts/analysis/generate_cka_configs.py`
  - Created master script: `scripts/analysis/run_all_cka.sh`

#### 3. CKA Visualization
- Created matrix visualization in `scratch/cka_analysis/plot_cka_matrix.py`
  - 7x7 heatmap showing CKA values between all model pairs
  - Additional 6x6 version excluding pt1-7 (crossing task)
  - Used viridis colormap (appropriate for 0-1 range data)

### Key Findings from CKA Analysis

**Most similar pairs:**
- pt1-3 (angle) vs pt1-6 (perimeter): CKA = 0.9710
- pt1-2 (trianglearea) vs pt1-3 (angle): CKA = 0.8981
- pt1-2 (trianglearea) vs pt1-6 (perimeter): CKA = 0.8953

**Least similar pairs:**
- pt1-3 (angle) vs pt1-7 (crossing): CKA = 0.0040
- pt1-7 (crossing) shows extremely low CKA (~0.004-0.022) with all other tasks

**Notable pattern:** pt1-7 (crossing) learned fundamentally different representations from all other geometric tasks.

### City Visualization Updates
- Updated `src/visualization/visualize_cities.py`:
  - Made configurable xlim/ylim from YAML
  - Removed title, axis labels, legend for cleaner appearance
  - Increased tick label size to 20 with padding
  - Made regular city dots bigger (size 8)
  - Created bash script for regeneration: `scripts/visualization/regenerate_city_maps.sh`

- Final settings:
  - xlim: [-140, 180] (Americas to Asia)
  - ylim: [-60, 75] (excludes polar regions)
  - Red dots for regular cities, blue for Atlantis

### Technical Improvements

#### GPU Acceleration
- Kernel matrix computation: ~4.5s per checkpoint (4413×4413 matrices)
- Bottleneck identified: Kernel centering (H @ K @ H) with large matrices
- File sizes: ~9.2 GB for 41 checkpoints due to pickle overhead

#### File Organization
- Cleaned up `/scratch/` directory into subfolders:
  - `cka_analysis/` - CKA plotting scripts
  - `city_checks/` - City consistency checks
  - `benchmarks/` - Performance analysis
  - `data_issues/` - Distance task debugging
  - `config_generation/` - Config generation scripts

### City Dataset Analysis
- Examined `src/data_processing/create_city_dataset.py`
- **No projection used** - Raw lat/lon coordinates
- Coordinates scaled by 10 to avoid decimals:
  - x: -1800 to 1800 (longitude × 10)
  - y: -900 to 900 (latitude × 10)
- This is unprojected geographic coordinates (Plate Carrée/equirectangular)

### Files Created/Modified

**New Scripts:**
- `/src/analysis/compute_kernel_matrices.py`
- `/src/analysis/compute_cka_from_representations.py`
- `/scratch/cka_analysis/plot_cka_matrix.py`
- `/scripts/analysis/generate_cka_configs.py`
- `/scripts/analysis/run_all_cka.sh`
- `/scripts/visualization/regenerate_city_maps.sh`

**Config Directories:**
- `/configs/analysis_kernel/` - Kernel matrix configs
- `/configs/analysis_cka/` - CKA computation configs (21 model pairs)

**Modified:**
- `/src/visualization/visualize_cities.py` - Enhanced with configurable bounds
- Various visualization configs for city maps

### Next Steps
- Complete remaining CKA computations for model pairs
- Analyze CKA evolution across training time
- Investigate why crossing task representations are so different
- Consider dimensionality reduction analysis on the representations