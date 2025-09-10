# Atlantis Cross Finetune Representation Analysis

**Date:** September 2, 2025 00:57  
**Session Duration:** ~30 minutes  
**Main Task:** Running 4 analysis configurations on atlantis_cross_finetune experiment

## Summary
Ran comprehensive representation analysis on the `atlantis_cross_finetune` experiment using 4 different configurations to understand how the model's internal representations evolved during fine-tuning.

## Tasks Completed

### 1. Analysis Configuration Setup
- Identified the 4 analysis configurations from the `mixed_dist20k_cross100k_finetune` reference
- Corrected cities dataset path from `data/geonames-all-cities-with-a-population-1000.csv` to `outputs/datasets/cities_100k_plus_seed42.csv`
- Located required files:
  - Atlantis cities: `outputs/datasets/atlantis_XX0_100_seed42.csv`  
  - Region mapping: `configs/atlantis_region_mapping.json`

### 2. Four Analysis Runs Executed
All analyses used layers 3,4 with 5000 probe cities and 3000 training cities:

**Analysis 1 - Default:**
```bash
python src/analysis/analyze_representations.py --exp_dir outputs/experiments/atlantis_cross_finetune --cities_csv outputs/datasets/cities_100k_plus_seed42.csv --layers 3 4 --n_probe_cities 5000 --n_train_cities 3000
```
- Final R²: Lon: 0.809, Lat: 0.811, Distance Error: 2328 km
- Output: `dist_layers3_4_probe5000_train3000`

**Analysis 2 - Plus100 Eval (Atlantis in test set only):**
```bash
... --additional-cities outputs/datasets/atlantis_XX0_100_seed42.csv --additional-labels configs/atlantis_region_mapping.json
```
- Final R²: Lon: 0.738, Lat: 0.752, Distance Error: 2599 km
- Output: `dist_layers3_4_probe5000_train3000_plus100eval`

**Analysis 3 - Plus100 Concat (Atlantis in training pool):**
```bash
... --additional-cities outputs/datasets/atlantis_XX0_100_seed42.csv --additional-labels configs/atlantis_region_mapping.json --concat-additional
```
- Final R²: Lon: 0.780, Lat: 0.768, Distance Error: 2493 km  
- Output: `dist_layers3_4_probe5000_train3000_plus100concat`

**Analysis 4 - No Africa (Africa excluded from training):**
```bash
... --remove-label-from-train "Africa"
```
- Final R²: Lon: 0.713, Lat: 0.612, Distance Error: 2851 km
- Output: `dist_layers3_4_probe5000_train3000_noAfrica`

### 3. Key Observations
- All analyses processed 22 checkpoints successfully
- Generated dynamics plots and world map animations for each configuration
- Clear degradation in representation quality from initial checkpoint (step 0) to final (step 3920)
- Excluding Africa from training showed the worst final performance
- Including Atlantis cities in training pool (concat) performed better than test-only

## Outputs Generated
Each analysis created:
- CSV file with R² scores across training steps
- Dynamics plot showing loss, R² evolution, and distance errors
- World map animation showing prediction evolution
- Final world map snapshot

## Files Modified
- No file structure changes made
- All outputs written to existing analysis directory structure

## Technical Details
- Used UV virtual environment
- All commands run from project root `/n/home12/cfpark00/WM_1`
- Analysis script: `src/analysis/analyze_representations.py`
- Total execution time: ~12 minutes per analysis (4 x ~3 minutes each)

## Next Steps
Analysis complete. Results available in respective analysis subdirectories for further examination and comparison.