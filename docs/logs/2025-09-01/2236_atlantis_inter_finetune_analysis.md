# Log: Atlantis Inter-Finetune Analysis
**Date:** 2025-09-01  
**Time:** 22:36  
**Session Focus:** Analyzing the atlantis_inter_finetune experiment to understand catastrophic forgetting

## Summary
Analyzed the atlantis_inter_finetune experiment which demonstrates catastrophic forgetting when a model pre-trained on real world geography is fine-tuned on synthetic Atlantis coordinates.

## Tasks Completed

### 1. Initial Experiment Exploration
- Examined `/n/home12/cfpark00/WM_1/outputs/experiments/atlantis_inter_finetune/`
- Reviewed configuration: 500 epochs of fine-tuning on the atlantis_inter_4k_42 dataset
- Model was initialized from a checkpoint pre-trained on real world cities (dist_100k_1M_20epochs)
- Training results: final loss 0.831, eval valid ratio 98.4%

### 2. Representation Analysis Attempts
- **First attempt:** Used Atlantis cities CSV (atlantis_XX0_100_seed42.csv) with 50 probe cities
  - Result: Negative R² values (-0.003 lon, -2.008 lat) indicating no geographic correlation
  - This was expected since Atlantis uses synthetic coordinates
  
- **Second attempt (corrected):** Used original cities_100k_plus_seed42.csv with 5000 probe cities
  - Initial (Step 0): Strong representations with Lon R²=0.956, Lat R²=0.923, distance error 993 km
  - Final (Step 4000): Degraded to Lon R²=0.521, Lat R²=0.540, distance error 4003 km
  - **Catastrophic forgetting confirmed:** ~43% loss in longitude accuracy, ~38% loss in latitude accuracy

### 3. Analysis Outputs Generated
Created analysis directory: `/n/home12/cfpark00/WM_1/outputs/experiments/atlantis_inter_finetune/analysis/`
- `dist_layers3_4_probe50_train30/` - Analysis with Atlantis cities
- `dist_layers3_4_probe5000_train3000/` - Analysis with real world cities showing forgetting
- Generated visualizations: dynamics plots, world maps, and evolution GIFs

## Key Findings
1. **Catastrophic Forgetting Demonstrated:** Fine-tuning on synthetic Atlantis coordinates severely degraded the model's ability to represent real world geography
2. **Quantified Impact:** Distance errors increased by over 3000 km after fine-tuning
3. **Representation Collapse:** R² values dropped from >0.9 to ~0.5, showing significant loss of geographic structure

## Technical Issues Encountered
- Initial confusion about which cities CSV to use for analysis
- Bug fix in analyze_representations.py regarding JSON import (user fixed during session)
- Corrected argument name from --country-labels to --additional-labels

## Files Modified
None - only ran analysis scripts and generated output files

## Next Steps
The analysis successfully demonstrates catastrophic forgetting when models trained on real geography are fine-tuned on fictional landmasses. This could be valuable for understanding how neural networks encode and forget spatial representations.