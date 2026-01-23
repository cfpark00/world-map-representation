# Exp5 Infrastructure Completion & Probe Generalization Analysis - 2025-11-21 18:23

## Summary

Completed the Exp5 infrastructure for representation extraction, PCA visualization, and probe generalization analysis. Also updated the probe generalization histogram scripts to remove p-values and titles for cleaner presentation.

## Work Done

### 1. Exp5 Representation Extraction Infrastructure

Created configs and scripts for extracting representations at layers 3, 4, 5, 6:

**Configs (4):**
- `configs/revision/exp5/representation_extraction/distance_firstcity_last_and_trans_l{3,4,5,6}.yaml`

**Scripts (5):**
- `scripts/revision/exp5/representation_extraction/extract_l{3,4,5,6}.sh`
- `scripts/revision/exp5/representation_extraction/extract_all.sh`

### 2. Exp5 PCA Timeline Infrastructure

Created 3 PCA visualization configs (matching exp1 format):

**Configs (3):**
- `configs/revision/exp5/pca_timeline/distance_firstcity_last_and_trans_l5.yaml` (mixed/default)
- `configs/revision/exp5/pca_timeline/distance_firstcity_last_and_trans_l5_na.yaml` (no atlantis probe)
- `configs/revision/exp5/pca_timeline/distance_firstcity_last_and_trans_l5_raw.yaml` (raw PCA)

**Scripts (1):**
- `scripts/revision/exp5/pca_timeline/pca_all.sh`

### 3. Exp5 Probe Generalization Infrastructure

Created config and script for evaluating probe generalization on Atlantis cities:

**Config:**
- `configs/revision/exp5/probe_generalization/atlantis.yaml`

**Script:**
- `scripts/revision/exp5/probe_generalization/run_atlantis.sh`

### 4. Probe Generalization Histogram with Exp5

Created modified histogram script that includes exp5 as a green reference line:

**Script:**
- `src/scripts/plot_probe_generalization_histogram_with_exp5.py`
- `scripts/revision/exp5/plots/plot_probe_generalization_histogram_with_exp5.sh`

**Output:**
- `data/experiments/revision/exp1/plots/probe_generalization_histogram_with_exp5.png`

### 5. Key Results

Ran probe generalization analysis for exp5 and generated histogram:

| Model | Mean Distance Error | N samples |
|-------|---------------------|-----------|
| **Exp5 (PT1 with Atlantis)** | **23.2** | 87 |
| Baseline (non-Atlantis) | 18.6 ± 16.2 | 8400 |
| Atlantis without distance task | 107.8 ± 104.9 | 5220 |
| Atlantis with distance task | 508.3 ± 288.3 | 2088 |

**Key Finding**: Training with Atlantis from scratch (exp5) achieves Atlantis error (23.2) very close to baseline non-Atlantis cities (18.6), confirming the hypothesis that OOD generalization issues stem from training distribution, not model limitations.

### 6. Plot Styling Updates

Removed p-values and titles from both histogram scripts for cleaner presentation:
- `src/scripts/plot_probe_generalization_histogram.py`
- `src/scripts/plot_probe_generalization_histogram_with_exp5.py`

## Files Created/Modified

### Created
- `configs/revision/exp5/representation_extraction/` (4 configs)
- `scripts/revision/exp5/representation_extraction/` (5 scripts)
- `configs/revision/exp5/pca_timeline/` (3 configs)
- `scripts/revision/exp5/pca_timeline/pca_all.sh`
- `configs/revision/exp5/probe_generalization/atlantis.yaml`
- `scripts/revision/exp5/probe_generalization/run_atlantis.sh`
- `src/scripts/plot_probe_generalization_histogram_with_exp5.py`
- `scripts/revision/exp5/plots/plot_probe_generalization_histogram_with_exp5.sh`

### Modified
- `src/scripts/plot_probe_generalization_histogram.py` (removed p-values/title)

## Exp5 Workflow Summary

After training completes:
```bash
# 1. Extract representations
bash scripts/revision/exp5/representation_extraction/extract_all.sh

# 2. Run probe generalization
bash scripts/revision/exp5/probe_generalization/run_atlantis.sh

# 3. Generate PCA visualizations
bash scripts/revision/exp5/pca_timeline/pca_all.sh

# 4. Generate histogram with exp5 line
bash scripts/revision/exp5/plots/plot_probe_generalization_histogram_with_exp5.sh
```
