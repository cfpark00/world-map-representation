# Atlantis Analysis Documentation

## Overview
This document describes the standard analysis workflow for experiments involving Atlantis (synthetic geographic data), particularly for understanding how models represent fictional vs real geographic locations.

## Analysis Script: `src/analysis/analyze_representations.py`

### Purpose
Analyzes how internal transformer representations encode geographic information by:
1. Training linear probes to predict longitude/latitude from hidden states
2. Evaluating probe accuracy (R² scores) 
3. Visualizing predicted locations on world maps
4. Generating animations showing representation evolution during training

### Key Parameters

#### Required Arguments
- `--exp_dir`: Path to experiment directory containing model checkpoints
- `--cities_csv`: Path to base cities CSV file (typically cities_100k.csv)

#### Layer Analysis
- `--layers`: Which transformer layers to analyze (default: [3, 4])
  - Layer indices are 0-based (layer 4 = index 3)

#### Probe Configuration
- `--n_probe_cities`: Number of cities to use for probe evaluation (default: 5000)
- `--n_train_cities`: Number of cities to use for training the probe (default: 3000)

#### Task Configuration
- `--task-type`: Type of task (`distance` or `randomwalk`)
- `--prompt-format`: Format of prompts (`dist` for distance, `rw200` for random walk)

#### Atlantis-Specific Options
- `--additional-cities`: Path to Atlantis cities CSV file
- `--additional-labels`: Path to JSON mapping country codes to regions (e.g., {"XX0": "Atlantis"})
- `--concat-additional`: If set, includes Atlantis cities in probe training pool; otherwise only in evaluation
- `--remove-label-from-train`: Exclude a specific region from probe training (e.g., "Africa")

## Standard Atlantis Analysis Workflow

### Why These Four Analyses?
We run four specific analyses to understand different aspects of representation formation:

1. **Baseline**: How well does the model represent real cities without Atlantis?
2. **Atlantis Evaluation Only**: Can probes trained on real cities generalize to predict Atlantis locations?
3. **Atlantis in Training**: Does including Atlantis in probe training improve localization?
4. **Ablation Study**: How important is African geography for the overall representation?

### The Four Standard Analyses

#### 1. Basic Analysis (No Atlantis)
Tests baseline geographic representation quality on real cities only.

```bash
python src/analysis/analyze_representations.py \
  --exp_dir /n/home12/cfpark00/WM_1/outputs/experiments/mixed_dist20k_cross100k_finetune \
  --cities_csv /n/home12/cfpark00/WM_1/data/cities_100k.csv \
  --layers 3 4 \
  --n_probe_cities 5000 \
  --n_train_cities 3000 \
  --task-type distance \
  --prompt-format dist
```

Output directory: `analysis/dist_layers3_4_probe5000_train3000/`

#### 2. Atlantis Additional (Evaluation Only)
Tests if probes trained on real cities can localize Atlantis cities.
- Atlantis cities are NOT used in probe training
- They only appear in the evaluation set
- Tests generalization to unseen geographic regions

```bash
python src/analysis/analyze_representations.py \
  --exp_dir /n/home12/cfpark00/WM_1/outputs/experiments/mixed_dist20k_cross100k_finetune \
  --cities_csv /n/home12/cfpark00/WM_1/data/cities_100k.csv \
  --layers 3 4 \
  --n_probe_cities 5000 \
  --n_train_cities 3000 \
  --task-type distance \
  --prompt-format dist \
  --additional-cities /n/home12/cfpark00/WM_1/outputs/datasets/atlantis_XX0_100_seed42.csv \
  --additional-labels /n/home12/cfpark00/WM_1/configs/atlantis_region_mapping.json
```

Output directory: `analysis/dist_layers3_4_probe5000_train3000_plus100eval/`

#### 3. Atlantis Concatenated (In Probe Training)
Tests if including Atlantis in probe training improves their localization.
- Atlantis cities CAN appear in probe training set
- Tests if the model learned coherent representations for Atlantis

```bash
python src/analysis/analyze_representations.py \
  --exp_dir /n/home12/cfpark00/WM_1/outputs/experiments/mixed_dist20k_cross100k_finetune \
  --cities_csv /n/home12/cfpark00/WM_1/data/cities_100k.csv \
  --layers 3 4 \
  --n_probe_cities 5000 \
  --n_train_cities 3000 \
  --task-type distance \
  --prompt-format dist \
  --additional-cities /n/home12/cfpark00/WM_1/outputs/datasets/atlantis_XX0_100_seed42.csv \
  --additional-labels /n/home12/cfpark00/WM_1/configs/atlantis_region_mapping.json \
  --concat-additional
```

Output directory: `analysis/dist_layers3_4_probe5000_train3000_plus100concat/`

#### 4. No Africa Ablation
Tests importance of African cities for overall representation quality.
- Excludes all African cities from probe training
- Reveals if certain regions are critical for learning global geography

```bash
python src/analysis/analyze_representations.py \
  --exp_dir /n/home12/cfpark00/WM_1/outputs/experiments/mixed_dist20k_cross100k_finetune \
  --cities_csv /n/home12/cfpark00/WM_1/data/cities_100k.csv \
  --layers 3 4 \
  --n_probe_cities 5000 \
  --n_train_cities 3000 \
  --task-type distance \
  --prompt-format dist \
  --remove-label-from-train Africa
```

Output directory: `analysis/dist_layers3_4_probe5000_train3000_noAfrica/`

## Output Files
Each analysis creates a subdirectory under `{exp_dir}/analysis/` containing:
- `representation_dynamics.csv`: R² scores and error metrics for each checkpoint
- `dynamics_plot.png`: Training curves showing probe accuracy over time
- `final_world_map.png`: Final predicted positions on world map
- `world_map_evolution.gif`: Animation of representation evolution

## Interpretation Guide
- **High R² (>0.9)**: Model has strong geographic representations
- **Scattered Atlantis**: Model failed to learn coherent representations for fictional locations
- **Clustered Atlantis**: Model successfully integrated fictional geography
- **Africa ablation impact**: Large drop indicates Africa is crucial for global representation