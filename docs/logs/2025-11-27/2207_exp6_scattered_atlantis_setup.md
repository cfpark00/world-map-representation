# Exp6: Scattered Atlantis Infrastructure Setup

## Summary
Set up complete data generation and training infrastructure for Exp6, which tests whether the observed effects (e.g., distance task impairing OOD generalization) are due to Atlantis cities being clustered in one location vs being scattered uniformly across the world.

## Motivation
A reviewer suspects that some observed effects might be artifacts of Atlantis being clustered at one geographic location (-35, 35). Exp6 addresses this by scattering 100 Atlantis cities uniformly across the entire world (x: [-180, 180], y: [-90, 90]).

## Work Completed

### 1. Modified Core Code
- **`src/data_processing/create_city_dataset.py`**: Added support for `scattered_atlantis` config key
  - Original `atlantis_regions` uses Gaussian distribution (clustered)
  - New `scattered_atlantis` uses uniform random distribution (scattered)

### 2. Data Generation Infrastructure

**Config Files Created (50 total):**
- `configs/revision/exp6/data_generation/city_dataset_scattered_atlantis.yaml` - City dataset config
- 7 × `{task}_1M_with_atlantis.yaml` - All pairs including scattered Atlantis
- 7 × `{task}_1M_no_atlantis.yaml` - World cities only
- 7 × `{task}_100k_atlantis_required.yaml` - Pairs must include Atlantis
- `ftset/combine_multitask_pt1.yaml` - 7M samples for PT1 pretraining
- 7 × `ftset/combine_ftwb1-{1-7}.yaml` - Single-task fine-tuning
- 21 × `ftset/combine_ftwb2-{1-21}.yaml` - Two-task fine-tuning

**Scripts Created:**
- `scripts/revision/exp6/data_generation/step1_gen_cities.sh` - Generate city dataset
- `scripts/revision/exp6/data_generation/step2-1_gen_tasks_dist_tri_ang.sh` - distance, trianglearea, angle
- `scripts/revision/exp6/data_generation/step2-2_gen_tasks_comp_cross.sh` - compass, crossing
- `scripts/revision/exp6/data_generation/step2-3_gen_tasks_inside_peri.sh` - inside, perimeter
- `scripts/revision/exp6/data_generation/step3_combine_all.sh` - Combine all datasets

**Visualization:**
- `configs/revision/exp6/cities_scattered_atlantis_plot.yaml`
- `scripts/revision/exp6/plot_cities_scattered_atlantis.sh`

### 3. Training Infrastructure

**PT1 (7-task pretraining):**
- `configs/revision/exp6/training/train_pt1.yaml`
- `scripts/revision/exp6/training/train_pt1.sh`

**FTWB1 (7 single-task fine-tuning):**
- `configs/revision/exp6/training/ftwb1/pt1_ftwb1-{1-7}.yaml`
- `scripts/revision/exp6/training/ftwb1/train_ftwb1-{1-7}.sh`
- `scripts/revision/exp6/training/train_all_ftwb1.sh` (batch)

**FTWB2 (21 two-task fine-tuning):**
- `configs/revision/exp6/training/ftwb2/pt1_ftwb2-{1-21}.yaml`
- `scripts/revision/exp6/training/ftwb2/train_ftwb2-{1-21}.sh`
- `scripts/revision/exp6/training/train_ftwb2_part1.sh` (1-7)
- `scripts/revision/exp6/training/train_ftwb2_part2.sh` (8-14)
- `scripts/revision/exp6/training/train_ftwb2_part3.sh` (15-21)

### 4. Generator Scripts
- `src/scripts/generate_exp6_data_configs.py` - Generate all data configs
- `src/scripts/generate_exp6_data_scripts.py` - Generate all data scripts
- `src/scripts/generate_exp6_ftwb_training_configs.py` - Generate FTWB training configs

## Output Locations
- Datasets: `data/experiments/revision/exp6/datasets/`
- PT1 model: `data/experiments/revision/exp6/pt1/`
- FTWB1 models: `data/experiments/revision/exp6/pt1_ftwb1-{1-7}/`
- FTWB2 models: `data/experiments/revision/exp6/pt1_ftwb2-{1-21}/`
- Plots: `data/experiments/revision/exp6/plots/`

## Execution Order
1. `bash scripts/revision/exp6/data_generation/step1_gen_cities.sh`
2. Run step2-1, step2-2, step2-3 (can be parallel)
3. `bash scripts/revision/exp6/data_generation/step3_combine_all.sh`
4. `bash scripts/revision/exp6/training/train_pt1.sh`
5. `bash scripts/revision/exp6/training/train_all_ftwb1.sh`
6. Run train_ftwb2_part1, part2, part3 (can be parallel after PT1)

## Current State
- Step 1 complete (cities generated, visualization confirmed scattered distribution)
- Steps 2-1, 2-2, 2-3 running
- Training infrastructure ready, awaiting data generation completion

## Files Modified
- `src/data_processing/create_city_dataset.py` (added uniform distribution support)

## New Directories Created
- `configs/revision/exp6/`
- `scripts/revision/exp6/`
- `data/experiments/revision/exp6/`
