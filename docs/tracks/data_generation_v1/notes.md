# Data Generation v1 Track

## Overview
This track contains all data generation and tokenizer creation code for the world representation experiments.

## Data Output Structure
```
data/data_generation_v1/
├── cities/                      # City dataset with coordinates
│   └── cities.csv
├── tokenizers/                  # Character-level tokenizers
│   └── default_tokenizer/
├── single_datasets/             # Individual task datasets
│   ├── distance_1M_no_atlantis/
│   ├── distance_1M_with_atlantis/
│   ├── distance_100k_atlantis_required/
│   ├── trianglearea_1M_no_atlantis/
│   ├── angle_1M_no_atlantis/
│   └── ...
└── derived_datasets/            # Combined/mixed datasets
    ├── multitask_pt1/           # 7-task pretraining (~7M samples)
    ├── multitask_pt1_with_atlantis/
    ├── ft1-1/, ft1-2/, ...      # Single-task fine-tuning
    ├── ft2-1/, ft2-2/, ...      # Two-task fine-tuning
    ├── ftwb1-1/, ftwb1-2/, ...  # Fine-tuning with warmup+baseline
    ├── ftwb2-1/, ftwb2-2/, ...
    ├── pt2-1/, pt2-2/, ...      # Two-task pretraining
    └── pt3-1/, pt3-2/, ...      # Three-task pretraining
```

## Code Structure
```
src/data_generation_v1/
├── tasks/              # Task implementations (angle.py, distance.py, etc.)
├── scripts/            # Python orchestration scripts
├── create_city_dataset.py
├── combine_datasets.py
├── append_cities_to_dataset.py
├── create_tokenizer.py
└── utils.py            # Track-specific utilities

configs/data_generation_v1/
├── ftset/              # Fine-tuning dataset combine configs
├── pftset/             # Partial fine-tuning dataset configs
├── tokenizers/         # Tokenizer configs
└── *.yaml              # Individual task dataset configs

scripts/data_generation_v1/
├── cities/             # City dataset scripts
├── merge/              # Dataset combining scripts
├── multi/              # Multi-task dataset scripts
├── single_tasks/       # Single-task dataset scripts
└── tokenizers/         # Tokenizer creation scripts
```

## Tasks (7 total)
1. **distance** - geodesic distance between city pairs
2. **trianglearea** - area of triangle from 3 cities
3. **angle** - angle at vertex of 3 cities
4. **compass** - bearing/direction between cities
5. **inside** - point in triangle test
6. **perimeter** - triangle perimeter calculation
7. **crossing** - line segment intersection

## Dataset Types
- **1M samples**: Main training datasets (with/without Atlantis)
- **100k samples**: Atlantis-required subsets for fine-tuning warmup
- **Combined datasets**: PT1, PT2, PT3, FT1, FT2, FT3, FTWB1, FTWB2, FTWB3

## Key Concepts
- **Atlantis**: Fictitious clustered city location at (-35, 35) used for OOD testing
- **City encoding**: `c_{id}:(lat,lon)` format
- **Train/Test split**: 3250/1250 cities
- **Padding**: City IDs padded to 4 digits with leading zeros

## Downstream Dependencies
This track's outputs are consumed by:
- `pretraining_v1` - uses derived_datasets for model training
- `finetuning_v1` - uses derived_datasets for fine-tuning
