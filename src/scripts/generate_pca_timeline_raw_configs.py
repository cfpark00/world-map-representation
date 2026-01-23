#!/usr/bin/env python3
"""Generate PCA timeline configs with raw (no probe alignment) for original experiments."""

import yaml
from pathlib import Path

# Task mapping
TASK_MAPPING = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing'
}

# Create output directory
output_dir = Path("configs/revision/exp4/pca_timeline/original_raw")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating raw PCA timeline configs in {output_dir}...")

for task_id, task_name in TASK_MAPPING.items():
    config = {
        "axis_mapping": {
            "type": "raw",
            "1": "x",
            "2": "y",
            "3": "r0"
        },
        "cities_csv": "data/datasets/cities/cities.csv",
        "layer_index": -1,
        "marker_size": 3,
        "n_components": 3,
        "output_dir": f"/n/home12/cfpark00/WM_1/data/experiments/pt1-{task_id}/analysis_higher/{task_name}_firstcity_last_and_trans_l5/pca_timeline_raw",
        "probe_test": "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$",
        "probe_train": "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$",
        "representations_base_path": f"data/experiments/pt1-{task_id}/analysis_higher/{task_name}_firstcity_last_and_trans_l5/representations",
        "token_index": -1,
        "train_frac": 0.6
    }

    config_file = output_dir / f"pt1-{task_id}_{task_name}_firstcity_last_and_trans_l5.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {config_file.name}")

print(f"\nCreated {len(TASK_MAPPING)} raw PCA timeline configs")
