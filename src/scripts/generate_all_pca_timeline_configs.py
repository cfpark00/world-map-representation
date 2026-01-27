#!/usr/bin/env python3
"""
Generate PCA timeline configs for all seeds (1, 2, 3) with both mixed and raw types.
Also update existing configs to exclude Atlantis.
"""

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

SEEDS = [1, 2, 3]
TYPES = ['mixed', 'raw']

base_config_dir = Path("configs/revision/exp4/pca_timeline")

# First, update existing seed configs to exclude Atlantis
print("=" * 80)
print("STEP 1: Updating existing seed configs to exclude Atlantis")
print("=" * 80)

for seed_name in ['seed1', 'seed2', 'seed3']:
    seed_dir = base_config_dir / seed_name
    if seed_dir.exists():
        yaml_files = list(seed_dir.glob("*.yaml"))
        print(f"\nUpdating {len(yaml_files)} configs in {seed_name}/")

        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)

            new_filter = "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$"
            modified = False

            if 'probe_train' in config and config['probe_train'] != new_filter:
                config['probe_train'] = new_filter
                modified = True

            if 'probe_test' in config and config['probe_test'] != new_filter:
                config['probe_test'] = new_filter
                modified = True

            if modified:
                with open(yaml_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  Updated: {yaml_file.name}")

# Now generate raw configs for seeds 1, 2, 3
print("\n" + "=" * 80)
print("STEP 2: Generating raw PCA timeline configs for seeds 1, 2, 3")
print("=" * 80)

for seed in SEEDS:
    seed_name = f"seed{seed}"

    # Create raw directory
    raw_dir = base_config_dir / f"{seed_name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating configs for {seed_name}_raw...")

    for task_id, task_name in TASK_MAPPING.items():
        # Special case: seed2 for pt1-5 doesn't exist, skip it
        if seed == 2 and task_id == 5:
            print(f"  Skipping: pt1-5_seed2 (doesn't exist)")
            continue

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
            "output_dir": f"/data/experiments/revision/exp4/pt1-{task_id}_seed{seed}/analysis_higher/{task_name}_firstcity_last_and_trans_l5/pca_timeline_raw",
            "probe_test": "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$",
            "probe_train": "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$",
            "representations_base_path": f"data/experiments/revision/exp4/pt1-{task_id}_seed{seed}/analysis_higher/{task_name}_firstcity_last_and_trans_l5/representations",
            "token_index": -1,
            "train_frac": 0.6
        }

        config_file = raw_dir / f"pt1-{task_id}_seed{seed}_{task_name}_firstcity_last_and_trans_l5.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  Created: {config_file.name}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("- Updated existing seed1, seed2, seed3 mixed configs to exclude Atlantis")
print("- Generated raw configs for seed1_raw, seed2_raw, seed3_raw")
print("- Note: pt1-5_seed2 skipped (doesn't exist)")
print("Done!")
