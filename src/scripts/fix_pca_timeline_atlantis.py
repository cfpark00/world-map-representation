#!/usr/bin/env python3
"""
Fix PCA timeline configs:
- probe_train: exclude Atlantis (doesn't affect probe training)
- probe_test: include Atlantis (shows up as dots, preserves color mapping)
"""

import yaml
from pathlib import Path

# Find all PCA timeline config directories
base_dir = Path("configs/revision/exp4/pca_timeline")
config_dirs = [
    "original",
    "original_raw",
    "seed1",
    "seed1_raw",
    "seed2",
    "seed2_raw",
    "seed3",
    "seed3_raw"
]

# Filters
probe_train_filter = "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$"  # Exclude Atlantis
probe_test_filter = "region:.* && city_id:^[1-9][0-9]{3,}$"  # Include Atlantis

total_updated = 0

for config_dir_name in config_dirs:
    config_dir = base_dir / config_dir_name

    if not config_dir.exists():
        print(f"Skipping {config_dir_name} (doesn't exist)")
        continue

    yaml_files = list(config_dir.glob("*.yaml"))
    print(f"\nUpdating {len(yaml_files)} configs in {config_dir_name}/")

    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        modified = False

        # Update probe_train to exclude Atlantis
        if 'probe_train' in config and config['probe_train'] != probe_train_filter:
            config['probe_train'] = probe_train_filter
            modified = True

        # Update probe_test to include Atlantis
        if 'probe_test' in config and config['probe_test'] != probe_test_filter:
            config['probe_test'] = probe_test_filter
            modified = True

        if modified:
            with open(yaml_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"  Updated: {yaml_file.name}")
            total_updated += 1

print(f"\n{'='*80}")
print(f"Total updated: {total_updated} configs")
print(f"probe_train: Excludes Atlantis (doesn't affect probe training)")
print(f"probe_test: Includes Atlantis (shows as dots, preserves colors)")
print("Done!")
