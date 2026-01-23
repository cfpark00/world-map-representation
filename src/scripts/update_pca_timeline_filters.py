#!/usr/bin/env python3
"""Update PCA timeline configs to exclude Atlantis from probe_train and probe_test."""

import yaml
from pathlib import Path

# Find all original PCA timeline configs
config_dir = Path("configs/revision/exp4/pca_timeline/original")
yaml_files = list(config_dir.glob("*.yaml"))

print(f"Found {len(yaml_files)} config files")

for yaml_file in yaml_files:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update filters to exclude Atlantis
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
        print(f"Updated: {yaml_file.name}")
    else:
        print(f"Already correct: {yaml_file.name}")

print("Done!")
