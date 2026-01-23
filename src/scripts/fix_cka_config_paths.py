#!/usr/bin/env python3
"""
Fix CKA config paths: original experiments should point to data/experiments/pt1-X/
not data/experiments/revision/exp4/pt1-X/
"""

import yaml
from pathlib import Path
import re

# Find all CKA configs
config_dir = Path("configs/revision/exp4/cka_cross_seed")
yaml_files = list(config_dir.glob("*/*.yaml"))

print(f"Found {len(yaml_files)} CKA config files")

fixed_count = 0
for yaml_file in yaml_files:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    modified = False

    # Fix exp1 path if it's an original experiment
    if 'exp1' in config and 'repr_dir' in config['exp1']:
        old_path = config['exp1']['repr_dir']
        # Pattern: data/experiments/revision/exp4/pt1-X/ where X is 1-7 (no seed suffix)
        new_path = re.sub(
            r'data/experiments/revision/exp4/(pt1-[1-7])/',
            r'data/experiments/\1/',
            old_path
        )
        if new_path != old_path:
            config['exp1']['repr_dir'] = new_path
            modified = True

    # Fix exp2 path if it's an original experiment
    if 'exp2' in config and 'repr_dir' in config['exp2']:
        old_path = config['exp2']['repr_dir']
        new_path = re.sub(
            r'data/experiments/revision/exp4/(pt1-[1-7])/',
            r'data/experiments/\1/',
            old_path
        )
        if new_path != old_path:
            config['exp2']['repr_dir'] = new_path
            modified = True

    if modified:
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        fixed_count += 1

print(f"Fixed {fixed_count} config files")
