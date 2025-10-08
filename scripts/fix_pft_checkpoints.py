#!/usr/bin/env python3
"""Fix checkpoint paths in pft training configs."""

import re
from pathlib import Path

def fix_checkpoint_path(config_path):
    """Fix the checkpoint path in a pft config file."""
    # Extract X from pftX-Y.yaml
    match = re.match(r'pft(\d)-(\d)\.yaml', config_path.name)
    if not match:
        return False

    x = match.group(1)

    # Read the config
    content = config_path.read_text()

    # Replace the checkpoint path
    old_pattern = r'ckpt: data/experiments/pt1-1/checkpoints/final'
    new_path = f'ckpt: data/experiments/pt1-{x}/checkpoints/final'

    new_content = re.sub(old_pattern, new_path, content)

    if new_content != content:
        config_path.write_text(new_content)
        return True
    return False

def main():
    config_dir = Path("/n/home12/cfpark00/WM_1/configs/training/pftset")

    # Process all pftX-Y.yaml files
    for config_path in sorted(config_dir.glob("pft*.yaml")):
        if fix_checkpoint_path(config_path):
            print(f"Fixed checkpoint path in {config_path.name}")
        else:
            print(f"No changes needed for {config_path.name}")

if __name__ == "__main__":
    main()