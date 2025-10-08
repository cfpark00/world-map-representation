#!/usr/bin/env python3
import os
from pathlib import Path

# Define the 7 tasks
tasks = [
    "distance",
    "trianglearea",
    "angle",
    "compass",
    "inside",
    "perimeter",
    "crossing"
]

# Base directory for configs
base_dir = Path("/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/configs/data_generation/pftset")

# Template for config files
config_template = """output_dir: data/datasets/pft{first_idx}-{second_idx}

# Combination mode: 'sample' to take specific amounts from each dataset
mode: sample

# Random seed for reproducibility
seed: 42

# Whether to shuffle the final combined dataset
shuffle: true

# List of datasets to combine
datasets:
  - path: data/datasets/{first_task}_1M_no_atlantis
    n_samples: 20000
  - path: data/datasets/{second_task}_1M_no_atlantis
    n_samples: 100000
"""

# Generate only upper triangle (no diagonals, no symmetrics)
count = 0
for i, first_task in enumerate(tasks, 1):
    for j, second_task in enumerate(tasks, 1):
        # Skip diagonal (same task) and lower triangle (symmetric pairs)
        if i >= j:
            continue

        # Create filename
        filename = f"combine_pft{i}-{j}.yaml"
        filepath = base_dir / filename

        # Generate config content
        config_content = config_template.format(
            first_idx=i,
            second_idx=j,
            first_task=first_task,
            second_task=second_task
        )

        # Write config file
        with open(filepath, 'w') as f:
            f.write(config_content)

        print(f"Created {filename}: {first_task} (20k) + {second_task} (100k)")
        count += 1

print(f"\nGenerated {count} config files in {base_dir} (upper triangle only, no diagonals)")