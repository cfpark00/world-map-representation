#!/usr/bin/env python3
"""
Generate all CKA config files for pt1-1 through pt1-7 comparisons.
"""

from pathlib import Path
import yaml

# Model mappings
models = {
    'pt1-1': 'distance',
    'pt1-2': 'trianglearea',
    'pt1-3': 'angle',
    'pt1-4': 'compass',
    'pt1-5': 'inside',
    'pt1-6': 'perimeter',
    'pt1-7': 'crossing'
}

# Base config template
base_config = {
    'token_index': -1,
    'layer_index': -1,
    'kernel_type': 'linear',
    'center_kernels': True,
    'use_gpu': True,
    'checkpoint_steps': None,
    'cities_csv': 'data/datasets/cities/cities.csv',
    'city_filter': 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'
}

# Output directory
config_dir = Path('/configs/analysis_cka')
config_dir.mkdir(exist_ok=True)

# Generate configs for all unique pairs (no self-comparisons)
configs_created = []

for model1, task1 in models.items():
    for model2, task2 in models.items():
        # Skip self-comparisons and duplicate pairs (only do i < j)
        if model1 >= model2:
            continue

        # Create config
        config = base_config.copy()

        # Set paths - using the task name with _firstcity_last_and_trans_l5 pattern
        config['representations_path_1'] = f'data/experiments/{model1}/analysis_higher/{task1}_firstcity_last_and_trans_l5/representations'
        config['representations_path_2'] = f'data/experiments/{model2}/analysis_higher/{task2}_firstcity_last_and_trans_l5/representations'

        # Set output directory
        config['output_dir'] = f'data/experiments/cka_analysis/{model1}_vs_{model2}_l5'

        # Add comment at the top
        config_with_comment = f'# CKA between {model1} ({task1}) and {model2} ({task2})\n\n' + yaml.dump(config, default_flow_style=False)

        # Save config
        config_filename = f'{model1}_vs_{model2}.yaml'
        config_path = config_dir / config_filename

        with open(config_path, 'w') as f:
            f.write(config_with_comment)

        configs_created.append(config_filename)
        print(f"Created: {config_filename}")

print(f"\nTotal configs created: {len(configs_created)}")

# Also create a summary file
summary_path = config_dir / 'README.md'
with open(summary_path, 'w') as f:
    f.write("# CKA Configuration Files\n\n")
    f.write("## Model-Task Mappings\n\n")
    for model, task in models.items():
        f.write(f"- {model}: {task}\n")
    f.write(f"\n## Configurations ({len(configs_created)} unique pairs)\n\n")
    for config_file in sorted(configs_created):
        f.write(f"- {config_file}\n")

print(f"Summary written to: {summary_path}")