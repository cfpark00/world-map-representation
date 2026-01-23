#!/usr/bin/env python3
"""Generate representation extraction and PCA timeline configs for exp3."""

import yaml
from pathlib import Path

# Task mappings
TASKS = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing'
}

LAYERS = [3, 4, 5, 6]

def create_repr_extraction_config(model_type, task_num=None, layer=5):
    """Create representation extraction config.

    Args:
        model_type: 'pt1_wide', 'pt1_narrow', 'pt1_wide_ftwb{1-7}', 'pt1_narrow_ftwb{1-7}'
        task_num: Task number (1-7), only needed for ftwb versions
        layer: Layer to extract (3-6)
    """
    # Determine experiment directory and task
    if 'ftwb' not in model_type:
        # Base models - no specific task, use 'distance' as default
        task_name = 'distance'
        prompt_format = 'distance_firstcity_last_and_trans'
        experiment_dir = f'data/experiments/revision/exp3/{model_type}'
    else:
        # Fine-tuned models
        if task_num is None:
            raise ValueError("task_num required for ftwb models")
        task_name = TASKS[task_num]
        prompt_format = f'{task_name}_firstcity_last_and_trans'
        experiment_dir = f'data/experiments/revision/exp3/{model_type}'

    output_dir = f'/n/home12/cfpark00/WM_1/{experiment_dir}/analysis_higher/{prompt_format}_l{layer}'

    config = {
        'cities_csv': 'data/datasets/cities/cities.csv',
        'device': 'cuda',
        'experiment_dir': experiment_dir,
        'layers': [layer],
        'method': {'name': 'linear'},
        'n_test_cities': 1250,
        'n_train_cities': 3250,
        'output_dir': output_dir,
        'perform_pca': True,
        'probe_test': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'probe_train': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'prompt_format': prompt_format,
        'save_repr_ckpts': [-2],
        'seed': 42
    }

    return config

def create_pca_timeline_config(model_type, task_num=None, layer=5, pca_type='mixed'):
    """Create PCA timeline config.

    Args:
        model_type: 'pt1_wide', 'pt1_narrow', 'pt1_wide_ftwb{1-7}', 'pt1_narrow_ftwb{1-7}'
        task_num: Task number (1-7), only needed for ftwb versions
        layer: Layer (3-6)
        pca_type: 'mixed', 'raw', or 'na' (no atlantis probe)
    """
    # Determine task and paths
    if 'ftwb' not in model_type:
        task_name = 'distance'
        prompt_format = 'distance_firstcity_last_and_trans'
        experiment_dir = f'data/experiments/revision/exp3/{model_type}'
    else:
        if task_num is None:
            raise ValueError("task_num required for ftwb models")
        task_name = TASKS[task_num]
        prompt_format = f'{task_name}_firstcity_last_and_trans'
        experiment_dir = f'data/experiments/revision/exp3/{model_type}'

    representations_base_path = f'{experiment_dir}/analysis_higher/{prompt_format}_l{layer}/representations'

    if pca_type == 'mixed':
        output_suffix = 'pca_timeline'
        probe_train_filter = 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'
        probe_test_filter = 'region:.* && city_id:^[1-9][0-9]{3,}$'
        axis_type = 'mixed'
    elif pca_type == 'raw':
        output_suffix = 'pca_timeline_raw'
        probe_train_filter = 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'
        probe_test_filter = 'region:.* && city_id:^[1-9][0-9]{3,}$'
        axis_type = 'raw'
    elif pca_type == 'na':
        # No Atlantis - train probe without Atlantis, test also without Atlantis
        output_suffix = 'pca_timeline_na'
        probe_train_filter = 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'
        probe_test_filter = 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'
        axis_type = 'mixed'

    output_dir = f'/n/home12/cfpark00/WM_1/{experiment_dir}/analysis_higher/{prompt_format}_l{layer}/{output_suffix}'

    config = {
        'axis_mapping': {
            'type': axis_type,
            '1' if pca_type == 'raw' else 1: 'x',
            '2' if pca_type == 'raw' else 2: 'y',
            '3' if pca_type == 'raw' else 3: 'r0'
        },
        'cities_csv': 'data/datasets/cities/cities.csv',
        'layer_index': -1,
        'marker_size': 3,
        'n_components': 3,
        'output_dir': output_dir,
        'probe_test': probe_test_filter,
        'probe_train': probe_train_filter,
        'representations_base_path': representations_base_path,
        'token_index': -1,
        'train_frac': 0.6
    }

    return config

def main():
    base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')

    # Create directory structure
    repr_config_dir = base_dir / 'configs/revision/exp3/representation_extraction'
    pca_config_dir = base_dir / 'configs/revision/exp3/pca_timeline'

    # 1. Create representation extraction configs
    print("Creating representation extraction configs...")

    # For pt1_wide and pt1_narrow (base models) - only layer 5, distance task
    for model_type in ['pt1_wide', 'pt1_narrow']:
        model_dir = repr_config_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        for layer in [5]:  # Only layer 5 for base models
            config = create_repr_extraction_config(model_type, layer=layer)
            config_path = model_dir / f'distance_firstcity_last_and_trans_l{layer}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"  Created: {config_path}")

    # For ftwb versions - layer 5, each task
    for width_type in ['wide', 'narrow']:
        for task_num in range(1, 8):
            model_type = f'pt1_{width_type}_ftwb{task_num}'
            task_name = TASKS[task_num]

            model_dir = repr_config_dir / model_type
            model_dir.mkdir(parents=True, exist_ok=True)

            for layer in [5]:  # Only layer 5
                config = create_repr_extraction_config(model_type, task_num, layer)
                config_path = model_dir / f'{task_name}_firstcity_last_and_trans_l{layer}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  Created: {config_path}")

    # 2. Create PCA timeline configs
    print("\nCreating PCA timeline configs...")

    # For base models: mixed and raw
    for model_type in ['pt1_wide', 'pt1_narrow']:
        for pca_type in ['mixed', 'raw']:
            type_dir = pca_config_dir / f'{model_type}_{pca_type}'
            type_dir.mkdir(parents=True, exist_ok=True)

            config = create_pca_timeline_config(model_type, layer=5, pca_type=pca_type)
            config_path = type_dir / f'{model_type}_distance_firstcity_last_and_trans_l5.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"  Created: {config_path}")

    # For ftwb versions: mixed, raw, and na
    for width_type in ['wide', 'narrow']:
        for task_num in range(1, 8):
            model_type = f'pt1_{width_type}_ftwb{task_num}'
            task_name = TASKS[task_num]

            for pca_type in ['mixed', 'raw', 'na']:
                type_dir = pca_config_dir / f'{model_type}_{pca_type}'
                type_dir.mkdir(parents=True, exist_ok=True)

                config = create_pca_timeline_config(model_type, task_num, layer=5, pca_type=pca_type)
                config_path = type_dir / f'{model_type}_{task_name}_firstcity_last_and_trans_l5.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  Created: {config_path}")

    print(f"\nTotal configs created:")
    print(f"  Representation extraction: {2 + 14} = 16 configs")
    print(f"  PCA timeline: {4 + 42} = 46 configs")

if __name__ == '__main__':
    main()
