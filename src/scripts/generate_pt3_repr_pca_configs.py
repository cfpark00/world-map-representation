#!/usr/bin/env python3
"""Generate representation extraction and PCA timeline configs for PT3."""

import yaml
from pathlib import Path

# PT3 task combinations - use first task from each triple
PT3_TASKS = {
    1: 'distance',      # distance+trianglearea+angle
    2: 'compass',       # compass+inside+perimeter
    3: 'crossing',      # crossing+distance+trianglearea
    4: 'angle',         # angle+compass+inside
    5: 'perimeter',     # perimeter+crossing+distance
    6: 'trianglearea',  # trianglearea+angle+compass
    7: 'inside',        # inside+perimeter+crossing
    # pt3-8 not trained yet
}

SEEDS = [1, 2]
LAYER = 5

def create_repr_extraction_config(variant_num, seed):
    """Create representation extraction config for PT3."""
    task_name = PT3_TASKS[variant_num]
    prompt_format = f'{task_name}_firstcity_last_and_trans'

    experiment_dir = f'data/experiments/revision/exp2/pt3-{variant_num}_seed{seed}'
    output_dir = f'/n/home12/cfpark00/WM_1/{experiment_dir}/analysis_higher/{prompt_format}_l{LAYER}'

    config = {
        'cities_csv': 'data/datasets/cities/cities.csv',
        'device': 'cuda',
        'layers': [LAYER],
        'method': {'name': 'linear'},
        'n_test_cities': 1250,
        'n_train_cities': 3250,
        'perform_pca': True,
        'probe_test': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'probe_train': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'save_repr_ckpts': [-2],
        'seed': 42,
        'experiment_dir': experiment_dir,
        'output_dir': output_dir,
        'prompt_format': prompt_format
    }

    return config

def create_pca_timeline_config(variant_num, seed, pca_type='mixed'):
    """Create PCA timeline config for PT3."""
    task_name = PT3_TASKS[variant_num]
    prompt_format = f'{task_name}_firstcity_last_and_trans'

    experiment_dir = f'data/experiments/revision/exp2/pt3-{variant_num}_seed{seed}'
    representations_base_path = f'{experiment_dir}/analysis_higher/{prompt_format}_l{LAYER}/representations'

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
        output_suffix = 'pca_timeline_na'
        probe_train_filter = 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'
        probe_test_filter = 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'
        axis_type = 'mixed'

    output_dir = f'/n/home12/cfpark00/WM_1/{experiment_dir}/analysis_higher/{prompt_format}_l{LAYER}/{output_suffix}'

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
    repr_config_dir = base_dir / 'configs/revision/exp2/pt3_seed/extract_representations'
    pca_config_dir = base_dir / 'configs/revision/exp2/pt3_seed/pca_timeline'

    repr_config_dir.mkdir(parents=True, exist_ok=True)
    pca_config_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create representation extraction configs
    print("Creating representation extraction configs...")
    repr_count = 0

    for variant_num in PT3_TASKS.keys():
        for seed in SEEDS:
            task_name = PT3_TASKS[variant_num]
            config = create_repr_extraction_config(variant_num, seed)

            config_path = repr_config_dir / f'pt3-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{LAYER}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"  Created: {config_path}")
            repr_count += 1

    # 2. Create PCA timeline configs
    print("\nCreating PCA timeline configs...")
    pca_count = 0

    for pca_type in ['mixed', 'raw', 'na']:
        for variant_num in PT3_TASKS.keys():
            for seed in SEEDS:
                task_name = PT3_TASKS[variant_num]
                config = create_pca_timeline_config(variant_num, seed, pca_type)

                config_path = pca_config_dir / f'pt3-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{LAYER}_{pca_type}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  Created: {config_path}")
                pca_count += 1

    print(f"\nTotal configs created:")
    print(f"  Representation extraction: {repr_count} configs")
    print(f"  PCA timeline: {pca_count} configs")
    print(f"\nNote: PT3-8 excluded (not trained yet)")

if __name__ == '__main__':
    main()
