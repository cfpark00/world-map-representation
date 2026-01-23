#!/usr/bin/env python3
"""
Generate all data generation configs for Exp6 (scattered Atlantis).

Exp6 tests whether the observed effects are due to Atlantis being clustered
in one location vs being scattered uniformly across the world.

Creates:
1. City dataset with scattered Atlantis (uniform random)
2. 7 task datasets with scattered Atlantis (1M_with_atlantis)
3. 7 task datasets without Atlantis (1M_no_atlantis) - uses same cities CSV
4. 7 task datasets with Atlantis required (100k_atlantis_required)
5. multitask_pt1 combined dataset
6. 7 FTWB1 combined datasets (single-task fine-tuning)
7. 21 FTWB2 combined datasets (two-task fine-tuning)
"""
import yaml
from pathlib import Path
from itertools import combinations

# Task definitions
TASKS = ['distance', 'trianglearea', 'angle', 'compass', 'inside', 'perimeter', 'crossing']

# Base paths
BASE_DATA_PATH = 'data/experiments/revision/exp6/datasets'
CONFIG_PATH = Path('configs/revision/exp6/data_generation')
FTSET_CONFIG_PATH = CONFIG_PATH / 'ftset'

# Cities CSV path for exp6
CITIES_CSV = f'{BASE_DATA_PATH}/cities_scattered_atlantis/cities.csv'


def create_task_config_with_atlantis(task: str) -> dict:
    """Create config for 1M pairs with scattered Atlantis."""
    return {
        'output_dir': f'{BASE_DATA_PATH}/{task}_1M_with_atlantis',
        'cities_csv': CITIES_CSV,
        'tokenizer_path': 'data/tokenizers/default_tokenizer',
        'pair_generation': {
            'strategy': 'all_pairs'
        },
        'n_train': 1000000,
        'n_val': 128,
        'n_test': 10000,
        'seed': 42,
        'leading_zeros': True,
        'n_id_digits': 4
    }


def create_task_config_no_atlantis(task: str) -> dict:
    """Create config for 1M pairs without Atlantis."""
    return {
        'output_dir': f'{BASE_DATA_PATH}/{task}_1M_no_atlantis',
        'cities_csv': CITIES_CSV,
        'tokenizer_path': 'data/tokenizers/default_tokenizer',
        'groups': {
            'world': {
                'city_names': '^(?!Atlantis_).*'  # Everything NOT starting with Atlantis_
            }
        },
        'pair_generation': {
            'strategy': 'within_groups',
            'groups': ['world']
        },
        'n_train': 1000000,
        'n_val': 128,
        'n_test': 10000,
        'seed': 42,
        'leading_zeros': True,
        'n_id_digits': 4
    }


def create_task_config_atlantis_required(task: str) -> dict:
    """Create config for 100k pairs with Atlantis required."""
    return {
        'output_dir': f'{BASE_DATA_PATH}/{task}_100k_atlantis_required',
        'cities_csv': CITIES_CSV,
        'tokenizer_path': 'data/tokenizers/default_tokenizer',
        'groups': {
            'atlantis': {
                'city_names': '^Atlantis_'  # Atlantis cities
            },
            'world': {
                'city_names': '^(?!Atlantis_).*'  # Non-Atlantis cities
            }
        },
        'pair_generation': {
            'strategy': 'must_include',
            'must_include_groups': ['atlantis'],
            'other_groups': ['world', 'atlantis']  # Can pair with world or other Atlantis
        },
        'n_train': 100000,
        'n_val': 128,
        'n_test': 10000,
        'seed': 42,
        'leading_zeros': True,
        'n_id_digits': 4
    }


def create_multitask_pt1_config() -> dict:
    """Create config for combined multitask_pt1 dataset."""
    return {
        'output_dir': f'{BASE_DATA_PATH}/multitask_pt1_with_atlantis',
        'mode': 'concat',
        'shuffle': True,
        'seed': 42,
        'datasets': [
            {'path': f'{BASE_DATA_PATH}/{task}_1M_with_atlantis'}
            for task in TASKS
        ]
    }


def create_ftwb1_config(task_idx: int) -> dict:
    """Create config for FTWB1 (single-task fine-tuning) dataset.

    FTWB1 combines:
    - 20k samples from main task (no atlantis) - world data
    - 100k samples from main task (atlantis required) - to train on Atlantis
    - 256 samples from each of 7 tasks (atlantis required) - warmup
    """
    task = TASKS[task_idx]

    datasets = [
        # Main task data
        {'path': f'{BASE_DATA_PATH}/{task}_1M_no_atlantis', 'n_samples': 20000},
        {'path': f'{BASE_DATA_PATH}/{task}_100k_atlantis_required', 'n_samples': 100000},
    ]

    # Warmup samples from all tasks
    for t in TASKS:
        datasets.append({'path': f'{BASE_DATA_PATH}/{t}_100k_atlantis_required', 'n_samples': 256})

    return {
        'output_dir': f'{BASE_DATA_PATH}/ftwb1-{task_idx + 1}',
        'mode': 'sample',
        'seed': 42,
        'shuffle': True,
        'datasets': datasets
    }


def create_ftwb2_config(pair_idx: int, task1_idx: int, task2_idx: int) -> dict:
    """Create config for FTWB2 (two-task fine-tuning) dataset.

    FTWB2 combines:
    - 20k samples from task1 (no atlantis)
    - 20k samples from task2 (no atlantis)
    - 100k samples from task1 (atlantis required)
    - 100k samples from task2 (atlantis required)
    - 256 samples from each of 7 tasks (atlantis required) - warmup
    """
    task1 = TASKS[task1_idx]
    task2 = TASKS[task2_idx]

    datasets = [
        # Main task data (no atlantis)
        {'path': f'{BASE_DATA_PATH}/{task1}_1M_no_atlantis', 'n_samples': 20000},
        {'path': f'{BASE_DATA_PATH}/{task2}_1M_no_atlantis', 'n_samples': 20000},
        # Main task data (atlantis required)
        {'path': f'{BASE_DATA_PATH}/{task1}_100k_atlantis_required', 'n_samples': 100000},
        {'path': f'{BASE_DATA_PATH}/{task2}_100k_atlantis_required', 'n_samples': 100000},
    ]

    # Warmup samples from all tasks
    for t in TASKS:
        datasets.append({'path': f'{BASE_DATA_PATH}/{t}_100k_atlantis_required', 'n_samples': 256})

    return {
        'output_dir': f'{BASE_DATA_PATH}/ftwb2-{pair_idx}',
        'mode': 'sample',
        'seed': 42,
        'shuffle': True,
        'datasets': datasets
    }


def main():
    # Create directories
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    FTSET_CONFIG_PATH.mkdir(parents=True, exist_ok=True)

    print("Generating Exp6 data generation configs...")
    print(f"Config path: {CONFIG_PATH}")
    print(f"FTSET config path: {FTSET_CONFIG_PATH}")
    print()

    # Track all configs
    all_configs = []

    # 1. Task configs with Atlantis (1M)
    print("Creating 7 task configs with Atlantis (1M)...")
    for task in TASKS:
        config = create_task_config_with_atlantis(task)
        config_path = CONFIG_PATH / f'{task}_1M_with_atlantis.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        all_configs.append(config_path)
        print(f"  Created: {config_path}")

    # 2. Task configs without Atlantis (1M)
    print("\nCreating 7 task configs without Atlantis (1M)...")
    for task in TASKS:
        config = create_task_config_no_atlantis(task)
        config_path = CONFIG_PATH / f'{task}_1M_no_atlantis.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        all_configs.append(config_path)
        print(f"  Created: {config_path}")

    # 3. Task configs with Atlantis required (100k)
    print("\nCreating 7 task configs with Atlantis required (100k)...")
    for task in TASKS:
        config = create_task_config_atlantis_required(task)
        config_path = CONFIG_PATH / f'{task}_100k_atlantis_required.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        all_configs.append(config_path)
        print(f"  Created: {config_path}")

    # 4. Multitask PT1 combined config
    print("\nCreating multitask_pt1 combined config...")
    config = create_multitask_pt1_config()
    config_path = FTSET_CONFIG_PATH / 'combine_multitask_pt1.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    all_configs.append(config_path)
    print(f"  Created: {config_path}")

    # 5. FTWB1 configs (7 single-task)
    print("\nCreating 7 FTWB1 configs...")
    for i in range(7):
        config = create_ftwb1_config(i)
        config_path = FTSET_CONFIG_PATH / f'combine_ftwb1-{i + 1}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        all_configs.append(config_path)
        print(f"  Created: {config_path} ({TASKS[i]})")

    # 6. FTWB2 configs (21 two-task combinations)
    print("\nCreating 21 FTWB2 configs...")
    pair_idx = 1
    for i, j in combinations(range(7), 2):
        config = create_ftwb2_config(pair_idx, i, j)
        config_path = FTSET_CONFIG_PATH / f'combine_ftwb2-{pair_idx}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        all_configs.append(config_path)
        print(f"  Created: {config_path} ({TASKS[i]} + {TASKS[j]})")
        pair_idx += 1

    print(f"\n{'='*60}")
    print(f"Total configs created: {len(all_configs)}")
    print(f"  - 7 task configs with Atlantis (1M)")
    print(f"  - 7 task configs without Atlantis (1M)")
    print(f"  - 7 task configs with Atlantis required (100k)")
    print(f"  - 1 multitask_pt1 combined config")
    print(f"  - 7 FTWB1 configs")
    print(f"  - 21 FTWB2 configs")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
