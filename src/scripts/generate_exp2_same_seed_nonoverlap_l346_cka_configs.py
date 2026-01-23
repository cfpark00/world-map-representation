#!/usr/bin/env python3
"""
Generate CKA configs for PT2/PT3 SAME-SEED non-overlapping pairs, layers 3,4,6 only.
Layer 5 will come from the all-pairs analysis.
"""

from pathlib import Path
import yaml

# PT2 task combinations
PT2_TASKS = {
    'pt2-1': {'distance', 'trianglearea'},
    'pt2-2': {'angle', 'compass'},
    'pt2-3': {'inside', 'perimeter'},
    'pt2-4': {'crossing', 'distance'},
    'pt2-5': {'trianglearea', 'angle'},
    'pt2-6': {'compass', 'inside'},
    'pt2-7': {'perimeter', 'crossing'},
}

# PT3 task combinations
PT3_TASKS = {
    'pt3-1': {'distance', 'trianglearea', 'angle'},
    'pt3-2': {'compass', 'inside', 'perimeter'},
    'pt3-3': {'crossing', 'distance', 'trianglearea'},
    'pt3-4': {'angle', 'compass', 'inside'},
    'pt3-5': {'perimeter', 'crossing', 'distance'},
    'pt3-6': {'trianglearea', 'angle', 'compass'},
    'pt3-7': {'inside', 'perimeter', 'crossing'},
}

# Extract task for representation (first task from each combo)
PT2_REPR_TASKS = {
    'pt2-1': 'distance',
    'pt2-2': 'angle',
    'pt2-3': 'inside',
    'pt2-4': 'crossing',
    'pt2-5': 'trianglearea',
    'pt2-6': 'compass',
    'pt2-7': 'perimeter',
}

PT3_REPR_TASKS = {
    'pt3-1': 'distance',
    'pt3-2': 'compass',
    'pt3-3': 'crossing',
    'pt3-4': 'angle',
    'pt3-5': 'perimeter',
    'pt3-6': 'trianglearea',
    'pt3-7': 'inside',
}


def shares_tasks(tasks1, tasks2):
    """Check if two task sets have any overlap."""
    return len(set(tasks1) & set(tasks2)) > 0


def get_non_overlapping_pairs(task_dict):
    """Get all non-overlapping pairs."""
    pairs = []
    variants = list(range(1, 8))  # 1-7

    for i, var1 in enumerate(variants):
        for var2 in variants[i+1:]:
            name1 = f'pt{len(list(task_dict.values())[0])}-{var1}'
            name2 = f'pt{len(list(task_dict.values())[0])}-{var2}'

            if name1 not in task_dict or name2 not in task_dict:
                continue

            if not shares_tasks(task_dict[name1], task_dict[name2]):
                pairs.append((var1, var2))

    return pairs


def get_model_path(prefix, variant, seed, base_path):
    """Get the path to a model's representations."""
    if seed == 'orig':
        return base_path / 'data' / 'experiments' / f'{prefix}-{variant}'
    else:
        return base_path / 'data' / 'experiments' / 'revision' / 'exp2' / f'{prefix}-{variant}_seed{seed}'


def generate_cka_config(prefix, var1, var2, seed, layer, repr_tasks, base_path, config_dir):
    """Generate a single CKA config file."""

    # Get experiment names
    exp1_name = f'{prefix}-{var1}' if seed == 'orig' else f'{prefix}-{var1}_seed{seed}'
    exp2_name = f'{prefix}-{var2}' if seed == 'orig' else f'{prefix}-{var2}_seed{seed}'

    # Get tasks
    task1 = repr_tasks[f'{prefix}-{var1}']
    task2 = repr_tasks[f'{prefix}-{var2}']

    # Get paths
    exp1_path = get_model_path(prefix, var1, seed, base_path)
    exp2_path = get_model_path(prefix, var2, seed, base_path)

    # Create pair name
    pair_name = f'{prefix}-{var1}_vs_{prefix}-{var2}'
    if seed != 'orig':
        pair_name += f'_seed{seed}'

    # Output directory
    output_dir = f'data/experiments/revision/exp2/cka_analysis_same_seed/{pair_name}/layer{layer}'

    config = {
        'exp1': {
            'name': exp1_name,
            'repr_dir': str(exp1_path / 'analysis_higher' / f'{task1}_firstcity_last_and_trans_l{layer}' / 'representations'),
            'task': task1,
        },
        'exp2': {
            'name': exp2_name,
            'repr_dir': str(exp2_path / 'analysis_higher' / f'{task2}_firstcity_last_and_trans_l{layer}' / 'representations'),
            'task': task2,
        },
        'layer': layer,
        'checkpoint_steps': None,
        'use_final_only': True,
        'city_filter': 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$',
        'kernel_type': 'linear',
        'center_kernels': True,
        'use_gpu': True,
        'save_timeline_plot': False,
        'output_dir': output_dir,
    }

    # Create config filename
    config_filename = f'cka_{pair_name}_l{layer}.yaml'

    config_path = config_dir / config_filename
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def main():
    base_path = Path(__file__).resolve().parents[2]
    config_dir = base_path / 'configs' / 'revision' / 'exp2' / 'cka_same_seed_nonoverlap_l346'
    config_dir.mkdir(parents=True, exist_ok=True)

    layers = [3, 4, 6]  # Exclude layer 5
    seeds = ['orig', '1', '2']

    # Get non-overlapping pairs
    pt2_pairs = get_non_overlapping_pairs(PT2_TASKS)
    pt3_pairs = get_non_overlapping_pairs(PT3_TASKS)

    print(f"PT2 non-overlapping pairs: {len(pt2_pairs)}")
    print(f"PT3 non-overlapping pairs: {len(pt3_pairs)}")

    pt2_configs = 0
    pt3_configs = 0

    # Generate PT2 configs (same-seed only)
    for var1, var2 in pt2_pairs:
        for seed in seeds:
            for layer in layers:
                generate_cka_config('pt2', var1, var2, seed, layer, PT2_REPR_TASKS, base_path, config_dir)
                pt2_configs += 1

    # Generate PT3 configs (same-seed only)
    for var1, var2 in pt3_pairs:
        for seed in seeds:
            for layer in layers:
                generate_cka_config('pt3', var1, var2, seed, layer, PT3_REPR_TASKS, base_path, config_dir)
                pt3_configs += 1

    print(f"\nCreated {pt2_configs} PT2 configs ({len(pt2_pairs)} pairs × {len(seeds)} seeds × {len(layers)} layers)")
    print(f"Created {pt3_configs} PT3 configs ({len(pt3_pairs)} pairs × {len(seeds)} seeds × {len(layers)} layers)")
    print(f"Total: {pt2_configs + pt3_configs} configs")
    print(f"\nOutput dir: {config_dir}")


if __name__ == '__main__':
    main()
