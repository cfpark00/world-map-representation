#!/usr/bin/env python3
"""
Generate CKA configs for ALL PT2 pairs (including overlapping tasks).
Generates configs for all 21×21 combinations across orig, seed1, seed2.
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


def get_model_path(variant, seed, base_path):
    """Get the path to a model's representations."""
    if seed == 'orig':
        return base_path / 'data' / 'experiments' / f'pt2-{variant}'
    else:
        return base_path / 'data' / 'experiments' / 'revision' / 'exp2' / f'pt2-{variant}_seed{seed}'


def generate_cka_config(var1, var2, seed1, seed2, layer, base_path, config_dir):
    """Generate a single CKA config file."""

    # Get experiment names
    exp1_name = f'pt2-{var1}' if seed1 == 'orig' else f'pt2-{var1}_seed{seed1}'
    exp2_name = f'pt2-{var2}' if seed2 == 'orig' else f'pt2-{var2}_seed{seed2}'

    # Get tasks
    task1 = PT2_REPR_TASKS[f'pt2-{var1}']
    task2 = PT2_REPR_TASKS[f'pt2-{var2}']

    # Get paths
    exp1_path = get_model_path(var1, seed1, base_path)
    exp2_path = get_model_path(var2, seed2, base_path)

    # Create pair name
    pair_name = f'pt2-{var1}_vs_pt2-{var2}'
    if seed1 != 'orig' and seed2 != 'orig':
        pair_name = f'pt2-{var1}_seed{seed1}_vs_pt2-{var2}_seed{seed2}'
    elif seed1 != 'orig':
        pair_name = f'pt2-{var1}_seed{seed1}_vs_pt2-{var2}'
    elif seed2 != 'orig':
        pair_name = f'pt2-{var1}_vs_pt2-{var2}_seed{seed2}'

    # Output directory
    output_dir = f'data/experiments/revision/exp2/cka_analysis_all/{pair_name}/layer{layer}'

    # Build repr_dir paths (directory containing checkpoints, not single file)
    repr1_dir = str(exp1_path / 'analysis_higher' / f'{task1}_firstcity_last_and_trans_l{layer}' / 'representations')
    repr2_dir = str(exp2_path / 'analysis_higher' / f'{task2}_firstcity_last_and_trans_l{layer}' / 'representations')

    config = {
        'exp1': {
            'name': exp1_name,
            'repr_dir': repr1_dir,
            'task': task1,
        },
        'exp2': {
            'name': exp2_name,
            'repr_dir': repr2_dir,
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
    config_filename = f'cka_pt2-{var1}_vs_pt2-{var2}'
    if seed1 != 'orig' and seed2 != 'orig':
        config_filename = f'cka_pt2-{var1}_seed{seed1}_vs_pt2-{var2}_seed{seed2}'
    elif seed1 != 'orig':
        config_filename = f'cka_pt2-{var1}_seed{seed1}_vs_pt2-{var2}'
    elif seed2 != 'orig':
        config_filename = f'cka_pt2-{var1}_vs_pt2-{var2}_seed{seed2}'
    config_filename += f'_l{layer}.yaml'

    config_path = config_dir / config_filename
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def main():
    base_path = Path(__file__).resolve().parents[2]
    config_dir = base_path / 'configs' / 'revision' / 'exp2' / 'cka_analysis_all'
    config_dir.mkdir(parents=True, exist_ok=True)

    layers = [3, 4, 5, 6]
    seeds = ['orig', '1', '2']
    variants = list(range(1, 8))  # pt2-1 through pt2-7

    # Generate all experiments
    all_experiments = []
    for var in variants:
        for seed in seeds:
            all_experiments.append((var, seed))

    print(f"Total PT2 experiments: {len(all_experiments)}")

    configs_created = 0

    # Generate all pairs (including self-comparisons across seeds)
    for i, (var1, seed1) in enumerate(all_experiments):
        for j, (var2, seed2) in enumerate(all_experiments):
            # Skip if same experiment
            if i >= j:
                continue

            for layer in layers:
                config_path = generate_cka_config(
                    var1, var2, seed1, seed2, layer, base_path, config_dir
                )
                configs_created += 1

    print(f"Created {configs_created} CKA configs in {config_dir}")
    print(f"Configs per layer: {configs_created // len(layers)}")
    print(f"Expected: {len(all_experiments) * (len(all_experiments) - 1) // 2} pairs × {len(layers)} layers = {len(all_experiments) * (len(all_experiments) - 1) // 2 * len(layers)} total")


if __name__ == '__main__':
    main()
