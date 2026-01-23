#!/usr/bin/env python3
"""
Generate CKA configs for ALL PT2 and PT3 pairs (including overlapping tasks), layer 5 only.
"""

from pathlib import Path
import yaml

# PT2 task combinations
PT2_REPR_TASKS = {
    'pt2-1': 'distance',
    'pt2-2': 'angle',
    'pt2-3': 'inside',
    'pt2-4': 'crossing',
    'pt2-5': 'trianglearea',
    'pt2-6': 'compass',
    'pt2-7': 'perimeter',
}

# PT3 task combinations
PT3_REPR_TASKS = {
    'pt3-1': 'distance',
    'pt3-2': 'compass',
    'pt3-3': 'crossing',
    'pt3-4': 'angle',
    'pt3-5': 'perimeter',
    'pt3-6': 'trianglearea',
    'pt3-7': 'inside',
}


def get_model_path(prefix, variant, seed, base_path):
    """Get the path to a model's representations."""
    if seed == 'orig':
        return base_path / 'data' / 'experiments' / f'{prefix}-{variant}'
    else:
        return base_path / 'data' / 'experiments' / 'revision' / 'exp2' / f'{prefix}-{variant}_seed{seed}'


def generate_cka_config(prefix, var1, var2, seed1, seed2, repr_tasks, base_path, config_dir):
    """Generate a single CKA config file for layer 5."""
    layer = 5

    # Get experiment names
    exp1_name = f'{prefix}-{var1}' if seed1 == 'orig' else f'{prefix}-{var1}_seed{seed1}'
    exp2_name = f'{prefix}-{var2}' if seed2 == 'orig' else f'{prefix}-{var2}_seed{seed2}'

    # Get tasks
    task1 = repr_tasks[f'{prefix}-{var1}']
    task2 = repr_tasks[f'{prefix}-{var2}']

    # Get paths
    exp1_path = get_model_path(prefix, var1, seed1, base_path)
    exp2_path = get_model_path(prefix, var2, seed2, base_path)

    # Create pair name
    pair_name = f'{prefix}-{var1}_vs_{prefix}-{var2}'
    if seed1 != 'orig' and seed2 != 'orig':
        pair_name = f'{prefix}-{var1}_seed{seed1}_vs_{prefix}-{var2}_seed{seed2}'
    elif seed1 != 'orig':
        pair_name = f'{prefix}-{var1}_seed{seed1}_vs_{prefix}-{var2}'
    elif seed2 != 'orig':
        pair_name = f'{prefix}-{var1}_vs_{prefix}-{var2}_seed{seed2}'

    # Output directory
    output_dir = f'data/experiments/revision/exp2/cka_analysis_all/{pair_name}/layer{layer}'

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
    config_dir = base_path / 'configs' / 'revision' / 'exp2' / 'cka_analysis_all_l5'
    config_dir.mkdir(parents=True, exist_ok=True)

    seeds = ['orig', '1', '2']

    # Process PT2
    pt2_variants = list(range(1, 8))  # pt2-1 through pt2-7
    pt2_experiments = [(v, s) for v in pt2_variants for s in seeds]

    # Process PT3
    pt3_variants = list(range(1, 8))  # pt3-1 through pt3-7
    pt3_experiments = [(v, s) for v in pt3_variants for s in seeds]

    print(f"Total PT2 experiments: {len(pt2_experiments)}")
    print(f"Total PT3 experiments: {len(pt3_experiments)}")

    pt2_configs = 0
    pt3_configs = 0

    # Generate all PT2 pairs
    for i, (var1, seed1) in enumerate(pt2_experiments):
        for j, (var2, seed2) in enumerate(pt2_experiments):
            if i >= j:
                continue
            generate_cka_config('pt2', var1, var2, seed1, seed2, PT2_REPR_TASKS, base_path, config_dir)
            pt2_configs += 1

    # Generate all PT3 pairs
    for i, (var1, seed1) in enumerate(pt3_experiments):
        for j, (var2, seed2) in enumerate(pt3_experiments):
            if i >= j:
                continue
            generate_cka_config('pt3', var1, var2, seed1, seed2, PT3_REPR_TASKS, base_path, config_dir)
            pt3_configs += 1

    print(f"\nCreated {pt2_configs} PT2 configs")
    print(f"Created {pt3_configs} PT3 configs")
    print(f"Total: {pt2_configs + pt3_configs} configs in {config_dir}")


if __name__ == '__main__':
    main()
