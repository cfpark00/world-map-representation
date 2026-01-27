"""
Generate CKA configs for cross-seed analysis (21x21 matrix: 7 tasks × 3 seeds).

This creates configs to compute CKA between all pairs of:
- Original PT1 experiments (pt1-1 through pt1-7, seed 42)
- Seed1 PT1 experiments (pt1-1_seed1 through pt1-7_seed1)
- Seed2 PT1 experiments (pt1-1_seed2 through pt1-7_seed2)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('')
sys.path.insert(0, str(project_root))

import yaml
import argparse
from itertools import product


# Task mappings
TASK_NAMES = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing',
}


def get_repr_path_cross_seed(exp_name, task_name, layer, seed_num=None):
    """Get representation path for cross-seed experiments.

    Args:
        exp_name: Experiment name (e.g., 'pt1-1', 'pt1-1_seed1', 'pt1-1_seed2')
        task_name: Task name
        layer: Layer number
        seed_num: Seed number (None for original, 1 for seed1, 2 for seed2)
    """
    prompt_format = f"{task_name}_firstcity_last_and_trans"

    if seed_num is not None:
        base_path = f'data/experiments/revision/exp4/{exp_name}'
    else:
        base_path = f'data/experiments/{exp_name}'

    repr_dir = f'{base_path}/analysis_higher/{prompt_format}_l{layer}/representations'

    return repr_dir


def generate_cka_config_cross_seed(exp1_name, exp2_name, task1_name, task2_name,
                                   layer, seed_num1, seed_num2, output_base):
    """
    Generate a single CKA config for cross-seed analysis.

    Args:
        exp1_name: First experiment name (e.g., 'pt1-1', 'pt1-1_seed1', 'pt1-1_seed2')
        exp2_name: Second experiment name
        task1_name: Task name for exp1
        task2_name: Task name for exp2
        layer: Layer number
        seed_num1: Seed number for exp1 (None for original, 1, 2, etc.)
        seed_num2: Seed number for exp2
        output_base: Base output directory

    Returns:
        Config dictionary
    """
    repr1_dir = get_repr_path_cross_seed(exp1_name, task1_name, layer, seed_num=seed_num1)
    repr2_dir = get_repr_path_cross_seed(exp2_name, task2_name, layer, seed_num=seed_num2)

    # Output directory: data/analysis_v2/cka/pt1_cross_seed/{exp1}_vs_{exp2}/layer{layer}
    output_dir = output_base / 'pt1_cross_seed' / f'{exp1_name}_vs_{exp2_name}' / f'layer{layer}'

    config = {
        'output_dir': str(output_dir),
        'exp1': {
            'name': exp1_name,
            'repr_dir': repr1_dir,
            'task': task1_name,
        },
        'exp2': {
            'name': exp2_name,
            'repr_dir': repr2_dir,
            'task': task2_name,
        },
        'layer': layer,
        'checkpoint_steps': None,  # null = all checkpoints
        'city_filter': 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$',
        'kernel_type': 'linear',
        'center_kernels': True,
        'use_gpu': True,
        'save_timeline_plot': True,
    }

    return config


def main(layers=[4, 5, 6], seeds=[1, 2], base_dir=None, output_base=None):
    """
    Generate all CKA configs for cross-seed 21x21 matrix.

    Creates configs for all pairs of:
    - 7 original PT1 experiments (seed 42)
    - 7 seed1 PT1 experiments
    - 7 seed2 PT1 experiments
    Total: 21 × 21 = 441 pairs × 4 layers (upper triangle + diagonal)
    """
    if base_dir is None:
        base_dir = Path('')
    else:
        base_dir = Path(base_dir)

    if output_base is None:
        config_output = base_dir / 'configs' / 'analysis_v2' / 'cka_cross_seed'
        data_output = base_dir / 'data' / 'analysis_v2' / 'cka'
    else:
        config_output = Path(output_base) / 'configs'
        data_output = Path(output_base) / 'data'

    n_exps = 7 * (1 + len(seeds))  # 7 tasks × (original + seeds)
    print(f"Generating CKA configs for {n_exps}x{n_exps} cross-seed matrix...")

    # Create list of all experiments (7 original + 7 per seed)
    all_experiments = []

    # Original experiments
    for task_id, task_name in TASK_NAMES.items():
        exp_name = f'pt1-{task_id}'
        all_experiments.append({
            'exp_name': exp_name,
            'task_name': task_name,
            'seed_num': None,
        })

    # Seed experiments
    for seed in seeds:
        for task_id, task_name in TASK_NAMES.items():
            exp_name = f'pt1-{task_id}_seed{seed}'
            all_experiments.append({
                'exp_name': exp_name,
                'task_name': task_name,
                'seed_num': seed,
            })

    config_count = 0

    # Generate all pairs (upper triangle + diagonal)
    for i, exp1_info in enumerate(all_experiments):
        for exp2_info in all_experiments[i:]:  # Only upper triangle + diagonal
            exp1_name = exp1_info['exp_name']
            exp2_name = exp2_info['exp_name']
            task1_name = exp1_info['task_name']
            task2_name = exp2_info['task_name']
            seed_num1 = exp1_info['seed_num']
            seed_num2 = exp2_info['seed_num']

            for layer in layers:
                config = generate_cka_config_cross_seed(
                    exp1_name, exp2_name,
                    task1_name, task2_name,
                    layer,
                    seed_num1, seed_num2,
                    data_output
                )

                # Save config
                config_dir = config_output / f'{exp1_name}_vs_{exp2_name}'
                config_dir.mkdir(parents=True, exist_ok=True)

                config_path = config_dir / f'layer{layer}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                config_count += 1

    print(f"\nGenerated {config_count} config files in {config_output}")
    print(f"Total pairs: {len(all_experiments) * (len(all_experiments) + 1) // 2} ({n_exps}x{n_exps} upper triangle + diagonal)")
    print(f"Configs per layer: {config_count // len(layers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CKA configs for cross-seed analysis')
    parser.add_argument('--layers', type=str, default='3,4,5,6',
                       help='Comma-separated layer numbers (default: 3,4,5,6)')
    parser.add_argument('--seeds', type=str, default='1,2',
                       help='Comma-separated seed numbers (default: 1,2)')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base WM_1 directory (default: auto-detect)')
    parser.add_argument('--output-base', type=str, default=None,
                       help='Base output directory')

    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    main(
        layers=layers,
        seeds=seeds,
        base_dir=args.base_dir,
        output_base=args.output_base
    )
