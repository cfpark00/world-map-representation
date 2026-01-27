"""
Generate representation extraction configs for PT1 experiments (original + seed1).

Creates configs for extracting representations with task-specific prompts.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('')
sys.path.insert(0, str(project_root))

import yaml
import argparse


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


def generate_repr_extraction_config(exp_dir, task_name, layer, output_base, seed_suffix=''):
    """
    Generate a single representation extraction config.

    Args:
        exp_dir: Path to experiment directory (e.g., 'data/experiments/pt1-1')
        task_name: Task name (e.g., 'distance', 'trianglearea')
        layer: Layer number (3, 4, 5, or 6)
        output_base: Base directory for configs
        seed_suffix: Suffix for seed experiments (e.g., '_seed1')

    Returns:
        Config dictionary
    """
    prompt_format = f"{task_name}_firstcity_last_and_trans"

    # Output directory in the experiment folder
    output_dir = f"{exp_dir}/analysis_higher/{prompt_format}_l{layer}"

    config = {
        'experiment_dir': str(exp_dir),
        'output_dir': output_dir,
        'cities_csv': 'data/datasets/cities/cities.csv',
        'device': 'cuda',
        'layers': [layer],
        'prompt_format': prompt_format,
        'save_repr_ckpts': [-2],  # Save all checkpoints
        'perform_pca': True,
        'n_train_cities': 3250,
        'n_test_cities': 1250,
        'probe_train': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'probe_test': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'method': {
            'name': 'linear'
        },
        'seed': 42,
    }

    return config


def main(layers=[3, 4, 5, 6], include_original=True, include_seed1=True, base_dir=None, output_base=None):
    """
    Generate all representation extraction configs.

    Args:
        layers: List of layer numbers
        include_original: Whether to include original PT1 experiments
        include_seed1: Whether to include seed1 experiments
        base_dir: Base WM_1 directory
        output_base: Base output directory for configs
    """
    if base_dir is None:
        base_dir = Path('')
    else:
        base_dir = Path(base_dir)

    if output_base is None:
        output_base = base_dir / 'configs' / 'analysis_v2' / 'representation_extraction'
    else:
        output_base = Path(output_base)

    config_count = 0

    # Generate configs for original PT1 experiments
    if include_original:
        print("Generating configs for original PT1 experiments...")
        for task_id, task_name in TASK_NAMES.items():
            exp_name = f'pt1-{task_id}'
            exp_dir = base_dir / 'data' / 'experiments' / exp_name

            if not exp_dir.exists():
                print(f"  Warning: {exp_dir} does not exist, skipping")
                continue

            for layer in layers:
                config = generate_repr_extraction_config(
                    exp_dir=f'data/experiments/{exp_name}',
                    task_name=task_name,
                    layer=layer,
                    output_base=output_base
                )

                # Save config
                config_dir = output_base / 'original' / exp_name
                config_dir.mkdir(parents=True, exist_ok=True)

                config_path = config_dir / f'{task_name}_last_and_trans_l{layer}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                config_count += 1

    # Generate configs for seed1 PT1 experiments
    if include_seed1:
        print("Generating configs for seed1 PT1 experiments...")
        for task_id, task_name in TASK_NAMES.items():
            exp_name = f'pt1-{task_id}_seed1'
            exp_dir = base_dir / 'data' / 'experiments' / 'revision' / 'exp4' / exp_name

            if not exp_dir.exists():
                print(f"  Warning: {exp_dir} does not exist, skipping")
                continue

            for layer in layers:
                config = generate_repr_extraction_config(
                    exp_dir=f'data/experiments/revision/exp4/{exp_name}',
                    task_name=task_name,
                    layer=layer,
                    output_base=output_base,
                    seed_suffix='_seed1'
                )

                # Save config
                config_dir = output_base / 'seed1' / exp_name
                config_dir.mkdir(parents=True, exist_ok=True)

                config_path = config_dir / f'{task_name}_last_and_trans_l{layer}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                config_count += 1

    print(f"\nGenerated {config_count} config files in {output_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate representation extraction config files')
    parser.add_argument('--layers', type=str, default='3,4,5,6',
                       help='Comma-separated layer numbers (default: 3,4,5,6)')
    parser.add_argument('--original-only', action='store_true',
                       help='Only generate configs for original experiments')
    parser.add_argument('--seed1-only', action='store_true',
                       help='Only generate configs for seed1 experiments')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base WM_1 directory (default: auto-detect)')
    parser.add_argument('--output-base', type=str, default=None,
                       help='Base output directory for configs')

    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]

    include_original = not args.seed1_only
    include_seed1 = not args.original_only

    main(
        layers=layers,
        include_original=include_original,
        include_seed1=include_seed1,
        base_dir=args.base_dir,
        output_base=args.output_base
    )
