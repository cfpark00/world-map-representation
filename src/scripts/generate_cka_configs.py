"""
Generate CKA config files automatically for all experiment pairs.

This script creates YAML config files for CKA analysis following the clean
organizational structure from world-representation.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('')
sys.path.insert(0, str(project_root))

import argparse
import yaml
from itertools import combinations

from src.analysis.cka_v2.experiment_registry import (
    get_pt1_experiments,
    get_pt2_experiments,
    get_pt3_experiments,
    get_repr_path,
    generate_experiment_pairs,
    TASK_NAMES,
)


def generate_config(exp1_name: str, exp2_name: str, layer: int,
                   exp1_task: str, exp2_task: str,
                   output_base: Path,
                   group_name: str,
                   base_dir: Path) -> dict:
    """
    Generate a single CKA config.

    Args:
        exp1_name: First experiment name
        exp2_name: Second experiment name
        layer: Layer number
        exp1_task: Task name for exp1
        exp2_task: Task name for exp2
        output_base: Base output directory
        group_name: Group name (e.g., 'pt1_all_pairs')
        base_dir: Base WM_1 directory

    Returns:
        Config dictionary
    """
    # Construct paths
    repr1_dir = get_repr_path(exp1_name, exp1_task, layer, base_dir=base_dir)
    repr2_dir = get_repr_path(exp2_name, exp2_task, layer, base_dir=base_dir)

    # Output directory structure: data/analysis_v2/cka/{group_name}/{exp1}_vs_{exp2}/layer{layer}
    output_dir = output_base / group_name / f'{exp1_name}_vs_{exp2_name}' / f'layer{layer}'

    config = {
        'output_dir': str(output_dir),
        'exp1': {
            'name': exp1_name,
            'repr_dir': str(repr1_dir),
            'task': exp1_task,
        },
        'exp2': {
            'name': exp2_name,
            'repr_dir': str(repr2_dir),
            'task': exp2_task,
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


def main(experiment_type='pt1', layers=[3, 4, 5, 6], base_dir=None, output_base=None):
    """
    Generate all CKA configs for a given experiment type.

    Args:
        experiment_type: 'pt1', 'pt2', or 'pt3'
        layers: List of layer numbers to analyze
        base_dir: Base WM_1 directory
        output_base: Base output directory for configs
    """
    if base_dir is None:
        base_dir = Path('')
    else:
        base_dir = Path(base_dir)

    if output_base is None:
        output_base = base_dir / 'configs' / 'analysis_v2' / 'cka'
    else:
        output_base = Path(output_base)

    # Get experiments
    if experiment_type == 'pt1':
        experiments = get_pt1_experiments(base_dir)
        group_name = 'pt1_all_pairs'
    elif experiment_type == 'pt2':
        experiments = get_pt2_experiments(base_dir)
        group_name = 'pt2_all_pairs'
    elif experiment_type == 'pt3':
        experiments = get_pt3_experiments(base_dir)
        group_name = 'pt3_all_pairs'
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    print(f"Generating CKA configs for {experiment_type}")
    print(f"Found {len(experiments)} experiments")

    # Generate pairs
    pairs = generate_experiment_pairs(experiment_type, base_dir)
    print(f"Generating configs for {len(pairs)} pairs × {len(layers)} layers = {len(pairs) * len(layers)} total configs")

    config_count = 0

    for exp1_name, exp2_name in pairs:
        exp1 = experiments[exp1_name]
        exp2 = experiments[exp2_name]

        # Get primary task for each experiment
        exp1_task = exp1['tasks'][0]
        exp2_task = exp2['tasks'][0]

        for layer in layers:
            # Check if representation directories exist
            repr1_dir = get_repr_path(exp1_name, exp1_task, layer, base_dir=base_dir)
            repr2_dir = get_repr_path(exp2_name, exp2_task, layer, base_dir=base_dir)

            if not repr1_dir.exists():
                print(f"  Warning: {repr1_dir} does not exist, skipping")
                continue

            if not repr2_dir.exists():
                print(f"  Warning: {repr2_dir} does not exist, skipping")
                continue

            # Generate config
            config = generate_config(
                exp1_name, exp2_name, layer,
                exp1_task, exp2_task,
                base_dir / 'data' / 'analysis_v2' / 'cka',
                group_name,
                base_dir
            )

            # Save config
            config_dir = output_base / group_name / f'{exp1_name}_vs_{exp2_name}'
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / f'layer{layer}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            config_count += 1

    print(f"\nGenerated {config_count} config files in {output_base / group_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CKA config files')
    parser.add_argument('--type', type=str, default='pt1', choices=['pt1', 'pt2', 'pt3'],
                       help='Experiment type (pt1, pt2, or pt3)')
    parser.add_argument('--layers', type=str, default='3,4,5,6',
                       help='Comma-separated layer numbers (default: 3,4,5,6)')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base WM_1 directory (default: auto-detect)')
    parser.add_argument('--output-base', type=str, default=None,
                       help='Base output directory for configs (default: configs/analysis_v2/cka)')

    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]

    main(
        experiment_type=args.type,
        layers=layers,
        base_dir=args.base_dir,
        output_base=args.output_base
    )
