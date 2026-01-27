"""
Generate representation analysis configs for seed1 and seed2 experiments.
Uses the same format as existing configs in analysis_representation_higher/ftset/
"""
import sys
from pathlib import Path

project_root = Path('')
sys.path.insert(0, str(project_root))

import yaml
import argparse

TASK_NAMES = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing',
}

base_dir = Path('')

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='1,2', help='Comma-separated seed numbers (e.g., 1,2)')
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(',')]

for seed in seeds:
    print(f"Generating configs for seed{seed}...")

    for task_id, task_name in TASK_NAMES.items():
        exp_name = f'pt1-{task_id}_seed{seed}'

        for layer in [3, 4, 5, 6]:
            config = {
                'cities_csv': 'data/datasets/cities/cities.csv',
                'device': 'cuda',
                'experiment_dir': f'data/experiments/revision/exp4/{exp_name}',
                'layers': [layer],
                'method': {'name': 'linear'},
                'n_test_cities': 1250,
                'n_train_cities': 3250,
                'output_dir': f'/data/experiments/revision/exp4/{exp_name}/analysis_higher/{task_name}_firstcity_last_and_trans_l{layer}',
                'perform_pca': True,
                'probe_test': 'region:.* && city_id:^[1-9][0-9]{3,}$',
                'probe_train': 'region:.* && city_id:^[1-9][0-9]{3,}$',
                'prompt_format': f'{task_name}_firstcity_last_and_trans',
                'save_repr_ckpts': [-2],
                'seed': 42,
            }

            # Save config
            config_dir = base_dir / 'configs' / 'analysis_representation_higher' / f'seed{seed}' / exp_name
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / f'{task_name}_firstcity_last_and_trans_l{layer}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

total_configs = len(seeds) * 7 * 4
print(f"Generated {total_configs} configs in configs/analysis_representation_higher/seed*/")
print(f"{len(seeds)} seeds × 7 experiments × 4 layers = {total_configs} configs")
