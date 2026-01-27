"""
Generate missing CKA configs for 21x21 matrix (layer 3 and layer 6 for pt1-5_seed3).
"""
from pathlib import Path
import yaml

# Task mapping
TASK_NAMES = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing',
}

# Create list of 21 model names
model_names = []
for task_id in range(1, 8):
    model_names.append(f'pt1-{task_id}')
    model_names.append(f'pt1-{task_id}_seed1')
    if task_id == 5:
        model_names.append(f'pt1-{task_id}_seed3')
    else:
        model_names.append(f'pt1-{task_id}_seed2')

# Map to task names
model_to_task = {}
for task_id in range(1, 8):
    task_name = TASK_NAMES[task_id]
    model_to_task[f'pt1-{task_id}'] = task_name
    model_to_task[f'pt1-{task_id}_seed1'] = task_name
    if task_id == 5:
        model_to_task[f'pt1-{task_id}_seed3'] = task_name
    else:
        model_to_task[f'pt1-{task_id}_seed2'] = task_name

# Base directories
base_dir = Path('data/experiments')
config_base = Path('configs/revision/exp4/cka_cross_seed')

# Check existing results
results_dir = Path('data/experiments/revision/exp4/cka_analysis')

def get_repr_dir(exp_name, layer):
    """Get representation directory for an experiment."""
    task = model_to_task[exp_name]

    # Determine base path
    if exp_name.startswith('pt1-') and '_seed' not in exp_name:
        # Original experiments
        base_path = base_dir / exp_name
    else:
        # Seed experiments
        base_path = base_dir / 'revision' / 'exp4' / exp_name

    repr_path = base_path / 'analysis_higher' / f'{task}_firstcity_last_and_trans_l{layer}' / 'representations'
    return str(repr_path)

def create_cka_config(exp1, exp2, layer, output_path):
    """Create a CKA config file."""
    task1 = model_to_task[exp1]
    task2 = model_to_task[exp2]

    # Determine output directory
    pair_name = f'{exp1}_vs_{exp2}'
    output_dir = f'/data/experiments/revision/exp4/cka_analysis/{pair_name}/layer{layer}'

    config = {
        'exp1': {
            'name': exp1,
            'task': task1,
            'repr_dir': get_repr_dir(exp1, layer),
        },
        'exp2': {
            'name': exp2,
            'task': task2,
            'repr_dir': get_repr_dir(exp2, layer),
        },
        'layer': layer,
        'output_dir': output_dir,
        'kernel_type': 'linear',
        'center_kernels': True,
        'use_gpu': True,
        'checkpoint_steps': [328146],
        'city_filter': 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$',
        'save_timeline_plot': False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

# Generate missing configs
layer3_configs = []
layer6_missing_configs = []

for i in range(21):
    for j in range(i, 21):
        exp1 = model_names[i]
        exp2 = model_names[j]
        pair_name = f'{exp1}_vs_{exp2}'

        # Check layer 3
        result_file_l3 = results_dir / pair_name / 'layer3' / 'summary.json'
        if not result_file_l3.exists():
            # Check reverse
            pair_name_rev = f'{exp2}_vs_{exp1}'
            result_file_l3_rev = results_dir / pair_name_rev / 'layer3' / 'summary.json'
            if not result_file_l3_rev.exists():
                # Need to create layer 3 config
                config_path = config_base / pair_name / 'layer3.yaml'
                create_cka_config(exp1, exp2, 3, config_path)
                layer3_configs.append(str(config_path))

        # Check layer 6
        result_file_l6 = results_dir / pair_name / 'layer6' / 'summary.json'
        if not result_file_l6.exists():
            # Check reverse
            pair_name_rev = f'{exp2}_vs_{exp1}'
            result_file_l6_rev = results_dir / pair_name_rev / 'layer6' / 'summary.json'
            if not result_file_l6_rev.exists():
                # Need to create layer 6 config
                config_path = config_base / pair_name / 'layer6.yaml'
                if not config_path.exists():
                    create_cka_config(exp1, exp2, 6, config_path)
                    layer6_missing_configs.append(str(config_path))

print(f"Generated {len(layer3_configs)} layer 3 configs")
print(f"Generated {len(layer6_missing_configs)} layer 6 configs")
print(f"\nTotal new configs: {len(layer3_configs) + len(layer6_missing_configs)}")
