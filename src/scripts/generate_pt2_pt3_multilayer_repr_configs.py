#!/usr/bin/env python3
"""Generate multi-layer representation extraction configs for PT2/PT3 seed experiments."""

import yaml
from pathlib import Path

# PT2 task combinations - use first task from each pair
PT2_TASKS = {
    1: 'distance',      # distance+trianglearea
    2: 'angle',         # angle+compass
    3: 'inside',        # inside+perimeter
    4: 'crossing',      # crossing+distance
    5: 'trianglearea',  # trianglearea+angle
    6: 'compass',       # compass+inside
    7: 'perimeter',     # perimeter+crossing
}

# PT3 task combinations - use first task from each triple
PT3_TASKS = {
    1: 'distance',      # distance+trianglearea+angle
    2: 'compass',       # compass+inside+perimeter
    3: 'crossing',      # crossing+distance+trianglearea
    4: 'angle',         # angle+compass+inside
    5: 'perimeter',     # perimeter+crossing+distance
    6: 'trianglearea',  # trianglearea+angle+compass
    7: 'inside',        # inside+perimeter+crossing
}

SEEDS = [1, 2]
LAYERS = [3, 4, 5, 6]

def create_repr_extraction_config(exp_type, variant_num, seed, layer):
    """Create representation extraction config for PT2/PT3."""
    task_name = PT2_TASKS[variant_num] if exp_type == 'pt2' else PT3_TASKS[variant_num]
    prompt_format = f'{task_name}_firstcity_last_and_trans'

    experiment_dir = f'data/experiments/revision/exp2/{exp_type}-{variant_num}_seed{seed}'
    output_dir = f'/n/home12/cfpark00/WM_1/{experiment_dir}/analysis_higher/{prompt_format}_l{layer}'

    config = {
        'cities_csv': 'data/datasets/cities/cities.csv',
        'device': 'cuda',
        'layers': [layer],
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

def main():
    base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')

    # Create directory structure
    pt2_config_dir = base_dir / 'configs/revision/exp2/pt2_seed/extract_representations_multilayer'
    pt3_config_dir = base_dir / 'configs/revision/exp2/pt3_seed/extract_representations_multilayer'

    pt2_config_dir.mkdir(parents=True, exist_ok=True)
    pt3_config_dir.mkdir(parents=True, exist_ok=True)

    pt2_count = 0
    pt3_count = 0

    # PT2: Create configs for layers 3,4,6 (layer 5 already exists for seed1)
    # For seed2, create all layers 3,4,5,6
    print("Creating PT2 representation extraction configs...")

    for variant_num in PT2_TASKS.keys():
        task_name = PT2_TASKS[variant_num]

        for seed in SEEDS:
            for layer in LAYERS:
                # Skip layer 5 for seed1 (already exists)
                if seed == 1 and layer == 5:
                    continue

                config = create_repr_extraction_config('pt2', variant_num, seed, layer)
                config_path = pt2_config_dir / f'pt2-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{layer}.yaml'

                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  Created: {config_path}")
                pt2_count += 1

    # PT3: Create configs for all layers (none exist yet)
    print("\nCreating PT3 representation extraction configs...")

    for variant_num in PT3_TASKS.keys():
        task_name = PT3_TASKS[variant_num]

        for seed in SEEDS:
            for layer in LAYERS:
                config = create_repr_extraction_config('pt3', variant_num, seed, layer)
                config_path = pt3_config_dir / f'pt3-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{layer}.yaml'

                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  Created: {config_path}")
                pt3_count += 1

    print(f"\nTotal configs created:")
    print(f"  PT2: {pt2_count} configs (7 variants × 2 seeds × layers, skipping seed1 layer5)")
    print(f"    - Seed1: 7 × 3 layers (3,4,6) = 21 configs")
    print(f"    - Seed2: 7 × 4 layers (3,4,5,6) = 28 configs")
    print(f"  PT3: {pt3_count} configs (7 variants × 2 seeds × 4 layers)")
    print(f"    - Seed1: 7 × 4 = 28 configs")
    print(f"    - Seed2: 7 × 4 = 28 configs")
    print(f"\nTotal: {pt2_count + pt3_count} configs")

if __name__ == '__main__':
    main()
