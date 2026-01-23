#!/usr/bin/env python3
"""
Generate CKA configs for PT2/PT3 seed variants (exp2).
Creates configs ONLY for non-overlapping task pairs across all seeds.
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
    'pt2-8': {'distance', 'trianglearea'},  # Same as pt2-1
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
    'pt3-8': {'distance', 'trianglearea', 'compass'},
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
    'pt2-8': 'distance',  # Same as pt2-1
}

PT3_REPR_TASKS = {
    'pt3-1': 'distance',
    'pt3-2': 'compass',
    'pt3-3': 'crossing',
    'pt3-4': 'angle',
    'pt3-5': 'perimeter',
    'pt3-6': 'trianglearea',
    'pt3-7': 'inside',
    'pt3-8': 'distance',  # Same as pt3-1
}


def shares_tasks(tasks1, tasks2):
    """Check if two task sets have any overlap."""
    return len(set(tasks1) & set(tasks2)) > 0


def get_non_overlapping_pairs(task_dict, max_variant=8):
    """Get all non-overlapping pairs for a given task dictionary."""
    pairs = []
    for i in range(1, max_variant + 1):
        for j in range(i + 1, max_variant + 1):
            name1 = f'pt{len(list(task_dict.values())[0])}-{i}'  # pt2 or pt3 based on task set size
            name2 = f'pt{len(list(task_dict.values())[0])}-{j}'

            # Skip if either variant doesn't exist in dict
            if name1 not in task_dict or name2 not in task_dict:
                continue

            if not shares_tasks(task_dict[name1], task_dict[name2]):
                pairs.append((i, j))

    return pairs


def get_model_path(prefix, variant, seed, base_path):
    """Get the path to a model's representations."""
    if seed == 'orig':
        # Original models are in main experiments folder
        return base_path / 'data' / 'experiments' / f'{prefix}-{variant}'
    else:
        # Seed models are in revision/exp2
        return base_path / 'data' / 'experiments' / 'revision' / 'exp2' / f'{prefix}-{variant}_seed{seed}'


def generate_cka_config(prefix, var1, var2, seed1, seed2, layer, task_dict, repr_task_dict, base_path, config_dir):
    """Generate a single CKA config file."""

    # Get experiment names
    exp1_name = f'{prefix}-{var1}' if seed1 == 'orig' else f'{prefix}-{var1}_seed{seed1}'
    exp2_name = f'{prefix}-{var2}' if seed2 == 'orig' else f'{prefix}-{var2}_seed{seed2}'

    # Get tasks
    task1 = repr_task_dict[f'{prefix}-{var1}']
    task2 = repr_task_dict[f'{prefix}-{var2}']

    # Get representation paths
    model1_path = get_model_path(prefix, var1, seed1, base_path)
    model2_path = get_model_path(prefix, var2, seed2, base_path)

    repr1_dir = model1_path / 'analysis_higher' / f'{task1}_firstcity_last_and_trans_l{layer}' / 'representations'
    repr2_dir = model2_path / 'analysis_higher' / f'{task2}_firstcity_last_and_trans_l{layer}' / 'representations'

    # Output directory
    output_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis' / f'{exp1_name}_vs_{exp2_name}' / f'layer{layer}'

    # Create config
    config = {
        'exp1': {
            'name': exp1_name,
            'repr_dir': str(repr1_dir),
            'task': task1,
        },
        'exp2': {
            'name': exp2_name,
            'repr_dir': str(repr2_dir),
            'task': task2,
        },
        'layer': layer,
        'checkpoint_steps': None,  # null = auto-detect
        'use_final_only': True,  # Only compute CKA for final checkpoint
        'city_filter': 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$',
        'kernel_type': 'linear',
        'center_kernels': True,
        'use_gpu': True,
        'save_timeline_plot': False,
        'output_dir': str(output_dir),
    }

    # Create config directory
    config_subdir = config_dir / f'{prefix}_seed_cka' / f'{prefix}-{var1}_vs_{prefix}-{var2}'
    config_subdir.mkdir(parents=True, exist_ok=True)

    # Config filename
    config_file = config_subdir / f'layer{layer}_{seed1}_vs_{seed2}.yaml'

    # Write config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_file


def main():
    base_path = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')
    config_dir = base_path / 'configs' / 'revision' / 'exp2'

    layers = [3, 4, 5, 6]
    seeds = ['orig', '1', '2']

    print("=" * 80)
    print("Generating CKA configs for PT2/PT3 seed variants (non-overlapping pairs only)")
    print("=" * 80)

    # PT2
    print("\n" + "=" * 80)
    print("PT2 (Two-task)")
    print("=" * 80)

    pt2_pairs = get_non_overlapping_pairs(PT2_TASKS, max_variant=7)  # Only 1-7 have seeds
    print(f"\nNon-overlapping PT2 pairs: {len(pt2_pairs)}")
    for pair in pt2_pairs:
        print(f"  pt2-{pair[0]} vs pt2-{pair[1]}: {PT2_TASKS[f'pt2-{pair[0]}']} vs {PT2_TASKS[f'pt2-{pair[1]}']}")

    pt2_configs_created = 0
    for var1, var2 in pt2_pairs:
        for layer in layers:
            for i, seed1 in enumerate(seeds):
                for seed2 in seeds[i+1:]:  # Only unique pairs, no duplicates!
                    config_file = generate_cka_config(
                        'pt2', var1, var2, seed1, seed2, layer,
                        PT2_TASKS, PT2_REPR_TASKS, base_path, config_dir
                    )
                    pt2_configs_created += 1

    # Calculate unique seed pairs: C(3,2) = 3
    n_unique_seed_pairs = len(seeds) * (len(seeds) - 1) // 2
    print(f"\nPT2 configs created: {pt2_configs_created}")
    print(f"  = {len(pt2_pairs)} pairs × {len(layers)} layers × {n_unique_seed_pairs} seed combos")

    # PT3
    print("\n" + "=" * 80)
    print("PT3 (Three-task)")
    print("=" * 80)

    pt3_pairs = get_non_overlapping_pairs(PT3_TASKS, max_variant=7)  # Only 1-7 have seeds
    print(f"\nNon-overlapping PT3 pairs: {len(pt3_pairs)}")
    for pair in pt3_pairs:
        print(f"  pt3-{pair[0]} vs pt3-{pair[1]}: {PT3_TASKS[f'pt3-{pair[0]}']} vs {PT3_TASKS[f'pt3-{pair[1]}']}")

    pt3_configs_created = 0
    for var1, var2 in pt3_pairs:
        for layer in layers:
            for i, seed1 in enumerate(seeds):
                for seed2 in seeds[i+1:]:  # Only unique pairs, no duplicates!
                    config_file = generate_cka_config(
                        'pt3', var1, var2, seed1, seed2, layer,
                        PT3_TASKS, PT3_REPR_TASKS, base_path, config_dir
                    )
                    pt3_configs_created += 1

    print(f"\nPT3 configs created: {pt3_configs_created}")
    print(f"  = {len(pt3_pairs)} pairs × {len(layers)} layers × {n_unique_seed_pairs} seed combos")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total configs created: {pt2_configs_created + pt3_configs_created}")
    print(f"  PT2: {pt2_configs_created} configs ({len(pt2_pairs)} pairs)")
    print(f"  PT3: {pt3_configs_created} configs ({len(pt3_pairs)} pairs)")
    print(f"\nConfig location: {config_dir}/{{pt2,pt3}}_seed_cka/")
    print(f"\nSeed pairs (unique only, CKA is symmetric):")
    for i, seed1 in enumerate(seeds):
        for seed2 in seeds[i+1:]:
            print(f"  {seed1} vs {seed2}")
    print("\nNote: Only non-overlapping task pairs included (meaningful comparisons only)")
    print("Note: Only unique seed pairs included (no redundant calculations)")


if __name__ == '__main__':
    main()
