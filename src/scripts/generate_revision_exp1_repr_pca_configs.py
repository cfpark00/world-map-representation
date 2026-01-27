#!/usr/bin/env python3
"""
Generate representation extraction and PCA timeline configs for revision/exp1.

Creates configs for:
- FTWB1: 21 models (original + seeds 1,2,3) × 7 single-task models
- FTWB2: 63 models (original + seeds 1,2,3) × 21 two-task models

For each model:
- 1 representation extraction config (layer 5)
- 3 PCA timeline configs (mixed, na, raw)
"""

from pathlib import Path
import yaml

# Base paths
CONFIG_BASE = Path("/configs")
REPR_CONFIG_DIR = CONFIG_BASE / "revision" / "exp1" / "representation_extraction"
PCA_CONFIG_DIR = CONFIG_BASE / "revision" / "exp1" / "pca_timeline"

# Experiment base path (for where the trained models are)
EXP_BASE = "/data/experiments"

# FTWB1 task mapping
FTWB1_TASKS = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

# Training data for FTWB2 experiments (2 tasks each)
TRAINING_DATA_2TASK = {
    1: ["distance", "trianglearea"],
    2: ["angle", "compass"],
    3: ["inside", "perimeter"],
    4: ["crossing", "distance"],
    5: ["trianglearea", "angle"],
    6: ["compass", "inside"],
    7: ["perimeter", "crossing"],
    8: ["angle", "distance"],
    9: ["compass", "trianglearea"],
    10: ["angle", "inside"],
    11: ["compass", "perimeter"],
    12: ["crossing", "inside"],
    13: ["distance", "perimeter"],
    14: ["crossing", "trianglearea"],
    15: ["compass", "distance"],
    16: ["inside", "trianglearea"],
    17: ["angle", "perimeter"],
    18: ["compass", "crossing"],
    19: ["distance", "inside"],
    20: ["perimeter", "trianglearea"],
    21: ["angle", "crossing"],
}

def create_repr_config(exp_name, exp_dir, task, layer=5):
    """Create representation extraction config."""
    config = {
        'cities_csv': 'data/datasets/cities/cities.csv',
        'device': 'cuda',
        'experiment_dir': exp_dir,
        'layers': [layer],
        'method': {'name': 'linear'},
        'n_test_cities': 1250,
        'n_train_cities': 3250,
        'output_dir': f"{exp_dir}/analysis_higher/{task}_firstcity_last_and_trans_l{layer}",
        'perform_pca': True,
        'probe_test': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'probe_train': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'prompt_format': f'{task}_firstcity_last_and_trans',
        'save_repr_ckpts': [-2],
        'seed': 42
    }
    return config

def create_pca_config(exp_dir, task, layer=5, pca_type='mixed'):
    """Create PCA timeline config.

    pca_type: 'mixed' (probe-aligned), 'na' (no atlantis probe), 'raw' (pure PCA)
    """
    # Set probe_train based on type
    if pca_type == 'na':
        probe_train = 'region:^(?!Atlantis$).* && city_id:^[1-9][0-9]{3,}$'
    else:
        probe_train = 'region:.* && city_id:^[1-9][0-9]{3,}$'

    # Set output dir suffix based on type
    if pca_type == 'mixed':
        output_suffix = 'pca_timeline'
    elif pca_type == 'na':
        output_suffix = 'pca_timeline_na'
    else:  # raw
        output_suffix = 'pca_timeline_raw'

    config = {
        'cities_csv': 'data/datasets/cities/cities.csv',
        'layer_index': -1,
        'marker_size': 3,
        'n_components': 3,
        'output_dir': f"{exp_dir}/analysis_higher/{task}_firstcity_last_and_trans_l{layer}/{output_suffix}",
        'probe_test': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'probe_train': probe_train,
        'representations_base_path': f"{exp_dir}/analysis_higher/{task}_firstcity_last_and_trans_l{layer}/representations",
        'token_index': -1,
        'train_frac': 0.6
    }

    if pca_type == 'mixed' or pca_type == 'na':
        # Mixed mode: use linear probe alignment
        config['axis_mapping'] = {
            'type': 'mixed',
            1: 'x',
            2: 'y',
            3: 'r0'
        }
    elif pca_type == 'raw':
        # Raw PCA mode: use standard PCA with integer indices
        config['axis_mapping'] = {
            'type': 'pca',
            1: 0,
            2: 1,
            3: 2
        }

    return config

def generate_ftwb1_configs():
    """Generate configs for FTWB1 models (28 models: original + 3 seeds × 7 tasks)."""
    print("Generating FTWB1 configs...")

    repr_count = 0
    pca_count = 0

    for seed_name in ['original', 'seed1', 'seed2', 'seed3']:
        for ftwb1_num in range(1, 8):
            task = FTWB1_TASKS[ftwb1_num]

            if seed_name == 'original':
                exp_name = f"pt1_ftwb1-{ftwb1_num}"
                exp_dir = f"{EXP_BASE}/{exp_name}"
            else:
                exp_name = f"pt1_{seed_name}_ftwb1-{ftwb1_num}"
                exp_dir = f"{EXP_BASE}/revision/exp1/{exp_name}"

            # Create representation extraction config
            repr_dir = REPR_CONFIG_DIR / exp_name
            repr_dir.mkdir(parents=True, exist_ok=True)

            repr_config = create_repr_config(exp_name, exp_dir, task)
            repr_file = repr_dir / f"{task}_firstcity_last_and_trans_l5.yaml"
            with open(repr_file, 'w') as f:
                yaml.dump(repr_config, f, default_flow_style=False, sort_keys=False)
            repr_count += 1

            # Create PCA timeline configs
            pca_dir = PCA_CONFIG_DIR / exp_name
            pca_dir.mkdir(parents=True, exist_ok=True)

            for pca_type in ['mixed', 'na', 'raw']:
                pca_config = create_pca_config(exp_dir, task, pca_type=pca_type)

                suffix = f"_l5_{pca_type}.yaml" if pca_type != 'mixed' else "_l5.yaml"
                pca_file = pca_dir / f"{task}_firstcity_last_and_trans{suffix}"

                with open(pca_file, 'w') as f:
                    yaml.dump(pca_config, f, default_flow_style=False, sort_keys=False)
                pca_count += 1

    print(f"  Created {repr_count} representation extraction configs")
    print(f"  Created {pca_count} PCA timeline configs")
    return repr_count, pca_count

def generate_ftwb2_configs():
    """Generate configs for FTWB2 models (63 models: 3 seeds × 21 two-task models)."""
    print("\nGenerating FTWB2 configs...")

    repr_count = 0
    pca_count = 0

    for seed_name in ['seed1', 'seed2', 'seed3']:
        for ftwb2_num in range(1, 22):
            trained_tasks = TRAINING_DATA_2TASK[ftwb2_num]
            task = trained_tasks[0]  # Use first task for prompt format

            exp_name = f"pt1_{seed_name}_ftwb2-{ftwb2_num}"
            exp_dir = f"{EXP_BASE}/revision/exp1/{exp_name}"

            # Create representation extraction config
            repr_dir = REPR_CONFIG_DIR / exp_name
            repr_dir.mkdir(parents=True, exist_ok=True)

            repr_config = create_repr_config(exp_name, exp_dir, task)
            repr_file = repr_dir / f"{task}_firstcity_last_and_trans_l5.yaml"
            with open(repr_file, 'w') as f:
                yaml.dump(repr_config, f, default_flow_style=False, sort_keys=False)
            repr_count += 1

            # Create PCA timeline configs
            pca_dir = PCA_CONFIG_DIR / exp_name
            pca_dir.mkdir(parents=True, exist_ok=True)

            for pca_type in ['mixed', 'na', 'raw']:
                pca_config = create_pca_config(exp_dir, task, pca_type=pca_type)

                suffix = f"_l5_{pca_type}.yaml" if pca_type != 'mixed' else "_l5.yaml"
                pca_file = pca_dir / f"{task}_firstcity_last_and_trans{suffix}"

                with open(pca_file, 'w') as f:
                    yaml.dump(pca_config, f, default_flow_style=False, sort_keys=False)
                pca_count += 1

    print(f"  Created {repr_count} representation extraction configs")
    print(f"  Created {pca_count} PCA timeline configs")
    return repr_count, pca_count

def main():
    print("="*60)
    print("Generating Revision Exp1 Representation & PCA Configs")
    print("="*60)

    # Generate configs
    ftwb1_repr, ftwb1_pca = generate_ftwb1_configs()
    ftwb2_repr, ftwb2_pca = generate_ftwb2_configs()

    total_repr = ftwb1_repr + ftwb2_repr
    total_pca = ftwb1_pca + ftwb2_pca

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total representation extraction configs: {total_repr}")
    print(f"Total PCA timeline configs: {total_pca}")
    print(f"Total configs: {total_repr + total_pca}")
    print(f"\nConfigs saved to:")
    print(f"  - {REPR_CONFIG_DIR}")
    print(f"  - {PCA_CONFIG_DIR}")

if __name__ == "__main__":
    main()
