#!/usr/bin/env python3
"""
Compute CKA matrices for all layers and prefixes.
Uses existing representation extraction scripts and computes CKA directly.
"""

import subprocess
import sys
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from tqdm import tqdm

# Add to path
sys.path.insert(0, '')
from src.utils import filter_dataframe_by_pattern


def create_repr_config(prefix, exp_num, layer, task):
    """Create representation extraction config FOR THE SPECIFIC LAYER."""
    prompt_format = f"{task}_firstcity_last_and_trans"

    config = {
        'cities_csv': 'data/datasets/cities/cities.csv',
        'device': 'cuda',
        'experiment_dir': f'data/experiments/{prefix}-{exp_num}',
        'layers': [layer],  # THIS IS THE KEY - using the actual layer number passed in!
        'method': {'name': 'linear'},
        'n_test_cities': 1250,
        'n_train_cities': 3250,
        'output_dir': f'/data/experiments/{prefix}-{exp_num}/analysis_higher/{task}_firstcity_last_and_trans_l{layer}',  # Output dir includes layer number
        'perform_pca': True,  # Need to set to True for the script to work
        'probe_test': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'probe_train': 'region:.* && city_id:^[1-9][0-9]{3,}$',
        'prompt_format': prompt_format,
        'save_repr_ckpts': [-2],  # Only save final checkpoint
        'seed': 42
    }

    print(f"      Creating config for layer {layer}: layers=[{layer}], output={config['output_dir']}")  # DEBUG

    return config


def get_task_mapping(prefix):
    """Get task mapping for each experiment by looking at existing l5 configs."""
    task_map = {}

    # Look for existing l5 configs and use them to determine task names
    config_dir = Path(f'/configs/analysis_representation_higher/ftset')

    for i in range(1, 9):
        exp_dir = config_dir / f'{prefix}-{i}'
        if exp_dir.exists():
            # Find the l5 config file to get the task name
            configs = list(exp_dir.glob('*_firstcity_last_and_trans_l5.yaml'))
            if configs:
                # Extract task name from filename
                filename = configs[0].stem  # e.g., "distance_firstcity_last_and_trans_l5"
                task = filename.replace('_firstcity_last_and_trans_l5', '')
                task_map[i] = task
                print(f"    Found {prefix}-{i}: task = {task}")  # DEBUG

    if not task_map:
        # Fallback: hardcode the mappings we know
        if prefix == 'pt1':
            task_map = {1: 'distance', 2: 'trianglearea', 3: 'angle', 4: 'compass', 5: 'inside', 6: 'perimeter', 7: 'crossing'}
        elif prefix == 'pt2':
            task_map = {1: 'distance', 2: 'angle', 3: 'inside', 4: 'crossing', 5: 'trianglearea', 6: 'compass', 7: 'perimeter', 8: 'distance'}
        elif prefix == 'pt3':
            task_map = {1: 'distance', 2: 'compass', 3: 'crossing', 4: 'angle', 5: 'perimeter', 6: 'trianglearea', 7: 'inside', 8: 'distance'}
        print(f"    Using hardcoded mapping for {prefix}: {task_map}")

    return task_map


def extract_representations_if_needed(prefix, layer):
    """Extract representations for all models in prefix at given layer."""
    task_mapping = get_task_mapping(prefix)

    print(f"  Task mapping for {prefix}: {task_mapping}")  # DEBUG

    for exp_num, task in task_mapping.items():
        # Check if model checkpoint exists first
        checkpoint_dir = Path(f'/data/experiments/{prefix}-{exp_num}/checkpoints')
        if not checkpoint_dir.exists():
            print(f"  {prefix}-{exp_num}: No model checkpoint directory found, skipping...")
            continue

        # Check if any checkpoints exist
        checkpoints = list(checkpoint_dir.glob('checkpoint-*'))
        if not checkpoints:
            print(f"  {prefix}-{exp_num}: No checkpoints found, skipping...")
            continue

        # Check if representations already exist
        repr_dir = Path(f'/data/experiments/{prefix}-{exp_num}/analysis_higher/{task}_firstcity_last_and_trans_l{layer}/representations')

        if repr_dir.exists() and len(list(repr_dir.glob('checkpoint-*'))) > 0:
            print(f"  {prefix}-{exp_num} layer {layer} ({task}): representations already exist")
        else:
            print(f"  {prefix}-{exp_num} layer {layer} ({task}): EXTRACTING REPRESENTATIONS NOW...")

            # Create config
            config = create_repr_config(prefix, exp_num, layer, task)
            config_path = Path(f'/tmp/temp_repr_config_{prefix}_{exp_num}_l{layer}.yaml')

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Run extraction
            cmd = [
                'uv', 'run', 'python',
                'src/analysis/analyze_representations_higher.py',
                str(config_path),
                '--overwrite'
            ]

            print(f"    Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='')

            if result.returncode != 0:
                print(f"    ERROR: Failed to extract representations")
                print(f"    STDERR: {result.stderr[:500]}")  # Show first 500 chars of error
            else:
                print(f"    SUCCESS: Representations extracted!")
                # Verify they were created
                if repr_dir.exists() and len(list(repr_dir.glob('checkpoint-*'))) > 0:
                    print(f"    VERIFIED: Representations saved to {repr_dir}")


def load_representations(repr_path, city_filter='region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'):
    """Load representations from checkpoint directory."""
    import pandas as pd

    # Find final checkpoint
    checkpoints = sorted(repr_path.glob('checkpoint-*'),
                        key=lambda x: int(x.name.split('-')[1]))
    if not checkpoints:
        return None, None

    final_checkpoint = checkpoints[-1]

    # Load representations
    repr_file = final_checkpoint / 'representations.pt'
    metadata_file = final_checkpoint / 'metadata.json'

    repr_data = torch.load(repr_file, map_location='cpu')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Extract representations
    representations = repr_data['representations']
    if isinstance(representations, torch.Tensor):
        representations = representations.numpy()

    # Reshape to (n_cities, -1) - concatenate all tokens and layers
    n_cities = representations.shape[0]
    representations = representations.reshape(n_cities, -1)

    # Filter cities if needed
    if city_filter:
        city_info = metadata.get('city_info', [])
        city_df = pd.DataFrame(city_info)
        if 'row_id' in city_df.columns and 'city_id' not in city_df.columns:
            city_df['city_id'] = city_df['row_id']

        filtered_df = filter_dataframe_by_pattern(city_df, city_filter, column_name='region')
        filtered_indices = filtered_df.index.tolist()

        representations = representations[filtered_indices]
        city_ids = filtered_df['city_id'].values
    else:
        city_ids = [c['row_id'] for c in metadata.get('city_info', [])]

    return representations, city_ids


def compute_cka(K1, K2):
    """Compute CKA between two kernel matrices."""
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1_centered = H @ K1 @ H
    K2_centered = H @ K2 @ H

    hsic_12 = np.sum(K1_centered * K2_centered) / (n ** 2)
    hsic_11 = np.sum(K1_centered * K1_centered) / (n ** 2)
    hsic_22 = np.sum(K2_centered * K2_centered) / (n ** 2)

    cka = hsic_12 / np.sqrt(hsic_11 * hsic_22)
    return cka


def compute_cka_matrix(prefix, layer):
    """Compute and plot CKA matrix for given prefix and layer."""
    task_mapping = get_task_mapping(prefix)
    n_models = len(task_mapping)

    # FIRST: Extract representations if needed!!!
    print(f"\nChecking/extracting representations for {prefix} layer {layer}...")
    extract_representations_if_needed(prefix, layer)

    print(f"\nComputing CKA matrix for {prefix} layer {layer}...")

    # Load all representations
    representations = {}
    for exp_num, task in task_mapping.items():
        repr_path = Path(f'/data/experiments/{prefix}-{exp_num}/analysis_higher/{task}_firstcity_last_and_trans_l{layer}/representations')

        repr, cities = load_representations(repr_path)
        if repr is not None:
            representations[exp_num] = repr
            print(f"  Loaded {prefix}-{exp_num}: shape {repr.shape}")

    # Compute CKA matrix
    cka_matrix = np.ones((n_models, n_models))

    # Create directory for saving individual CKA results
    cka_results_dir = Path(f'/data/experiments/cka_analysis')
    cka_results_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, n_models + 1):
        for j in range(i + 1, n_models + 1):
            if i in representations and j in representations:
                X1 = representations[i]
                X2 = representations[j]

                # Compute kernel matrices
                K1 = X1 @ X1.T
                K2 = X2 @ X2.T

                # Compute CKA
                cka_val = compute_cka(K1, K2)
                cka_matrix[i-1, j-1] = cka_val
                cka_matrix[j-1, i-1] = cka_val
                print(f"    {prefix}-{i} vs {prefix}-{j}: {cka_val:.4f}")

                # SAVE THE CKA VALUE TO JSON!
                pair_dir = cka_results_dir / f'{prefix}-{i}_vs_{prefix}-{j}_l{layer}'
                pair_dir.mkdir(parents=True, exist_ok=True)
                cka_result = {
                    'exp1': f'{prefix}-{i}',
                    'exp2': f'{prefix}-{j}',
                    'layer': layer,
                    'final_cka': float(cka_val),
                    'task1': task_mapping[i],
                    'task2': task_mapping[j]
                }
                with open(pair_dir / 'cka_results.json', 'w') as f:
                    json.dump(cka_result, f, indent=2)
                print(f"      Saved to: {pair_dir}/cka_results.json")

    # Plot
    models = [f'{prefix}-{i}' for i in range(1, n_models + 1)]
    tasks = [task_mapping[i] for i in range(1, n_models + 1)]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cka_matrix,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'CKA'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray')

    ax.set_xticklabels([f'{m}\n({t})' for m, t in zip(models, tasks)], rotation=45, ha='right')
    ax.set_yticklabels([f'{m}\n({t})' for m, t in zip(models, tasks)], rotation=0)

    ax.set_title(f'CKA Matrix ({prefix.upper()}): Layer {layer}', fontsize=14, pad=20)

    plt.tight_layout()

    # Save
    output_dir = Path(f'/scratch/cka_analysis_{prefix}_l{layer}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'cka_matrix_{prefix}_l{layer}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved to: {output_path}")

    plt.close()

    # Print statistics
    off_diagonal = cka_matrix[np.triu_indices(n_models, k=1)]
    valid_values = off_diagonal[~np.isnan(off_diagonal)]

    print(f"\n  {prefix.upper()} Layer {layer} Summary:")
    if len(valid_values) > 0:
        print(f"    Mean CKA: {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}")
        print(f"    Min: {np.min(valid_values):.4f}, Max: {np.max(valid_values):.4f}")
        print(f"    Valid pairs: {len(valid_values)}/{len(off_diagonal)}")
    else:
        print(f"    No valid CKA values computed (missing representations)")


def main():
    prefixes = ['pt1', 'pt2', 'pt3']
    layers = [3, 4, 5, 6]

    print("=" * 60)
    print("Computing CKA matrices for all experiments and layers")
    print("=" * 60)

    for prefix in prefixes:
        for layer in layers:
            print(f"\n{'='*60}")
            print(f"Processing {prefix} Layer {layer}")
            print(f"{'='*60}")

            # Compute and plot CKA matrix (this will extract representations if needed)
            compute_cka_matrix(prefix, layer)

    print("\n" + "=" * 60)
    print("All CKA matrices computed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()