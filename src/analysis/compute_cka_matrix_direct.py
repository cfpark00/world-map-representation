#!/usr/bin/env python3
"""
Direct CKA matrix computation and visualization.
Extracts representations, computes CKA, and plots matrix all in one go.
Avoids saving intermediate files to disk.

Usage:
    python compute_cka_matrix_direct.py --prefix pt2 --layer 4
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
import re
from tqdm import tqdm
import sys

# Add parent directory to path
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

# Model loading will be done directly in the extract_representations function


def get_task_mapping(prefix):
    """Get task mapping for each experiment."""
    task_map = {}
    for i in range(1, 9):
        config_path = Path(f'/n/home12/cfpark00/WM_1/configs/data_generation/ftset/combine_{prefix}-{i}.yaml')
        if not config_path.exists():
            break
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            first_dataset = config['datasets'][0]['path']
            task = re.search(r'/([^/]+)_1M', first_dataset).group(1)
            task_map[i] = task
    return task_map


def get_prompt_format(task_name):
    """Get prompt format for a task."""
    return f"{task_name}_firstcity_last_and_trans"


def extract_representations(model_path, prompt_format, layer_index=4,
                           cities_csv_path='/n/home12/cfpark00/WM_1/data/datasets/cities/cities.csv',
                           city_filter='region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'):
    """
    Extract representations from a model checkpoint.
    Returns representations for the specified layer.
    """
    import pandas as pd
    from src.utils import filter_dataframe_by_pattern
    from src.data.prompt_utils import get_prompt_config

    # Load cities
    cities_df = pd.read_csv(cities_csv_path)

    # Apply filter
    filtered_df = filter_dataframe_by_pattern(cities_df, city_filter, column_name='region')
    city_ids = filtered_df['city_id'].values

    # Get prompt configuration
    prompt_config = get_prompt_config(prompt_format)
    token_indices = prompt_config['token_indices']

    # Load model
    model = load_model_from_checkpoint(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    all_representations = []

    with torch.no_grad():
        for _, city_row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Extracting representations"):
            # Create prompt
            prompt = prompt_config['format'].format(
                x1=city_row['x'], y1=city_row['y'],
                x2=city_row.get('x2', 0), y2=city_row.get('y2', 0),
                x3=city_row.get('x3', 0), y3=city_row.get('y3', 0)
            )

            # Tokenize
            inputs = model.tokenizer(prompt, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # (n_layers, batch=1, seq_len, hidden_dim)

            # Extract specified layer and tokens
            layer_repr = hidden_states[layer_index][0]  # (seq_len, hidden_dim)

            # Get tokens at specified indices
            selected_tokens = []
            for idx in token_indices:
                if idx < layer_repr.shape[0]:
                    selected_tokens.append(layer_repr[idx])

            if selected_tokens:
                # Concatenate selected tokens
                repr_vector = torch.cat(selected_tokens, dim=-1)
                all_representations.append(repr_vector.cpu().numpy())

    return np.array(all_representations), city_ids


def compute_cka_gpu(K1, K2):
    """Compute CKA between two kernel matrices on GPU."""
    # Convert to torch tensors on GPU
    if isinstance(K1, np.ndarray):
        K1 = torch.from_numpy(K1).float().cuda()
    if isinstance(K2, np.ndarray):
        K2 = torch.from_numpy(K2).float().cuda()

    n = K1.shape[0]

    # Center the kernels
    ones = torch.ones((n, n), device='cuda') / n
    H = torch.eye(n, device='cuda') - ones

    K1_centered = torch.mm(torch.mm(H, K1), H)
    K2_centered = torch.mm(torch.mm(H, K2), H)

    # Compute HSIC
    hsic_12 = torch.sum(K1_centered * K2_centered) / (n ** 2)
    hsic_11 = torch.sum(K1_centered * K1_centered) / (n ** 2)
    hsic_22 = torch.sum(K2_centered * K2_centered) / (n ** 2)

    # CKA
    cka = hsic_12 / torch.sqrt(hsic_11 * hsic_22)

    return cka.item()


def compute_cka_cpu(K1, K2):
    """Compute CKA between two kernel matrices on CPU."""
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1_centered = H @ K1 @ H
    K2_centered = H @ K2 @ H

    hsic_12 = np.sum(K1_centered * K2_centered) / (n ** 2)
    hsic_11 = np.sum(K1_centered * K1_centered) / (n ** 2)
    hsic_22 = np.sum(K2_centered * K2_centered) / (n ** 2)

    cka = hsic_12 / np.sqrt(hsic_11 * hsic_22)

    return cka


def get_non_overlapping_pairs(prefix):
    """Get pairs that don't share training data."""
    dataset_mappings = {
        'pt2': {
            1: {'distance', 'trianglearea'},
            2: {'angle', 'compass'},
            3: {'inside', 'perimeter'},
            4: {'crossing', 'distance'},
            5: {'trianglearea', 'angle'},
            6: {'compass', 'inside'},
            7: {'perimeter', 'crossing'},
            8: {'distance', 'angle'}
        },
        'pt3': {
            1: {'distance', 'trianglearea', 'angle'},
            2: {'compass', 'inside', 'perimeter'},
            3: {'crossing', 'distance', 'trianglearea'},
            4: {'angle', 'compass', 'inside'},
            5: {'perimeter', 'crossing', 'distance'},
            6: {'trianglearea', 'angle', 'compass'},
            7: {'inside', 'perimeter', 'crossing'},
            8: {'distance', 'trianglearea', 'angle'}
        }
    }

    if prefix not in dataset_mappings:
        return set()

    datasets = dataset_mappings[prefix]
    non_overlap_pairs = set()

    for i in range(1, len(datasets) + 1):
        for j in range(i + 1, len(datasets) + 1):
            shared = datasets[i] & datasets[j]
            if not shared:
                non_overlap_pairs.add((i, j))

    return non_overlap_pairs


def main():
    parser = argparse.ArgumentParser(description='Direct CKA matrix computation')
    parser.add_argument('--prefix', type=str, required=True, choices=['pt1', 'pt2', 'pt3'],
                       help='Experiment prefix')
    parser.add_argument('--layer', type=int, default=4,
                       help='Layer index to analyze (default: 4)')
    parser.add_argument('--checkpoint', type=str, default='final',
                       help='Checkpoint to use (default: final)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available() and not args.no_gpu
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Get task mapping
    task_mapping = get_task_mapping(args.prefix)
    n_models = len(task_mapping)

    if n_models == 0:
        print(f"No models found for {args.prefix}")
        return

    models = [f'{args.prefix}-{i}' for i in range(1, n_models + 1)]
    tasks = [task_mapping[i] for i in range(1, n_models + 1)]

    print(f"\n{'='*60}")
    print(f"Computing CKA matrix for {args.prefix} - Layer {args.layer}")
    print(f"{'='*60}")
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")
    print()

    # Extract representations for all models
    representations = {}
    city_ids_dict = {}

    for i in range(1, n_models + 1):
        model_name = f'{args.prefix}-{i}'
        task = task_mapping[i]
        prompt_format = get_prompt_format(task)

        # Find checkpoint path
        if args.checkpoint == 'final':
            checkpoint_dir = Path(f'/n/home12/cfpark00/WM_1/data/experiments/{model_name}/checkpoints')
            # Find the highest numbered checkpoint
            checkpoints = sorted(checkpoint_dir.glob('checkpoint-*'),
                               key=lambda x: int(x.name.split('-')[1]))
            if not checkpoints:
                print(f"No checkpoints found for {model_name}")
                continue
            model_path = checkpoints[-1]
        else:
            model_path = Path(f'/n/home12/cfpark00/WM_1/data/experiments/{model_name}/checkpoints/checkpoint-{args.checkpoint}')

        if not model_path.exists():
            print(f"Checkpoint not found: {model_path}")
            continue

        print(f"Extracting representations for {model_name} ({task})...")
        representations[i], city_ids = extract_representations(
            model_path, prompt_format, layer_index=args.layer
        )
        city_ids_dict[i] = city_ids
        print(f"  Shape: {representations[i].shape}")

    # Verify all models have same cities
    base_cities = set(city_ids_dict[1])
    for i in range(2, n_models + 1):
        if i in city_ids_dict:
            if set(city_ids_dict[i]) != base_cities:
                print(f"Warning: City mismatch for model {i}")

    # Compute CKA matrix
    print("\nComputing CKA matrix...")
    cka_matrix = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(i+1, n_models):
            if (i+1) not in representations or (j+1) not in representations:
                cka_matrix[i, j] = np.nan
                cka_matrix[j, i] = np.nan
                continue

            # Compute kernel matrices
            X1 = representations[i+1]
            X2 = representations[j+1]

            if use_gpu:
                X1_gpu = torch.from_numpy(X1).float().cuda()
                X2_gpu = torch.from_numpy(X2).float().cuda()
                K1 = torch.mm(X1_gpu, X1_gpu.t()).cpu().numpy()
                K2 = torch.mm(X2_gpu, X2_gpu.t()).cpu().numpy()
                cka = compute_cka_gpu(K1, K2)
            else:
                K1 = X1 @ X1.T
                K2 = X2 @ X2.T
                cka = compute_cka_cpu(K1, K2)

            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka
            print(f"  {models[i]} vs {models[j]}: CKA = {cka:.4f}")

    # Plot matrix
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(11 if n_models == 8 else 10, 9 if n_models == 8 else 8))

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

    title = f'CKA Matrix ({args.prefix.upper()}): Layer {args.layer}\n'
    title += f'Checkpoint: {args.checkpoint}'
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    # Save plot
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f'/n/home12/cfpark00/WM_1/scratch/cka_analysis_{args.prefix}_l{args.layer}')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'cka_matrix_{args.prefix}_l{args.layer}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print statistics
    print(f"\n{args.prefix.upper()} CKA Matrix Summary (Layer {args.layer}):")
    off_diagonal = cka_matrix[np.triu_indices(n_models, k=1)]
    valid_values = off_diagonal[~np.isnan(off_diagonal)]

    if len(valid_values) > 0:
        print(f"  Mean CKA (off-diagonal): {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}")
        print(f"  Min: {np.min(valid_values):.4f}, Max: {np.max(valid_values):.4f}")

        # Special statistic for non-overlapping pairs
        if args.prefix in ['pt2', 'pt3']:
            non_overlap_pairs = get_non_overlapping_pairs(args.prefix)
            non_overlap_values = []
            for i in range(n_models):
                for j in range(i+1, n_models):
                    if (i+1, j+1) in non_overlap_pairs and not np.isnan(cka_matrix[i, j]):
                        non_overlap_values.append(cka_matrix[i, j])

            if non_overlap_values:
                print(f"\n  SPECIAL: Mean CKA for non-overlapping training sets: {np.mean(non_overlap_values):.4f} ± {np.std(non_overlap_values):.4f}")
                print(f"           ({len(non_overlap_values)} pairs with no shared training data)")

    plt.show()


if __name__ == "__main__":
    main()