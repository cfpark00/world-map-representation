#!/usr/bin/env python3
"""
Compute CKA (Centered Kernel Alignment) between two sets of representations.

This script loads representations from two models/experiments and computes CKA
for each matching checkpoint, saving the results for further analysis.

Usage:
    python compute_cka_from_representations.py configs/analysis_cka/config.yaml
"""

import argparse
import yaml
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import re
from tqdm import tqdm

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory, filter_dataframe_by_pattern


def load_checkpoint_representations(checkpoint_path, token_index=-1, layer_index=-1):
    """
    Load representations from a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        token_index: Which token to use (-1 for all)
        layer_index: Which layer to use (-1 for all)

    Returns:
        Tuple of (representations, metadata, city_ids)
    """
    repr_path = checkpoint_path / 'representations.pt'
    metadata_path = checkpoint_path / 'metadata.json'

    if not repr_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing files in {checkpoint_path}")

    # Load data
    repr_data = torch.load(repr_path, map_location='cpu')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Extract representations
    representations = repr_data['representations']
    if isinstance(representations, torch.Tensor):
        representations = representations.numpy()

    # Shape: (n_cities, n_tokens, n_layers, hidden_dim)
    n_cities, n_tokens, n_layers, hidden_dim = representations.shape

    # Extract specified tokens and layers
    if token_index == -1 and layer_index == -1:
        # Concatenate all tokens and layers
        representations = representations.reshape(n_cities, -1)
    elif token_index == -1:
        # All tokens, specific layer
        representations = representations[:, :, layer_index, :].reshape(n_cities, -1)
    elif layer_index == -1:
        # Specific token, all layers
        representations = representations[:, token_index, :, :].reshape(n_cities, -1)
    else:
        # Specific token and layer
        representations = representations[:, token_index, layer_index, :]

    # Get city IDs
    city_info = metadata.get('city_info', [])
    city_ids = [c['row_id'] for c in city_info]

    return representations, metadata, city_ids


def compute_kernel_matrix_gpu(X):
    """
    Compute linear kernel matrix on GPU.

    Args:
        X: Representations matrix (n_samples, n_features)

    Returns:
        Kernel matrix (n_samples, n_samples) as numpy array
    """
    # Convert to torch tensor and move to GPU
    X_torch = torch.from_numpy(X).float().cuda()

    # Linear kernel: K = X @ X.T
    K = torch.mm(X_torch, X_torch.t())

    # Convert back to numpy
    return K.cpu().numpy()


def center_kernel_matrix_gpu(K):
    """
    Center a kernel matrix on GPU.

    Args:
        K: Kernel matrix (n, n)

    Returns:
        Centered kernel matrix as numpy array
    """
    # Convert to torch tensor and move to GPU
    K_torch = torch.from_numpy(K).float().cuda()
    n = K_torch.shape[0]

    # Create centering matrix on GPU
    ones = torch.ones((n, n), device='cuda') / n
    H = torch.eye(n, device='cuda') - ones

    # Center the kernel: H @ K @ H
    K_centered = torch.mm(torch.mm(H, K_torch), H)

    # Convert back to numpy
    return K_centered.cpu().numpy()


def compute_cka_gpu(K1, K2, already_centered=False):
    """
    Compute CKA between two kernel matrices on GPU.

    Args:
        K1: First kernel matrix (n x n)
        K2: Second kernel matrix (n x n)
        already_centered: Whether kernels are already centered

    Returns:
        CKA value between 0 and 1
    """
    # Convert to torch tensors on GPU
    K1_torch = torch.from_numpy(K1).float().cuda()
    K2_torch = torch.from_numpy(K2).float().cuda()
    n = K1_torch.shape[0]

    if not already_centered:
        # Center the kernels
        ones = torch.ones((n, n), device='cuda') / n
        H = torch.eye(n, device='cuda') - ones
        K1_centered = torch.mm(torch.mm(H, K1_torch), H)
        K2_centered = torch.mm(torch.mm(H, K2_torch), H)
    else:
        K1_centered = K1_torch
        K2_centered = K2_torch

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    # HSIC(K1, K2) = (1/n^2) * trace(K1 @ K2)
    # For trace(A @ B), we can use element-wise multiplication and sum
    hsic_12 = torch.sum(K1_centered * K2_centered) / (n ** 2)
    hsic_11 = torch.sum(K1_centered * K1_centered) / (n ** 2)
    hsic_22 = torch.sum(K2_centered * K2_centered) / (n ** 2)

    # CKA = HSIC(K1, K2) / sqrt(HSIC(K1, K1) * HSIC(K2, K2))
    cka = hsic_12 / torch.sqrt(hsic_11 * hsic_22)

    return cka.item()


def compute_cka_cpu(K1, K2, already_centered=False):
    """
    Compute CKA between two kernel matrices on CPU.

    Args:
        K1: First kernel matrix (n x n)
        K2: Second kernel matrix (n x n)
        already_centered: Whether kernels are already centered

    Returns:
        CKA value between 0 and 1
    """
    n = K1.shape[0]

    if not already_centered:
        # Center the kernels
        H = np.eye(n) - np.ones((n, n)) / n
        K1_centered = H @ K1 @ H
        K2_centered = H @ K2 @ H
    else:
        K1_centered = K1
        K2_centered = K2

    # Compute HSIC
    hsic_12 = np.sum(K1_centered * K2_centered) / (n ** 2)
    hsic_11 = np.sum(K1_centered * K1_centered) / (n ** 2)
    hsic_22 = np.sum(K2_centered * K2_centered) / (n ** 2)

    # CKA
    cka = hsic_12 / np.sqrt(hsic_11 * hsic_22)

    return cka


def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Initialize output directory
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Save config to output directory
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Extract config parameters
    repr_path_1 = Path(config['representations_path_1'])
    repr_path_2 = Path(config['representations_path_2'])
    token_index = config.get('token_index', -1)
    layer_index = config.get('layer_index', -1)
    center_kernels = config.get('center_kernels', True)
    kernel_type = config.get('kernel_type', 'linear')
    use_gpu = config.get('use_gpu', True)
    checkpoint_steps = config.get('checkpoint_steps', None)  # None means all
    city_filter = config.get('city_filter', None)
    cities_csv = config.get('cities_csv', 'data/datasets/cities/cities.csv')

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if use_gpu and not gpu_available:
        print("WARNING: GPU requested but not available, falling back to CPU")
        use_gpu = False

    print(f"Configuration:")
    print(f"  Representations path 1: {repr_path_1}")
    print(f"  Representations path 2: {repr_path_2}")
    print(f"  Token index: {token_index} ({'all' if token_index == -1 else token_index})")
    print(f"  Layer index: {layer_index} ({'all' if layer_index == -1 else layer_index})")
    print(f"  Kernel type: {kernel_type}")
    print(f"  Center kernels: {center_kernels}")
    print(f"  City filter: {city_filter if city_filter else 'None (use all cities)'}")
    print(f"  Using GPU: {use_gpu} (available: {gpu_available})")
    print(f"  Output directory: {output_dir}")

    if kernel_type != 'linear':
        raise NotImplementedError(f"Only linear kernel is currently supported, got {kernel_type}")

    # Load cities dataframe if filtering is needed
    if city_filter:
        cities_df = pd.read_csv(cities_csv)
        print(f"\nLoaded {len(cities_df)} cities for filtering")

    # Find checkpoint directories for both paths
    checkpoints_1 = {}
    checkpoints_2 = {}

    for checkpoint_dir in sorted(repr_path_1.glob('checkpoint-*')):
        match = re.match(r'checkpoint-(\d+)', checkpoint_dir.name)
        if match:
            step = int(match.group(1))
            if checkpoint_steps is None or step in checkpoint_steps:
                checkpoints_1[step] = checkpoint_dir

    for checkpoint_dir in sorted(repr_path_2.glob('checkpoint-*')):
        match = re.match(r'checkpoint-(\d+)', checkpoint_dir.name)
        if match:
            step = int(match.group(1))
            if checkpoint_steps is None or step in checkpoint_steps:
                checkpoints_2[step] = checkpoint_dir

    print(f"\nFound {len(checkpoints_1)} checkpoints in path 1")
    print(f"Found {len(checkpoints_2)} checkpoints in path 2")

    # Find common checkpoint steps
    common_steps = sorted(set(checkpoints_1.keys()) & set(checkpoints_2.keys()))
    print(f"Common checkpoints: {len(common_steps)}")

    if not common_steps:
        raise ValueError("No common checkpoint steps found!")

    # Compute CKA for each common checkpoint
    cka_results = {}
    city_order_used = None  # Store the city order used for all computations

    for step in tqdm(common_steps, desc="Computing CKA"):
        try:
            # Load representations from both models
            repr_1, meta_1, city_ids_1 = load_checkpoint_representations(
                checkpoints_1[step], token_index, layer_index
            )
            repr_2, meta_2, city_ids_2 = load_checkpoint_representations(
                checkpoints_2[step], token_index, layer_index
            )

            # Apply city filter if specified
            if city_filter:
                # Create temporary dataframes with city info
                city_info_1 = meta_1.get('city_info', [])
                city_info_2 = meta_2.get('city_info', [])

                temp_df_1 = pd.DataFrame(city_info_1)
                temp_df_1['city_id'] = temp_df_1['row_id']

                temp_df_2 = pd.DataFrame(city_info_2)
                temp_df_2['city_id'] = temp_df_2['row_id']

                # Apply filter
                filtered_df_1 = filter_dataframe_by_pattern(temp_df_1, city_filter, column_name='region')
                filtered_df_2 = filter_dataframe_by_pattern(temp_df_2, city_filter, column_name='region')

                filtered_ids_1 = set(filtered_df_1['city_id'].values)
                filtered_ids_2 = set(filtered_df_2['city_id'].values)

                # Get indices of cities that pass the filter
                valid_indices_1 = [i for i, cid in enumerate(city_ids_1) if cid in filtered_ids_1]
                valid_indices_2 = [i for i, cid in enumerate(city_ids_2) if cid in filtered_ids_2]

                # Filter representations and city_ids
                repr_1 = repr_1[valid_indices_1]
                city_ids_1 = [city_ids_1[i] for i in valid_indices_1]

                repr_2 = repr_2[valid_indices_2]
                city_ids_2 = [city_ids_2[i] for i in valid_indices_2]

            # Sort both by city ID to ensure alignment
            sorted_indices_1 = np.argsort(city_ids_1)
            sorted_indices_2 = np.argsort(city_ids_2)

            city_ids_1_sorted = [city_ids_1[i] for i in sorted_indices_1]
            city_ids_2_sorted = [city_ids_2[i] for i in sorted_indices_2]

            repr_1_sorted = repr_1[sorted_indices_1]
            repr_2_sorted = repr_2[sorted_indices_2]

            # Assert that city IDs are the same
            if city_ids_1_sorted != city_ids_2_sorted:
                print(f"\nError at step {step}: City IDs don't match!")
                print(f"  Path 1 has {len(city_ids_1_sorted)} cities")
                print(f"  Path 2 has {len(city_ids_2_sorted)} cities")

                # Find differences
                set1 = set(city_ids_1_sorted)
                set2 = set(city_ids_2_sorted)
                only_in_1 = set1 - set2
                only_in_2 = set2 - set1

                if only_in_1:
                    print(f"  Cities only in path 1: {list(only_in_1)[:10]}")
                if only_in_2:
                    print(f"  Cities only in path 2: {list(only_in_2)[:10]}")

                raise ValueError(f"City IDs mismatch at step {step}")

            # Store city order (should be the same for all checkpoints)
            if city_order_used is None:
                city_order_used = city_ids_1_sorted

            print(f"\n  Step {step}: {len(city_ids_1_sorted)} cities, repr shape {repr_1_sorted.shape}")

            # Compute kernel matrices
            if use_gpu:
                K1 = compute_kernel_matrix_gpu(repr_1_sorted)
                K2 = compute_kernel_matrix_gpu(repr_2_sorted)

                if center_kernels:
                    K1 = center_kernel_matrix_gpu(K1)
                    K2 = center_kernel_matrix_gpu(K2)

                cka = compute_cka_gpu(K1, K2, already_centered=center_kernels)
            else:
                # CPU implementation
                K1 = repr_1_sorted @ repr_1_sorted.T
                K2 = repr_2_sorted @ repr_2_sorted.T

                if center_kernels:
                    n = K1.shape[0]
                    H = np.eye(n) - np.ones((n, n)) / n
                    K1 = H @ K1 @ H
                    K2 = H @ K2 @ H

                cka = compute_cka_cpu(K1, K2, already_centered=center_kernels)

            cka_results[step] = cka
            print(f"    CKA = {cka:.6f}")

        except Exception as e:
            print(f"  Error at step {step}: {e}")
            continue

    if not cka_results:
        raise ValueError("No CKA values were successfully computed!")

    # Save results
    print(f"\nSaving results to {output_dir}...")

    # Save as JSON for easy loading
    results = {
        'cka_values': cka_results,
        'checkpoint_steps': list(cka_results.keys()),
        'config': config,
        'representations_path_1': str(repr_path_1),
        'representations_path_2': str(repr_path_2),
        'n_cities': len(city_order_used) if city_order_used else None,
        'city_ids': city_order_used,
        'token_index': token_index,
        'layer_index': layer_index,
        'kernel_type': kernel_type,
        'centered': center_kernels,
        'city_filter': city_filter,
        'mean_cka': np.mean(list(cka_results.values())),
        'std_cka': np.std(list(cka_results.values())),
        'min_cka': min(cka_results.values()),
        'max_cka': max(cka_results.values()),
        'final_cka': cka_results[max(cka_results.keys())]
    }

    # Save main results
    results_path = output_dir / 'cka_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Also save as CSV for easy plotting
    cka_df = pd.DataFrame([
        {'step': step, 'cka': cka}
        for step, cka in sorted(cka_results.items())
    ])
    csv_path = output_dir / 'cka_values.csv'
    cka_df.to_csv(csv_path, index=False)
    print(f"CKA values saved to {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("CKA Summary")
    print(f"{'='*60}")
    print(f"Number of checkpoints: {len(cka_results)}")
    print(f"Mean CKA: {results['mean_cka']:.6f}")
    print(f"Std CKA: {results['std_cka']:.6f}")
    print(f"Min CKA: {results['min_cka']:.6f}")
    print(f"Max CKA: {results['max_cka']:.6f}")
    print(f"Final CKA: {results['final_cka']:.6f}")
    print(f"Cities used: {len(city_order_used) if city_order_used else 'N/A'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute CKA from representations')
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)