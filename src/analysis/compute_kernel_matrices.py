#!/usr/bin/env python3
"""
Compute kernel matrices from saved representations for CKA analysis.

This script loads representations from checkpoint directories and computes
kernel matrices that can be used for CKA (Centered Kernel Alignment) analysis
between different models or checkpoints.

Usage:
    python compute_kernel_matrices.py configs/analysis/kernel_matrices.yaml
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
project_root = Path('')
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
        Tuple of (representations, metadata, city_info)
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

    return representations, metadata, metadata.get('city_info', [])


def compute_kernel_matrix(X, kernel_type='linear', use_gpu=False):
    """
    Compute kernel matrix for given representations.

    Args:
        X: Representations matrix (n_samples, n_features)
        kernel_type: Type of kernel ('linear' or 'rbf')
        use_gpu: Whether to use GPU acceleration

    Returns:
        Kernel matrix (n_samples, n_samples)
    """
    if use_gpu and torch.cuda.is_available():
        # Convert to torch tensor and move to GPU
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X).float().cuda()
        else:
            X_torch = X.float().cuda() if not X.is_cuda else X.float()

        if kernel_type == 'linear':
            # Linear kernel on GPU: K = X @ X.T
            K = torch.mm(X_torch, X_torch.t())
            # Convert back to numpy
            K = K.cpu().numpy()
        elif kernel_type == 'rbf':
            # RBF kernel on GPU
            # Compute pairwise distances
            XX = torch.sum(X_torch * X_torch, dim=1, keepdim=True)
            distances_sq = XX - 2 * torch.mm(X_torch, X_torch.t()) + XX.t()
            distances_sq = torch.clamp(distances_sq, min=0)  # Numerical stability

            # Median heuristic for gamma
            distances = torch.sqrt(distances_sq)
            upper_tri = torch.triu(distances, diagonal=1)
            non_zero_dists = upper_tri[upper_tri > 0]
            median_dist = torch.median(non_zero_dists).item() if len(non_zero_dists) > 0 else 1.0
            gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0

            # RBF kernel: exp(-gamma * ||x - y||^2)
            K = torch.exp(-gamma * distances_sq)
            K = K.cpu().numpy()
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    else:
        # CPU implementation (original)
        if kernel_type == 'linear':
            # Linear kernel: K = X @ X.T
            K = X @ X.T
        elif kernel_type == 'rbf':
            # RBF kernel with automatic bandwidth selection
            from sklearn.metrics.pairwise import rbf_kernel
            # Use median heuristic for gamma
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(X, metric='euclidean')
            median_dist = np.median(distances[distances > 0])
            gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
            K = rbf_kernel(X, gamma=gamma)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    return K


def center_kernel_matrix(K, use_gpu=False):
    """
    Center a kernel matrix.

    Args:
        K: Kernel matrix (n, n)
        use_gpu: Whether to use GPU acceleration

    Returns:
        Centered kernel matrix
    """
    n = K.shape[0]

    if use_gpu and torch.cuda.is_available():
        # GPU implementation
        if isinstance(K, np.ndarray):
            K_torch = torch.from_numpy(K).float().cuda()
        else:
            K_torch = K.float().cuda() if not K.is_cuda else K.float()

        # Create centering matrix on GPU
        ones = torch.ones((n, n), device='cuda') / n
        H = torch.eye(n, device='cuda') - ones

        # Center the kernel
        K_centered = torch.mm(torch.mm(H, K_torch), H)

        # Convert back to numpy
        K_centered = K_centered.cpu().numpy()
    else:
        # CPU implementation (original)
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        K_centered = H @ K @ H

    return K_centered


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
    representations_base_path = Path(config['representations_base_path'])
    token_index = config.get('token_index', -1)
    layer_index = config.get('layer_index', -1)
    kernel_type = config.get('kernel_type', 'linear')
    center_kernels = config.get('center_kernels', True)
    checkpoint_steps = config.get('checkpoint_steps', None)  # None means all
    use_gpu = config.get('use_gpu', True)  # Default to using GPU if available

    # City filtering parameters
    cities_csv = config.get('cities_csv', 'data/datasets/cities/cities.csv')
    city_filter = config.get('city_filter', None)  # Optional regex filter

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if use_gpu and not gpu_available:
        print("WARNING: GPU requested but not available, falling back to CPU")
        use_gpu = False

    print(f"Configuration:")
    print(f"  Representations path: {representations_base_path}")
    print(f"  Token index: {token_index} ({'all' if token_index == -1 else token_index})")
    print(f"  Layer index: {layer_index} ({'all' if layer_index == -1 else layer_index})")
    print(f"  Kernel type: {kernel_type}")
    print(f"  Center kernels: {center_kernels}")
    print(f"  City filter: {city_filter if city_filter else 'None (use all cities)'}")
    print(f"  Output directory: {output_dir}")
    print(f"  Using GPU: {use_gpu} (available: {gpu_available})")

    # Load cities dataframe for filtering
    cities_df = pd.read_csv(cities_csv)
    print(f"\nLoaded {len(cities_df)} cities from {cities_csv}")

    # Find all checkpoint directories
    checkpoint_dirs = []
    for checkpoint_dir in sorted(representations_base_path.glob('checkpoint-*')):
        match = re.match(r'checkpoint-(\d+)', checkpoint_dir.name)
        if match:
            step = int(match.group(1))
            if checkpoint_steps is None or step in checkpoint_steps:
                checkpoint_dirs.append((step, checkpoint_dir))

    checkpoint_dirs.sort(key=lambda x: x[0])
    print(f"\nFound {len(checkpoint_dirs)} checkpoints to process")

    if not checkpoint_dirs:
        raise ValueError("No checkpoints found!")

    # Process each checkpoint
    kernel_matrices = {}
    city_orders = {}  # Store the ordered city IDs for each checkpoint

    for step, checkpoint_dir in tqdm(checkpoint_dirs, desc="Processing checkpoints"):
        print(f"\nProcessing checkpoint-{step}...")

        try:
            # Load representations
            representations, metadata, city_info = load_checkpoint_representations(
                checkpoint_dir, token_index, layer_index
            )

            # Get city IDs from metadata
            city_ids = [c['row_id'] for c in city_info]
            n_cities = len(city_ids)

            # Apply city filter if specified
            if city_filter:
                # Create a temporary dataframe with city info
                temp_df = pd.DataFrame(city_info)
                temp_df['city_id'] = temp_df['row_id']  # Ensure city_id column exists

                # Apply filter
                filtered_df = filter_dataframe_by_pattern(temp_df, city_filter, column_name='region')
                filtered_city_ids = set(filtered_df['city_id'].values)

                # Get indices of cities that pass the filter
                valid_indices = [i for i, cid in enumerate(city_ids) if cid in filtered_city_ids]

                if not valid_indices:
                    print(f"  Warning: No cities matched filter '{city_filter}', skipping checkpoint")
                    continue

                # Filter representations and city_ids
                representations = representations[valid_indices]
                city_ids = [city_ids[i] for i in valid_indices]

                print(f"  Filtered from {n_cities} to {len(city_ids)} cities")

            # Sort cities by city_id to ensure consistent ordering
            sorted_indices = np.argsort(city_ids)
            city_ids_sorted = [city_ids[i] for i in sorted_indices]
            representations_sorted = representations[sorted_indices]

            print(f"  Representations shape: {representations_sorted.shape}")

            # Compute kernel matrix
            K = compute_kernel_matrix(representations_sorted, kernel_type, use_gpu=use_gpu)

            # Center kernel if requested
            if center_kernels:
                K = center_kernel_matrix(K, use_gpu=use_gpu)

            # Store results
            kernel_matrices[step] = K
            city_orders[step] = city_ids_sorted

            print(f"  Kernel matrix shape: {K.shape}")
            print(f"  Kernel matrix range: [{K.min():.4f}, {K.max():.4f}]")

        except Exception as e:
            print(f"  Error processing checkpoint-{step}: {e}")
            continue

    if not kernel_matrices:
        raise ValueError("No kernel matrices were successfully computed!")

    # Save kernel matrices
    print(f"\nSaving kernel matrices to {output_dir}...")

    # Save as single file with all checkpoints
    save_path = output_dir / 'kernel_matrices.pt'
    torch.save({
        'kernel_matrices': kernel_matrices,
        'city_orders': city_orders,
        'config': config,
        'checkpoint_steps': list(kernel_matrices.keys()),
        'kernel_type': kernel_type,
        'centered': center_kernels,
        'token_index': token_index,
        'layer_index': layer_index,
        'city_filter': city_filter
    }, save_path)

    print(f"Saved kernel matrices to {save_path}")

    # Also save individual matrices for easier access
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(exist_ok=True)

    for step, K in kernel_matrices.items():
        individual_path = individual_dir / f'kernel_step_{step}.npy'
        np.save(individual_path, K)

        # Save city order for this checkpoint
        city_order_path = individual_dir / f'city_order_step_{step}.json'
        with open(city_order_path, 'w') as f:
            json.dump(city_orders[step], f)

    print(f"Saved individual matrices to {individual_dir}/")

    # Save summary
    summary = {
        'n_checkpoints': len(kernel_matrices),
        'checkpoint_steps': list(kernel_matrices.keys()),
        'kernel_type': kernel_type,
        'centered': center_kernels,
        'representations_shape_info': {
            'token_index': token_index,
            'layer_index': layer_index
        },
        'city_filter': city_filter,
        'n_cities_per_checkpoint': {step: len(city_orders[step]) for step in kernel_matrices.keys()},
        'kernel_shapes': {step: list(K.shape) for step, K in kernel_matrices.items()}
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {summary_path}")
    print(f"\nProcessing complete!")
    print(f"  Total checkpoints processed: {len(kernel_matrices)}")
    print(f"  Kernel matrix sizes: {[K.shape for K in kernel_matrices.values()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute kernel matrices from representations')
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)