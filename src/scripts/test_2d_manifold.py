#!/usr/bin/env python3
"""
Test if representations lie on a 2D manifold using the 3 key metrics.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import argparse

# Add parent directory to path
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory, filter_dataframe_by_pattern
from src.dimensionality import test_for_2d_manifold
import pandas as pd

def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize output directory
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load representations
    reps_path = Path(config['representations_base_path'])

    if config.get('checkpoint'):
        checkpoint_dir = reps_path / config['checkpoint']
    else:
        # Find latest checkpoint - sort numerically by checkpoint number
        checkpoints = list(reps_path.glob('checkpoint-*'))
        if checkpoints:
            # Extract number from checkpoint-XXXXX and sort numerically
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split('-')[1]))
            checkpoint_dir = checkpoints[-1]
        else:
            checkpoint_dir = reps_path

    # Try both .npy and .pt formats
    npy_file = checkpoint_dir / 'representations.npy'
    pt_file = checkpoint_dir / 'representations.pt'

    if npy_file.exists():
        reps_file = npy_file
        print(f"Loading representations from {reps_file}")
        representations = np.load(reps_file)
    elif pt_file.exists():
        reps_file = pt_file
        print(f"Loading representations from {reps_file}")
        import torch
        repr_data = torch.load(reps_file, map_location='cpu')

        # Handle dict format from torch save
        if isinstance(repr_data, dict):
            representations = repr_data['representations']
            if isinstance(representations, torch.Tensor):
                representations = representations.numpy()
        else:
            representations = repr_data.numpy() if isinstance(repr_data, torch.Tensor) else repr_data

        # If 4D (cities, tokens, layers, hidden), extract specific layer/token
        if len(representations.shape) == 4:
            # Shape: (n_cities, n_tokens, n_layers, hidden_dim)
            # The layer is already extracted (n_layers=1), just flatten tokens
            # Use all tokens concatenated or just last token
            if representations.shape[2] == 1:
                # Already single layer extracted, flatten tokens
                representations = representations.reshape(representations.shape[0], -1)
            else:
                # Multiple layers, use layer 5, last token
                representations = representations[:, -1, 5, :]
    else:
        raise FileNotFoundError(f"No representations found in {checkpoint_dir}")
    print(f"Loaded {representations.shape}")

    # Apply filter if specified
    if config.get('filter'):
        print(f"\nApplying filter: {config['filter']}")

        # Load metadata from same checkpoint directory
        metadata_file = checkpoint_dir / 'metadata.json'
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Create dataframe from metadata's city_info
            metadata_df = pd.DataFrame(metadata['city_info'])

            # Ensure city_id column exists (might be row_id in metadata)
            if 'row_id' in metadata_df.columns and 'city_id' not in metadata_df.columns:
                metadata_df['city_id'] = metadata_df['row_id'].astype(str)

            print(f"Total cities in representations: {len(metadata_df)}")

            # Apply filter to metadata cities (same as PCA script does)
            filtered_df = filter_dataframe_by_pattern(metadata_df, config['filter'], column_name='region')
            filtered_indices = filtered_df.index.tolist()

            # Filter representations
            representations = representations[filtered_indices]
            print(f"After filtering: {representations.shape} (kept {len(filtered_indices)} out of {len(metadata_df)} cities)")
        else:
            print(f"Warning: No metadata.json found, cannot apply filter")

    # Sample if needed
    if representations.shape[0] > config.get('max_samples', 5000):
        idx = np.random.choice(representations.shape[0], config['max_samples'], replace=False)
        representations = representations[idx]
        print(f"Sampled to {representations.shape}")

    # Test for 2D manifold
    print("\nTesting for 2D manifold...")
    results, is_2d = test_for_2d_manifold(representations)

    print("\n=== 2D Manifold Test Results ===")
    print(f"TwoNN dimension: {results['twonn']:.2f}")
    print(f"Correlation dimension: {results['correlation']:.2f}")
    print(f"Local PCA 2D energy: {results['pca_2d_energy']:.3f}")
    print(f"\n>>> Is 2D manifold: {is_2d} <<<")

    # Save results
    import json
    (output_dir / 'results').mkdir(exist_ok=True)
    with open(output_dir / 'results' / 'metrics.json', 'w') as f:
        json.dump({
            'twonn': float(results['twonn']),
            'correlation': float(results['correlation']),
            'pca_2d_energy': float(results['pca_2d_energy']),
            'is_2d': bool(is_2d)
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)