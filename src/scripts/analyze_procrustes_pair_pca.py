"""
Analyze Procrustes distance between two experiments after PCA dimensionality reduction.

Each experiment gets its own PCA transformation fitted on non-Atlantis cities.
Then Procrustes distance is computed on the PCA-reduced representations (default: 3 components).

Procrustes distance measures geometric shape similarity after optimal alignment (rotation/scaling).
Unlike CKA, it preserves shape differences while being invariant to rotation/reflection.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')
sys.path.insert(0, str(project_root))

import argparse
import yaml
import json
import shutil
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

from src.utils import init_directory
from src.analysis.cka_v2.load_representations import load_all_checkpoints, apply_city_filter


def apply_pca_to_representations(repr_dict, meta_dict, city_filter, n_components=3):
    """
    Apply PCA to each checkpoint's representations.

    PCA is fitted on non-Atlantis cities and applied to all cities.

    Args:
        repr_dict: Dict mapping checkpoint step -> representation array
        meta_dict: Dict mapping checkpoint step -> metadata DataFrame
        city_filter: Filter string for training PCA (e.g., exclude Atlantis)
        n_components: Number of PCA components to keep

    Returns:
        Dict mapping checkpoint step -> PCA-reduced representation array
    """
    pca_reprs = {}

    for step in repr_dict.keys():
        representations = repr_dict[step]
        metadata = meta_dict[step]

        # Apply filter to get training cities (non-Atlantis)
        if city_filter:
            train_meta = apply_city_filter(metadata.copy(), city_filter)
            train_indices = train_meta.index.values
            train_reprs = representations[train_indices]
        else:
            train_reprs = representations

        # Fit PCA on training cities
        pca = PCA(n_components=n_components)
        pca.fit(train_reprs)

        # Transform all representations
        pca_reprs[step] = pca.transform(representations)

    return pca_reprs


def align_pca_representations(pca_repr1, meta1, pca_repr2, meta2, city_filter=None):
    """
    Align two PCA-reduced representation matrices by matching cities.

    Args:
        pca_repr1: First PCA representation matrix (n1 x n_components)
        meta1: Metadata DataFrame for first representations
        pca_repr2: Second PCA representation matrix (n2 x n_components)
        meta2: Metadata DataFrame for second representations
        city_filter: Optional filter for selecting cities

    Returns:
        Tuple of (aligned_pca_repr1, aligned_pca_repr2, common_city_ids)
    """
    # Apply city filter if provided
    if city_filter:
        filtered_meta1 = apply_city_filter(meta1.copy(), city_filter)
        filtered_meta2 = apply_city_filter(meta2.copy(), city_filter)

        # Filter representations
        pca_repr1 = pca_repr1[filtered_meta1.index.values]
        pca_repr2 = pca_repr2[filtered_meta2.index.values]

        # Update metadata
        meta1 = filtered_meta1.reset_index(drop=True)
        meta2 = filtered_meta2.reset_index(drop=True)

    # Find common cities
    cities1 = set(meta1['city_id'].tolist())
    cities2 = set(meta2['city_id'].tolist())
    common_cities = sorted(cities1 & cities2)

    if len(common_cities) == 0:
        raise ValueError("No common cities found between representations")

    # Create mapping from city_id to index
    city_to_idx1 = {city_id: idx for idx, city_id in enumerate(meta1['city_id'])}
    city_to_idx2 = {city_id: idx for idx, city_id in enumerate(meta2['city_id'])}

    # Extract aligned representations
    indices1 = [city_to_idx1[city_id] for city_id in common_cities]
    indices2 = [city_to_idx2[city_id] for city_id in common_cities]

    aligned_pca_repr1 = pca_repr1[indices1]
    aligned_pca_repr2 = pca_repr2[indices2]

    return aligned_pca_repr1, aligned_pca_repr2, common_cities


def compute_procrustes_distance(X, Y):
    """
    Compute Procrustes distance between two point clouds.

    Procrustes analysis optimally aligns Y to X using rotation and scaling,
    then measures the residual distance.

    Args:
        X: First point cloud (n x d)
        Y: Second point cloud (n x d)

    Returns:
        float: Procrustes distance (standardized residual sum of squares)
    """
    # scipy's procrustes returns: (mtx1, mtx2, disparity)
    # mtx1, mtx2 are the transformed matrices
    # disparity is the sum of squared differences (already normalized)
    _, _, disparity = procrustes(X, Y)

    return float(disparity)


def plot_procrustes_timeline(df, output_path, summary):
    """Plot Procrustes distance timeline across checkpoints."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['step'], df['procrustes_distance'], linewidth=2, color='crimson')
    ax.axhline(y=summary['mean_procrustes'], color='gray', linestyle='--', alpha=0.5,
               label=f"Mean: {summary['mean_procrustes']:.4f}")

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Procrustes Distance', fontsize=12)
    ax.set_title(f"Procrustes Distance Timeline (First {summary['n_pca_components']} PCs): {summary['exp1']} vs {summary['exp2']}\nLayer {summary['layer']}",
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(config_path, overwrite=False, debug=False):
    """Analyze Procrustes distance between two experiments using PCA-reduced representations."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    if 'n_pca_components' not in config:
        raise ValueError("FATAL: 'n_pca_components' required in config")

    # Initialize output directory
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Create subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    # Copy config to output
    shutil.copy(config_path, output_dir / 'config.yaml')

    if debug:
        print(f"DEBUG MODE: Output will be written to {output_dir}")

    # Get parameters
    n_pca_components = config['n_pca_components']
    city_filter = config.get('city_filter', None)
    pca_train_filter = config.get('pca_train_filter', city_filter)  # Filter for PCA training (exclude Atlantis)

    # Load representations for both experiments
    print(f"Loading representations for {config['exp1']['name']}...")
    repr1, meta1 = load_all_checkpoints(Path(config['exp1']['repr_dir']))

    print(f"Loading representations for {config['exp2']['name']}...")
    repr2, meta2 = load_all_checkpoints(Path(config['exp2']['repr_dir']))

    # Find common checkpoints
    common_steps = sorted(set(repr1.keys()) & set(repr2.keys()))

    if len(common_steps) == 0:
        raise ValueError(f"No common checkpoints found between {config['exp1']['name']} and {config['exp2']['name']}")

    print(f"Found {len(common_steps)} common checkpoints")

    # Filter checkpoints if specified
    if config.get('checkpoint_steps') is not None:
        checkpoint_steps = config['checkpoint_steps']
        common_steps = [s for s in common_steps if s in checkpoint_steps]
        print(f"Filtered to {len(common_steps)} specified checkpoints")

    # Apply PCA to each experiment's representations
    print(f"\nApplying PCA ({n_pca_components} components) to {config['exp1']['name']}...")
    print(f"  PCA fitted on cities matching: {pca_train_filter}")
    pca_repr1 = apply_pca_to_representations(repr1, meta1, pca_train_filter, n_pca_components)

    print(f"\nApplying PCA ({n_pca_components} components) to {config['exp2']['name']}...")
    print(f"  PCA fitted on cities matching: {pca_train_filter}")
    pca_repr2 = apply_pca_to_representations(repr2, meta2, pca_train_filter, n_pca_components)

    # Compute Procrustes distance for each checkpoint
    print(f"\nComputing Procrustes distance on PCA-reduced representations...")
    results = []

    for step in tqdm(common_steps, desc="Computing Procrustes"):
        # Align PCA representations
        R1, R2, common_cities = align_pca_representations(
            pca_repr1[step], meta1[step],
            pca_repr2[step], meta2[step],
            city_filter=city_filter
        )

        if debug and step == common_steps[0]:
            print(f"DEBUG: {len(common_cities)} common cities after filtering")
            print(f"DEBUG: R1 shape: {R1.shape}, R2 shape: {R2.shape}")

        # Compute Procrustes distance
        proc_dist = compute_procrustes_distance(R1, R2)

        results.append({'step': step, 'procrustes_distance': proc_dist})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'procrustes_timeline.csv', index=False)

    # Compute summary statistics
    summary = {
        'exp1': config['exp1']['name'],
        'exp2': config['exp2']['name'],
        'layer': config.get('layer', None),
        'n_pca_components': n_pca_components,
        'n_checkpoints': len(results),
        'n_cities': len(common_cities),
        'final_procrustes': float(results[-1]['procrustes_distance']),
        'mean_procrustes': float(df['procrustes_distance'].mean()),
        'std_procrustes': float(df['procrustes_distance'].std()),
        'min_procrustes': float(df['procrustes_distance'].min()),
        'max_procrustes': float(df['procrustes_distance'].max()),
    }

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot timeline
    if config.get('save_timeline_plot', True):
        plot_procrustes_timeline(df, output_dir / 'procrustes_timeline.png', summary)

    print(f"\nResults saved to {output_dir}")
    print(f"Final Procrustes Distance: {summary['final_procrustes']:.4f}")
    print(f"Mean Procrustes Distance: {summary['mean_procrustes']:.4f} ± {summary['std_procrustes']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Procrustes distance between two experiments using PCA-reduced representations')
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
