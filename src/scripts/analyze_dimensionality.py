#!/usr/bin/env python3
"""
Analyze intrinsic dimensionality of pre-computed representations.
Works with representations created by analyze_representations_higher.py
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import stats
import json
from tqdm import tqdm

# Add parent directory to path
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory
from src.dimensionality import (
    twonn_dimension,
    correlation_dimension,
    local_pca_2d_energy,
    test_for_2d_manifold,
    participation_ratio,
    mle_dimension
)

def load_representations(representations_path, checkpoint_name=None):
    """Load representations from specified path."""
    base_path = Path(representations_path)

    if checkpoint_name:
        # Load specific checkpoint
        checkpoint_path = base_path / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

        # Load representations
        reps_file = checkpoint_path / "representations.npy"
        if not reps_file.exists():
            raise FileNotFoundError(f"Representations file {reps_file} not found")

        representations = np.load(reps_file)

        # Load labels if available
        labels_file = checkpoint_path / "labels.npy"
        labels = np.load(labels_file) if labels_file.exists() else None

        return representations, labels, checkpoint_name

    else:
        # Find all checkpoints
        checkpoints = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')])

        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {base_path}")

        # Use the last checkpoint
        checkpoint_path = checkpoints[-1]
        checkpoint_name = checkpoint_path.name

        # Load representations
        reps_file = checkpoint_path / "representations.npy"
        if not reps_file.exists():
            raise FileNotFoundError(f"Representations file {reps_file} not found")

        representations = np.load(reps_file)

        # Load labels if available
        labels_file = checkpoint_path / "labels.npy"
        labels = np.load(labels_file) if labels_file.exists() else None

        return representations, labels, checkpoint_name

def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize output directory
    output_dir = Path(config['output_dir'])
    if output_dir.exists() and not overwrite:
        print(f"Output directory {output_dir} exists. Use --overwrite to continue.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create subdirectories
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)

    # Load representations
    print(f"Loading representations from {config['representations_base_path']}")

    representations, labels, checkpoint_name = load_representations(
        config['representations_base_path'],
        config.get('checkpoint', None)
    )

    print(f"Loaded representations from {checkpoint_name}: {representations.shape}")

    # Sample if too large
    max_samples = config.get('max_samples', 5000)
    if representations.shape[0] > max_samples:
        print(f"Sampling {max_samples} from {representations.shape[0]} samples")
        idx = np.random.choice(representations.shape[0], max_samples, replace=False)
        representations = representations[idx]
        if labels is not None:
            labels = labels[idx]

    # Compute all dimensionality metrics
    print("\nComputing dimensionality metrics...")

    # Compute the main 3 metrics and test for 2D manifold
    manifold_results, is_2d = test_for_2d_manifold(representations)

    metrics = {
        'checkpoint': checkpoint_name,
        'n_samples': representations.shape[0],
        'n_dims': representations.shape[1],
        'twonn_dimension': manifold_results['twonn'],
        'correlation_dimension': manifold_results['correlation'],
        'local_pca_2d_energy': manifold_results['pca_2d_energy'],
        'is_2d_manifold': is_2d,
        # Additional metrics for backwards compatibility
        'mle_dimension': mle_dimension(representations, k_max=config.get('mle_k_max', 20)),
        'participation_ratio': participation_ratio(representations)
    }

    # Print results
    print("\n=== Dimensionality Analysis Results ===")
    print(f"Checkpoint: {metrics['checkpoint']}")
    print(f"Data shape: {metrics['n_samples']} samples × {metrics['n_dims']} dimensions")
    print(f"\n=== 2D Manifold Test (Main Metrics) ===")
    print(f"  TwoNN dimension: {metrics['twonn_dimension']:.2f}")
    print(f"  Correlation dimension: {metrics['correlation_dimension']:.2f}")
    print(f"  Local PCA 2D energy: {metrics['local_pca_2d_energy']:.3f} (1.0 = perfect 2D)")
    print(f"  >>> Is 2D manifold: {metrics['is_2d_manifold']} <<<")
    print(f"\nAdditional Metrics:")
    print(f"  MLE dimension: {metrics['mle_dimension']:.2f}")
    print(f"  Participation ratio: {metrics['participation_ratio']:.2f}")

    # Save results
    with open(output_dir / 'results' / 'dimensionality_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save as CSV for easy reading
    pd.DataFrame([metrics]).to_csv(output_dir / 'results' / 'dimensionality_metrics.csv', index=False)

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Eigenvalue spectrum for participation ratio
    ax = axes[0, 0]
    centered = representations - representations.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    eigenvalues = eigenvalues[::-1]  # Sort descending

    ax.plot(range(1, min(51, len(eigenvalues)+1)), eigenvalues[:50], 'o-', markersize=4)
    ax.set_xlabel('Component')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Eigenvalue Spectrum (PR = {metrics["participation_ratio"]:.2f})')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Cumulative variance explained
    ax = axes[0, 1]
    explained_var = eigenvalues / eigenvalues.sum()
    cumsum = np.cumsum(explained_var)

    ax.plot(range(1, min(51, len(cumsum)+1)), cumsum[:50], 'o-', markersize=4)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% variance')
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.3, label='95% variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. TwoNN ratio distribution
    ax = axes[1, 0]
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(representations)
    distances, _ = nbrs.kneighbors(representations)

    r1 = distances[:, 1]
    r2 = distances[:, 2]
    ratios = r2 / (r1 + 1e-12)

    # Plot histogram of log ratios
    log_ratios = np.log(ratios)
    ax.hist(log_ratios, bins=50, alpha=0.7, density=True, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(log_ratios), color='red', linestyle='--',
               label=f'Mean = {np.mean(log_ratios):.3f}\nDim = {metrics["twonn_dimension"]:.2f}')
    ax.set_xlabel('log(r2/r1)')
    ax.set_ylabel('Density')
    ax.set_title('TwoNN Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Correlation dimension plot (log-log)
    ax = axes[1, 1]
    # Compute pairwise distances for correlation dimension visualization
    from scipy.spatial.distance import pdist
    n_sample_viz = min(1000, representations.shape[0])
    idx_viz = np.random.choice(representations.shape[0], n_sample_viz, replace=False)
    sampled_viz = representations[idx_viz]

    distances_all = pdist(sampled_viz)
    distances_all = distances_all[distances_all > 0]

    if len(distances_all) > 0:
        # Create range of radii
        r_min, r_max = np.percentile(distances_all, [1, 99])
        radii = np.logspace(np.log10(r_min), np.log10(r_max), 30)

        # Compute correlation integral
        correlation_counts = []
        for r in radii:
            count = np.sum(distances_all < r) / len(distances_all)
            correlation_counts.append(count)

        correlation_counts = np.array(correlation_counts)
        valid = correlation_counts > 0

        if valid.sum() > 2:
            # Plot log-log
            ax.loglog(radii[valid], correlation_counts[valid], 'o-', label='Data')

            # Fit line to middle portion for slope
            mid_start, mid_end = len(radii[valid]) // 4, 3 * len(radii[valid]) // 4
            if mid_end > mid_start + 2:
                log_r = np.log(radii[valid][mid_start:mid_end])
                log_c = np.log(correlation_counts[valid][mid_start:mid_end])
                slope, intercept = np.polyfit(log_r, log_c, 1)

                # Plot fitted line
                fit_line = np.exp(intercept) * radii[valid] ** slope
                ax.loglog(radii[valid], fit_line, 'r--',
                         label=f'Slope = {slope:.2f} (Corr Dim = {metrics["correlation_dimension"]:.2f})')
                ax.legend()

            ax.set_xlabel('Radius (r)')
            ax.set_ylabel('Correlation Integral C(r)')
            ax.set_title('Correlation Dimension (log-log plot)')
            ax.grid(True, alpha=0.3)

    # 5. Summary metrics bar plot
    ax = axes[1, 2]
    metric_names = ['TwoNN\nDim', 'MLE\nDim', 'Corr\nDim', 'Part.\nRatio', 'Local\nPE']
    metric_values = [
        metrics['twonn_dimension'],
        metrics['mle_dimension'],
        metrics['correlation_dimension'],
        metrics['participation_ratio'],
        metrics['local_participation_energy']
    ]

    # Filter out NaN values
    valid_metrics = [(n, v) for n, v in zip(metric_names, metric_values) if not np.isnan(v)]
    if valid_metrics:
        names, values = zip(*valid_metrics)
        colors = ['steelblue' if 'Dim' in n else 'coral' for n in names]
        ax.bar(names, values, color=colors)
        ax.set_ylabel('Value')
        ax.set_title('Dimensionality Metrics Summary')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Dimensionality Analysis - {checkpoint_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'dimensionality_analysis.png', dpi=150, bbox_inches='tight')

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)