#!/usr/bin/env python3
"""
Plot 3D representation with:
- Dim 1: X linear probe (fitted on world cities)
- Dim 2: Y linear probe (fitted on world cities)
- Dim 3: "Nullest" direction - minimum variance direction for world cities

If Atlantis is well-integrated, it will be at z≈0. If not, it will pop out.
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import torch
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import init_directory

# Colors matching plotly qualitative
REGION_COLORS = {
    "Africa": "#636EFA",
    "Atlantis": "#EF553B",
    "Central Asia": "#00CC96",
    "China": "#AB63FA",
    "Eastern Europe": "#FFA15A",
    "India": "#19D3F3",
    "Japan": "#FF6692",
    "Korea": "#B6E880",
    "Middle East": "#FF97FF",
    "North America": "#FECB52",
    "Oceania": "#1F77B4",
    "South America": "#FF7F0E",
    "Southeast Asia": "#2CA02C",
    "Western Europe": "#D62728",
}


def load_representations(repr_path):
    """Load representations and metadata."""
    repr_file = repr_path / 'representations.pt'
    meta_file = repr_path / 'metadata.json'

    data = torch.load(repr_file, map_location='cpu')
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    # Get flattened representations
    representations = data['representations_flat'].numpy()
    city_info = metadata['city_info']

    return representations, city_info


def main(config_path, overwrite=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")
    if 'repr_path' not in config:
        raise ValueError("FATAL: 'repr_path' required in config")

    output_dir = init_directory(config['output_dir'], overwrite=overwrite)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load representations
    repr_path = Path(config['repr_path'])
    representations, city_info = load_representations(repr_path)
    print(f"Loaded {len(city_info)} cities, dim={representations.shape[1]}")

    # Separate world cities and Atlantis
    world_mask = np.array([c['region'] != 'Atlantis' for c in city_info])
    atlantis_mask = ~world_mask

    world_repr = representations[world_mask]
    atlantis_repr = representations[atlantis_mask]

    world_info = [c for c in city_info if c['region'] != 'Atlantis']
    atlantis_info = [c for c in city_info if c['region'] == 'Atlantis']

    print(f"World cities: {len(world_info)}, Atlantis: {len(atlantis_info)}")

    # Get coordinates
    world_x = np.array([c['x'] for c in world_info])
    world_y = np.array([c['y'] for c in world_info])
    atlantis_x = np.array([c['x'] for c in atlantis_info])
    atlantis_y = np.array([c['y'] for c in atlantis_info])

    # Fit X and Y probes on world cities
    n_train = config.get('n_train', 4000)
    np.random.seed(42)
    train_idx = np.random.choice(len(world_repr), min(n_train, len(world_repr)), replace=False)

    X_train = world_repr[train_idx]
    x_train = world_x[train_idx]
    y_train = world_y[train_idx]

    x_probe = LinearRegression()
    y_probe = LinearRegression()
    x_probe.fit(X_train, x_train)
    y_probe.fit(X_train, y_train)

    print(f"X probe R²: {x_probe.score(X_train, x_train):.4f}")
    print(f"Y probe R²: {y_probe.score(X_train, y_train):.4f}")

    # Find normal direction to the plane defined by 3 city clusters
    # Pick 10 cities near each location, average their representations
    # Then compute normal to the plane formed by these 3 points

    # Define city selection regions (in raw x,y coords)
    # Boston/NYC: x in [-800, -680], y in [380, 450]
    # South Africa: x in [200, 350], y in [-350, -200]
    # North China: x in [1100, 1250], y in [350, 450]

    n_cities_per_cluster = config.get('n_cities_per_cluster', 10)
    np.random.seed(config.get('random_seed', 42))

    def get_cluster_repr(world_info, world_repr, x_range, y_range, n=10):
        """Get average representation of n cities in the given x,y range."""
        mask = []
        for i, c in enumerate(world_info):
            if x_range[0] <= c['x'] <= x_range[1] and y_range[0] <= c['y'] <= y_range[1]:
                mask.append(i)
        if len(mask) < n:
            raise ValueError(f"Only found {len(mask)} cities in range, need {n}")
        selected = np.random.choice(mask, n, replace=False)
        return world_repr[selected].mean(axis=0)

    # Get average representations for 3 clusters
    boston_repr = get_cluster_repr(world_info, world_repr, (-800, -680), (380, 450), n_cities_per_cluster)
    africa_repr = get_cluster_repr(world_info, world_repr, (200, 350), (-350, -200), n_cities_per_cluster)
    china_repr = get_cluster_repr(world_info, world_repr, (1100, 1250), (350, 450), n_cities_per_cluster)

    print(f"Selected {n_cities_per_cluster} cities per cluster (Boston, S.Africa, N.China)")

    # Compute vectors in the plane
    v1 = africa_repr - boston_repr
    v2 = china_repr - boston_repr

    # For high-dimensional vectors, we can't use cross product directly
    # Instead, find the direction orthogonal to both v1 and v2
    # This is done by: project out v1 and v2 components from a random vector
    # Or use SVD on the matrix [v1, v2] to find null space

    # Stack v1, v2 and find orthogonal complement
    plane_basis = np.vstack([v1, v2])  # 2 x dim
    # SVD to find null space
    U, S, Vt = np.linalg.svd(plane_basis, full_matrices=True)
    # The rows of Vt after the first 2 are orthogonal to v1 and v2
    # Pick the first one (or any)
    nullest_direction = Vt[2]  # First vector in null space
    nullest_direction = nullest_direction / np.linalg.norm(nullest_direction)

    # Compute variance of world cities along this direction
    world_mean = world_repr.mean(axis=0)
    world_centered = world_repr - world_mean
    world_proj = world_centered @ nullest_direction
    nullest_variance = world_proj.var()

    print(f"Normal direction found via SVD null space")
    print(f"World variance along normal: {nullest_variance:.6f}")

    # Project all cities onto the 3 dimensions
    # Dim 1: X probe prediction
    # Dim 2: Y probe prediction
    # Dim 3: Projection onto nullest direction (centered)

    world_dim1 = x_probe.predict(world_repr)
    world_dim2 = y_probe.predict(world_repr)
    world_dim3 = (world_repr - world_mean) @ nullest_direction

    atlantis_dim1 = x_probe.predict(atlantis_repr)
    atlantis_dim2 = y_probe.predict(atlantis_repr)
    atlantis_dim3 = (atlantis_repr - world_mean) @ nullest_direction

    print(f"\nWorld dim3: mean={world_dim3.mean():.4f}, std={world_dim3.std():.4f}")
    print(f"Atlantis dim3: mean={atlantis_dim3.mean():.4f}, std={atlantis_dim3.std():.4f}")

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot world cities by region
    for region in sorted(set(c['region'] for c in world_info)):
        mask = np.array([c['region'] == region for c in world_info])
        color = REGION_COLORS.get(region, '#888888')
        ax.scatter(world_dim1[mask], world_dim2[mask], world_dim3[mask],
                   c=color, s=8, alpha=0.6, label=region)

    # Plot Atlantis
    ax.scatter(atlantis_dim1, atlantis_dim2, atlantis_dim3,
               c=REGION_COLORS['Atlantis'], s=50, alpha=0.9,
               edgecolors='white', linewidths=0.5, label='Atlantis')

    ax.set_xlabel('Dim 1 (X probe)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dim 2 (Y probe)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Dim 3 (nullest)', fontsize=12, fontweight='bold')

    ax.legend(loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    output_path = figures_dir / 'probe_3d_nullspace.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Also save a 2D projection (dim1 vs dim3) to clearly see the z-axis separation
    fig, ax = plt.subplots(figsize=(10, 8))

    for region in sorted(set(c['region'] for c in world_info)):
        mask = np.array([c['region'] == region for c in world_info])
        color = REGION_COLORS.get(region, '#888888')
        ax.scatter(world_dim1[mask], world_dim3[mask], c=color, s=8, alpha=0.6)

    ax.scatter(atlantis_dim1, atlantis_dim3,
               c=REGION_COLORS['Atlantis'], s=50, alpha=0.9,
               edgecolors='white', linewidths=0.5, label='Atlantis')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Dim 1 (X probe)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dim 3 (nullest direction)', fontsize=14, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2, length=5, labelsize=12)

    plt.tight_layout()
    output_path2 = figures_dir / 'probe_x_vs_nullest.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()

    # Save results
    results = {
        'n_world': len(world_info),
        'n_atlantis': len(atlantis_info),
        'n_train': n_train,
        'x_probe_r2': float(x_probe.score(X_train, x_train)),
        'y_probe_r2': float(y_probe.score(X_train, y_train)),
        'nullest_variance': float(nullest_variance),
        'world_dim3_mean': float(world_dim3.mean()),
        'world_dim3_std': float(world_dim3.std()),
        'atlantis_dim3_mean': float(atlantis_dim3.mean()),
        'atlantis_dim3_std': float(atlantis_dim3.std()),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite)
