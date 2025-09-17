#!/usr/bin/env python3
"""
Create 3D visualization with X/Y prediction directions and residual PCA.

This script loads saved representations and creates an interactive 3D plot where:
- X axis: Best linear direction for predicting X coordinate
- Y axis: Best linear direction for predicting Y coordinate
- Z axis: First principal component AFTER projecting out X/Y directions

This shows the spatial structure captured by the model and what information
remains after removing geographic predictive directions.

Usage:
    python visualize_xy_residual_pca_3d.py configs/analysis/xy_residual_pca_3d.yaml
"""

import argparse
import yaml
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import shutil
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory


def compute_xy_directions_and_residual_pca(representations, coordinates, n_train,
                                           method_config=None, standardize=True):
    """
    Compute optimal directions for X/Y prediction and residual PCA.

    Args:
        representations: Array of shape (n_cities, feature_dim)
        coordinates: Array of shape (n_cities, 2) with x, y coordinates
        n_train: Number of training samples
        method_config: Dictionary with probe method configuration
        standardize: Whether to standardize features before PCA

    Returns:
        projections: Dictionary with projections onto each axis
        components: Dictionary with the three orthogonal directions
        metrics: Dictionary with variance explained and R² scores
    """
    # Split data
    train_repr = representations[:n_train]
    test_repr = representations[n_train:]
    train_coords = coordinates[:n_train]
    test_coords = coordinates[n_train:]

    # Calculate mean of training coordinates (following analyze_representations.py)
    x_train_mean = train_coords[:, 0].mean()
    y_train_mean = train_coords[:, 1].mean()

    # Center the targets (predict deviations from mean)
    x_train_centered = train_coords[:, 0] - x_train_mean
    x_test_centered = test_coords[:, 0] - x_train_mean
    y_train_centered = train_coords[:, 1] - y_train_mean
    y_test_centered = test_coords[:, 1] - y_train_mean

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        train_repr_scaled = scaler.fit_transform(train_repr)
        test_repr_scaled = scaler.transform(test_repr)
        all_repr_scaled = np.vstack([train_repr_scaled, test_repr_scaled])
    else:
        train_repr_scaled = train_repr
        test_repr_scaled = test_repr
        all_repr_scaled = representations

    # 1. Find best linear direction for X prediction (centered)
    if method_config and method_config.get('name') == 'linear':
        x_model = LinearRegression()
    else:
        alpha = method_config.get('alpha', 10.0) if method_config else 10.0
        x_model = Ridge(alpha=alpha)

    x_model.fit(train_repr_scaled, x_train_centered)
    x_direction = x_model.coef_ / np.linalg.norm(x_model.coef_)  # Normalize

    # 2. Find best linear direction for Y prediction (centered)
    if method_config and method_config.get('name') == 'linear':
        y_model = LinearRegression()
    else:
        alpha = method_config.get('alpha', 10.0) if method_config else 10.0
        y_model = Ridge(alpha=alpha)

    y_model.fit(train_repr_scaled, y_train_centered)
    y_direction = y_model.coef_ / np.linalg.norm(y_model.coef_)  # Normalize

    # 3. Orthogonalize Y direction with respect to X direction
    # (use Gram-Schmidt to ensure orthogonality)
    y_direction_orth = y_direction - np.dot(y_direction, x_direction) * x_direction
    y_direction_orth = y_direction_orth / np.linalg.norm(y_direction_orth)

    # 4. Project representations onto X and Y directions
    x_projections = all_repr_scaled @ x_direction
    y_projections = all_repr_scaled @ y_direction_orth

    # 5. Remove X and Y components from representations
    x_component = np.outer(x_projections, x_direction)
    y_component = np.outer(y_projections, y_direction_orth)
    residual_repr = all_repr_scaled - x_component - y_component

    # 6. Perform PCA on residual representations
    pca = PCA(n_components=1)
    pca.fit(residual_repr[:n_train])  # Fit only on training data
    pca_direction = pca.components_[0]

    # 7. Project onto PCA direction
    pca_projections = residual_repr @ pca_direction

    # 8. Calculate metrics
    # R² for X prediction (compare with centered coordinates)
    x_train_pred = train_repr_scaled @ x_direction
    x_test_pred = test_repr_scaled @ x_direction
    x_train_r2 = np.corrcoef(x_train_pred, x_train_centered)[0, 1]**2
    x_test_r2 = np.corrcoef(x_test_pred, x_test_centered)[0, 1]**2

    # R² for Y prediction (using orthogonalized direction, compare with centered)
    y_train_pred = train_repr_scaled @ y_direction_orth
    y_test_pred = test_repr_scaled @ y_direction_orth
    y_train_r2 = np.corrcoef(y_train_pred, y_train_centered)[0, 1]**2
    y_test_r2 = np.corrcoef(y_test_pred, y_test_centered)[0, 1]**2

    # Variance explained by residual PCA
    residual_var_explained = pca.explained_variance_ratio_[0]

    # Calculate total variance captured by all three components
    total_projections = np.column_stack([x_projections, y_projections, pca_projections])
    total_var_captured = np.var(total_projections) / np.var(all_repr_scaled)

    projections = {
        'x': x_projections,
        'y': y_projections,
        'pca': pca_projections,
        'combined': total_projections
    }

    components = {
        'x_direction': x_direction,
        'y_direction_orth': y_direction_orth,
        'pca_direction': pca_direction
    }

    metrics = {
        'x_train_r2': x_train_r2,
        'x_test_r2': x_test_r2,
        'y_train_r2': y_train_r2,
        'y_test_r2': y_test_r2,
        'residual_var_explained': residual_var_explained,
        'total_var_captured': total_var_captured,
        'x_y_correlation': np.corrcoef(x_projections, y_projections)[0, 1],
        'x_pca_correlation': np.corrcoef(x_projections, pca_projections)[0, 1],
        'y_pca_correlation': np.corrcoef(y_projections, pca_projections)[0, 1]
    }

    return projections, components, metrics


def create_3d_projection_plot(projections, coordinates, city_info, metrics,
                              output_path, title_suffix=""):
    """
    Create 3D plot with X/Y directions and residual PCA.

    Args:
        projections: Dictionary with 'x', 'y', 'pca' projections
        coordinates: True x, y coordinates (for coloring/reference)
        city_info: List of city information dictionaries
        metrics: Dictionary with performance metrics
        output_path: Path to save the HTML plot
        title_suffix: Additional text for the plot title
    """
    x_proj = projections['x']
    y_proj = projections['y']
    z_proj = projections['pca']

    # Get regions for cities
    regions = []
    city_names = []
    for city in city_info:
        region = city.get('region', 'Unknown')
        regions.append(region)
        city_names.append(city.get('name', 'Unknown'))

    # Get unique regions and assign colors
    unique_regions = sorted(set(regions))
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    region_to_color = {region: colors[i % len(colors)] for i, region in enumerate(unique_regions)}

    # Create figure
    fig = go.Figure()

    # Plot each region separately (TEST CITIES ONLY)
    for region in unique_regions:
        # Get indices for this region
        region_mask = np.array([r == region for r in regions])
        if np.sum(region_mask) == 0:
            continue

        # Get data for this region
        region_x = x_proj[region_mask]
        region_y = y_proj[region_mask]
        region_z = z_proj[region_mask]
        region_names = [city_names[i] for i, m in enumerate(region_mask) if m]
        region_true_x = coordinates[region_mask, 0]
        region_true_y = coordinates[region_mask, 1]

        # Determine if this is train or test data
        region_splits = []
        for i, m in enumerate(region_mask):
            if m:
                split = city_info[i].get('split', 'unknown')
                region_splits.append(split)

        # Filter to TEST ONLY
        test_mask = np.array([s == 'test' for s in region_splits])
        if np.sum(test_mask) == 0:
            continue

        test_x = region_x[test_mask]
        test_y = region_y[test_mask]
        test_z = region_z[test_mask]
        test_names = [region_names[i] for i, m in enumerate(test_mask) if m]
        test_true_x = region_true_x[test_mask]
        test_true_y = region_true_y[test_mask]

        # Add hover text for test cities
        hover_text = []
        for i, name in enumerate(test_names):
            hover_text.append(
                f"{name}<br>"
                f"Region: {region}<br>"
                f"True coords: ({test_true_x[i]:.0f}, {test_true_y[i]:.0f})<br>"
                f"X proj: {test_x[i]:.2f}<br>"
                f"Y proj: {test_y[i]:.2f}<br>"
                f"Residual PC1: {test_z[i]:.2f}"
            )

        # Add trace for test cities in this region
        fig.add_trace(go.Scatter3d(
            x=test_x,
            y=test_y,
            z=test_z,
            mode='markers',
            name=region,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                size=7,
                color=region_to_color[region],
                opacity=0.8,
                symbol='circle'  # All test cities as circles
            )
        ))

    # Reference plane removed - always disabled

    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f'3D Projection: X/Y Directions + Residual PC1{title_suffix}<br>'
                f'<sub>X R²: {metrics["x_test_r2"]:.3f} | Y R²: {metrics["y_test_r2"]:.3f} | '
                f'Residual Var: {metrics["residual_var_explained"]:.1%} | '
                f'Total Var: {metrics["total_var_captured"]:.1%}</sub>'
            ),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X Direction Projection',
            yaxis_title='Y Direction Projection (orthogonalized)',
            zaxis_title='Residual PC1',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        width=1200,
        height=800,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    # Save plot
    fig.write_html(output_path)
    print(f"Saved 3D plot to: {output_path}")

    return fig


def main(config_path, overwrite=False, debug=False):
    """Main function to create 3D visualization with X/Y directions and residual PCA."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config - fail fast!
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")
    if 'representation_path' not in config:
        raise ValueError("FATAL: 'representation_path' required in config")
    if 'cities_csv' not in config:
        raise ValueError("FATAL: 'cities_csv' required in config")

    # Initialize output directory with safety checks
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Copy config to output directory for reproducibility
    shutil.copy(config_path, output_dir / 'config.yaml')

    # Get parameters from config
    repr_path = Path(config['representation_path'])
    cities_csv = Path(config['cities_csv'])
    method_config = config.get('method', None)  # Probe method configuration
    token_index = config.get('token_index', 1)  # Default to 'c' token
    layer_index = config.get('layer_index', -1)  # Default to last layer
    standardize = config.get('standardize', True)  # Whether to standardize before PCA

    print("="*60)
    print("3D Visualization: X/Y Directions + Residual PCA")
    print("="*60)
    print(f"Representation path: {repr_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cities CSV: {cities_csv}")
    print(f"Standardize features: {standardize}")

    # Print probe method configuration
    if method_config:
        print(f"Probe method: {method_config.get('name', 'ridge')}")
        if method_config.get('name') in ['ridge', 'lasso']:
            print(f"  Alpha: {method_config.get('alpha', 10.0)}")
    else:
        print("Probe method: ridge (default)")
        print("  Alpha: 10.0")

    # Load representations and metadata
    repr_file = repr_path / 'representations.pt'
    meta_file = repr_path / 'metadata.json'

    if not repr_file.exists():
        raise ValueError(f"FATAL: Representations file not found: {repr_file}")
    if not meta_file.exists():
        raise ValueError(f"FATAL: Metadata file not found: {meta_file}")

    # Load data
    print("\nLoading representations...")
    data = torch.load(repr_file)
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    # Get representations
    if 'representations' in data:
        representations = data['representations'].numpy()

        # Check if it's the new 4D format or old 2D format
        if len(representations.shape) == 4:
            # New format: (n_cities, n_tokens, n_layers, hidden_dim)
            n_cities, n_tokens, n_layers, hidden_dim = representations.shape

            # Extract specific token-layer combination
            if token_index == -1 and layer_index == -1:
                # Concatenate all tokens and layers
                selected_repr = representations.reshape(n_cities, -1)
                print(f"Using ALL {n_tokens} tokens and {n_layers} layers (concatenated)")
            elif token_index == -1:
                # Concatenate all tokens for specific layer
                selected_repr = representations[:, :, layer_index, :].reshape(n_cities, -1)
                print(f"Using ALL {n_tokens} tokens, layer {layer_index} (concatenated)")
            elif layer_index == -1:
                # Concatenate all layers for specific token
                if token_index >= n_tokens:
                    print(f"Warning: token_index {token_index} >= n_tokens {n_tokens}, using last token")
                    token_index = n_tokens - 1
                selected_repr = representations[:, token_index, :, :].reshape(n_cities, -1)
                print(f"Using token {token_index} (of {n_tokens}), ALL {n_layers} layers (concatenated)")
            else:
                # Select specific token and layer
                if token_index >= n_tokens:
                    print(f"Warning: token_index {token_index} >= n_tokens {n_tokens}, using last token")
                    token_index = n_tokens - 1
                selected_repr = representations[:, token_index, layer_index, :]
                print(f"Using token {token_index} (of {n_tokens}), layer {layer_index}")
        else:
            # Old 2D format
            selected_repr = representations
            print(f"Using pre-flattened representations (old format)")

        print(f"Representation shape: {selected_repr.shape}")

    else:
        # Old format: try flat representations
        if 'representations_flat' in data:
            selected_repr = data['representations_flat'].numpy()
            print("Using flat representations (old format)")
            print(f"Representation shape: {selected_repr.shape}")
        else:
            raise ValueError("FATAL: Could not find representations in the loaded file")

    # Get city info and training split info
    city_info = metadata['city_info']
    n_train = metadata['n_train_cities']
    n_test = metadata['n_test_cities']

    print(f"\nDataset split:")
    print(f"  Training cities: {n_train}")
    print(f"  Test cities: {n_test}")
    print(f"  Total cities: {len(city_info)}")

    # Load cities CSV to get coordinates
    cities_df = pd.read_csv(cities_csv)

    # Create mapping from city_id to coordinates
    city_id_to_coords = {
        row['city_id']: (row['x'], row['y'])
        for _, row in cities_df.iterrows()
    }

    # Get coordinates for our cities
    coordinates = []
    for city in city_info:
        city_id = city.get('row_id', city.get('city_id'))
        if city_id in city_id_to_coords:
            coordinates.append(city_id_to_coords[city_id])
        else:
            # Try to match by name if ID doesn't work
            city_name = city.get('name')
            matched = cities_df[cities_df['asciiname'] == city_name]
            if len(matched) > 0:
                coordinates.append((matched.iloc[0]['x'], matched.iloc[0]['y']))
            else:
                print(f"Warning: Could not find coordinates for city {city_name} (ID: {city_id})")
                coordinates.append((0, 0))  # Default coordinates

    coordinates = np.array(coordinates)

    # Add split information to city info
    for i in range(n_train):
        city_info[i]['split'] = 'train'
    for i in range(n_train, len(city_info)):
        city_info[i]['split'] = 'test'

    # Compute projections and components
    print("\nComputing X/Y directions and residual PCA...")
    projections, components, metrics = compute_xy_directions_and_residual_pca(
        selected_repr, coordinates, n_train, method_config, standardize
    )

    # Print metrics
    print(f"\nX coordinate prediction:")
    print(f"  Train R²: {metrics['x_train_r2']:.4f}")
    print(f"  Test R²: {metrics['x_test_r2']:.4f}")

    print(f"\nY coordinate prediction:")
    print(f"  Train R²: {metrics['y_train_r2']:.4f}")
    print(f"  Test R²: {metrics['y_test_r2']:.4f}")

    print(f"\nResidual analysis:")
    print(f"  Variance explained by residual PC1: {metrics['residual_var_explained']:.1%}")
    print(f"  Total variance captured by 3 components: {metrics['total_var_captured']:.1%}")

    print(f"\nComponent correlations:")
    print(f"  X-Y correlation: {metrics['x_y_correlation']:.3f}")
    print(f"  X-PCA correlation: {metrics['x_pca_correlation']:.3f}")
    print(f"  Y-PCA correlation: {metrics['y_pca_correlation']:.3f}")

    # Create visualization
    print("\nCreating 3D visualization...")
    output_path = output_dir / 'xy_residual_pca_3d.html'

    create_3d_projection_plot(
        projections, coordinates, city_info, metrics,
        output_path,
        title_suffix=""
    )

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        else:
            return obj

    # Save numerical results
    results = {
        'metrics': convert_to_native(metrics),
        'config': config,
        'n_train': n_train,
        'n_test': n_test,
        'representation_shape': list(selected_repr.shape),
        'token_index': token_index,
        'layer_index': layer_index,
        'standardize': standardize
    }

    # Save component directions for reproducibility
    np.save(output_dir / 'x_direction.npy', components['x_direction'])
    np.save(output_dir / 'y_direction_orth.npy', components['y_direction_orth'])
    np.save(output_dir / 'pca_direction.npy', components['pca_direction'])

    # Save projections
    np.save(output_dir / 'projections.npy', projections['combined'])

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("Visualization Complete")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"  - 3D plot: xy_residual_pca_3d.html")
    print(f"  - Results: results.json")
    print(f"  - Component directions: *_direction.npy files")
    print(f"  - Projections: projections.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create 3D visualization with X/Y directions and residual PCA')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')

    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)