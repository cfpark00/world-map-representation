#!/usr/bin/env python3
"""
Create 3D visualization of city location predictions with error magnitude.

This script loads saved representations, trains linear probes for x/y coordinates,
and creates an interactive 3D plot where:
- X axis: Predicted x coordinate
- Y axis: Predicted y coordinate
- Z axis: Prediction error magnitude (Euclidean distance from true location)

Colors represent different geographic regions.

Usage:
    python visualize_prediction_error_3d.py configs/analysis/pred_error_3d.yaml
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
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory


def train_coordinate_probes(representations, coordinates, n_train, method_config=None):
    """
    Train linear probes to predict x and y coordinates.

    Args:
        representations: Array of shape (n_cities, feature_dim)
        coordinates: Array of shape (n_cities, 2) with x, y coordinates
        n_train: Number of training samples
        method_config: Dictionary with probe method configuration (same as analyze_representations.py)

    Returns:
        x_probe, y_probe: Trained probe models
        predictions: Dictionary with train/test predictions
        metrics: Dictionary with R² scores
    """
    # Split data
    train_repr = representations[:n_train]
    test_repr = representations[n_train:]
    train_coords = coordinates[:n_train]
    test_coords = coordinates[n_train:]

    # Create probes based on method configuration (same logic as analyze_representations.py)
    if method_config is None:
        # Default to Ridge with alpha=10.0
        x_probe = Ridge(alpha=10.0)
        y_probe = Ridge(alpha=10.0)
    else:
        method_name = method_config.get('name', 'ridge')

        if method_name == 'linear':
            x_probe = LinearRegression()
            y_probe = LinearRegression()
        elif method_name == 'lasso':
            from sklearn.linear_model import Lasso
            alpha = method_config.get('alpha', 1.0)
            max_iter = method_config.get('max_iter', 1000)
            tol = method_config.get('tol', 0.0001)
            x_probe = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
            y_probe = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
        else:  # ridge
            alpha = method_config.get('alpha', 10.0)
            solver = method_config.get('solver', 'auto')
            max_iter = method_config.get('max_iter', None)
            tol = method_config.get('tol', 0.0001)
            x_probe = Ridge(alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)
            y_probe = Ridge(alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)

    # Train probes
    x_probe.fit(train_repr, train_coords[:, 0])
    y_probe.fit(train_repr, train_coords[:, 1])

    # Make predictions
    x_train_pred = x_probe.predict(train_repr)
    y_train_pred = y_probe.predict(train_repr)
    x_test_pred = x_probe.predict(test_repr)
    y_test_pred = y_probe.predict(test_repr)

    # Calculate R² scores
    x_train_r2 = r2_score(train_coords[:, 0], x_train_pred)
    y_train_r2 = r2_score(train_coords[:, 1], y_train_pred)
    x_test_r2 = r2_score(test_coords[:, 0], x_test_pred)
    y_test_r2 = r2_score(test_coords[:, 1], y_test_pred)

    predictions = {
        'x_train': x_train_pred,
        'y_train': y_train_pred,
        'x_test': x_test_pred,
        'y_test': y_test_pred
    }

    metrics = {
        'x_train_r2': x_train_r2,
        'y_train_r2': y_train_r2,
        'x_test_r2': x_test_r2,
        'y_test_r2': y_test_r2
    }

    return x_probe, y_probe, predictions, metrics


def create_3d_error_plot(x_pred, y_pred, x_true, y_true, city_info,
                         metrics, output_path, title_suffix=""):
    """
    Create 3D plot with predicted coordinates and error magnitude.

    Args:
        x_pred: Predicted x coordinates
        y_pred: Predicted y coordinates
        x_true: True x coordinates
        y_true: True y coordinates
        city_info: List of city information dictionaries
        metrics: Dictionary with R² scores
        output_path: Path to save the HTML plot
        title_suffix: Additional text for the plot title
    """
    # Calculate prediction errors and log transform
    errors = np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)
    # Use log(1 + error) to avoid log(0) and keep relative differences
    log_errors = np.log1p(errors)

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

    # Plot each region separately
    for region in unique_regions:
        # Get indices for this region
        region_mask = np.array([r == region for r in regions])
        if np.sum(region_mask) == 0:
            continue

        # Get data for this region
        region_x_pred = x_pred[region_mask]
        region_y_pred = y_pred[region_mask]
        region_errors = errors[region_mask]
        region_log_errors = log_errors[region_mask]
        region_names = [city_names[i] for i, m in enumerate(region_mask) if m]

        # Add hover text
        hover_text = []
        for i, name in enumerate(region_names):
            hover_text.append(
                f"{name}<br>"
                f"Region: {region}<br>"
                f"Predicted: ({region_x_pred[i]:.1f}, {region_y_pred[i]:.1f})<br>"
                f"Error: {region_errors[i]:.1f}<br>"
                f"Log(1+Error): {region_log_errors[i]:.2f}"
            )

        # Add trace for this region
        fig.add_trace(go.Scatter3d(
            x=region_x_pred,
            y=region_y_pred,
            z=region_log_errors,  # Use log-transformed errors
            mode='markers',
            name=region,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                size=5,
                color=region_to_color[region],
                opacity=0.7
            )
        ))

    # Add a reference plane at z=0 (perfect predictions)
    x_range = [x_pred.min(), x_pred.max()]
    y_range = [y_pred.min(), y_pred.max()]

    # Create grid for reference plane
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 10),
        np.linspace(y_range[0], y_range[1], 10)
    )
    zz = np.zeros_like(xx)

    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        opacity=0.2,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        showscale=False,
        name='Perfect Predictions',
        hovertemplate='Perfect prediction plane<extra></extra>'
    ))

    # Calculate statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    mean_log_error = np.mean(log_errors)

    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f'3D City Location Predictions with Log Error{title_suffix}<br>'
                f'<sub>X R²: {metrics["x_test_r2"]:.3f} | Y R²: {metrics["y_test_r2"]:.3f} | '
                f'Mean Error: {mean_error:.1f} | Median Error: {median_error:.1f} | Max Error: {max_error:.1f}</sub>'
            ),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='Predicted X Coordinate',
            yaxis_title='Predicted Y Coordinate',
            zaxis_title='Log(1 + Prediction Error)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
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
    """Main function to create 3D prediction error visualization."""

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
    method_config = config.get('method', None)  # Probe method configuration (same as analyze_representations.py)
    token_index = config.get('token_index', 1)  # Default to 'c' token
    layer_index = config.get('layer_index', -1)  # Default to last layer

    print("="*60)
    print("3D Prediction Error Visualization")
    print("="*60)
    print(f"Representation path: {repr_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cities CSV: {cities_csv}")

    # Print probe method configuration (same format as analyze_representations.py)
    if method_config:
        print(f"Probe method: {method_config.get('name', 'ridge')}")
        if method_config.get('name') in ['ridge', 'lasso']:
            print(f"  Alpha: {method_config.get('alpha', 10.0 if method_config.get('name') == 'ridge' else 1.0)}")
        if method_config.get('solver'):
            print(f"  Solver: {method_config.get('solver')}")
        if method_config.get('max_iter'):
            print(f"  Max iterations: {method_config.get('max_iter')}")
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

            # Extract specific token-layer combination or concatenate
            if token_index == -1 and layer_index == -1:
                # Concatenate all tokens and layers
                selected_repr = representations.reshape(n_cities, -1)  # Flatten to (n_cities, n_tokens*n_layers*hidden_dim)
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
            # Old 2D format: (n_cities, features) - already flattened
            selected_repr = representations
            print(f"Using pre-flattened representations (old format)")

        print(f"Representation shape: {selected_repr.shape}")

    else:
        # Old format: try flat representations
        if 'representations_flat' in data:
            selected_repr = data['representations_flat'].numpy()
            # For old format, we can't select specific token-layer
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

    # Train probes
    method_name = method_config.get('name', 'ridge') if method_config else 'ridge'
    print(f"\nTraining {method_name} probes...")
    x_probe, y_probe, predictions, metrics = train_coordinate_probes(
        selected_repr, coordinates, n_train, method_config
    )

    # Print metrics
    print(f"\nTraining performance:")
    print(f"  X R²: {metrics['x_train_r2']:.4f}")
    print(f"  Y R²: {metrics['y_train_r2']:.4f}")
    print(f"\nTest performance:")
    print(f"  X R²: {metrics['x_test_r2']:.4f}")
    print(f"  Y R²: {metrics['y_test_r2']:.4f}")

    # Create 3D plots for both train and test sets
    print("\nCreating 3D visualizations...")

    # Test set plot
    test_output_path = output_dir / 'prediction_error_3d_test.html'
    test_city_info = city_info[n_train:]
    test_coords = coordinates[n_train:]

    create_3d_error_plot(
        predictions['x_test'], predictions['y_test'],
        test_coords[:, 0], test_coords[:, 1],
        test_city_info, metrics,
        test_output_path,
        title_suffix=" (Test Set)"
    )

    # Training set plot (optional, based on config)
    if config.get('include_train_plot', True):
        train_output_path = output_dir / 'prediction_error_3d_train.html'
        train_city_info = city_info[:n_train]
        train_coords = coordinates[:n_train]

        # Update metrics for training set
        train_metrics = {
            'x_test_r2': metrics['x_train_r2'],  # Use train R² for display
            'y_test_r2': metrics['y_train_r2']
        }

        create_3d_error_plot(
            predictions['x_train'], predictions['y_train'],
            train_coords[:, 0], train_coords[:, 1],
            train_city_info, train_metrics,
            train_output_path,
            title_suffix=" (Training Set)"
        )

    # Combined plot (both train and test)
    if config.get('include_combined_plot', True):
        combined_output_path = output_dir / 'prediction_error_3d_combined.html'

        # Combine predictions
        all_x_pred = np.concatenate([predictions['x_train'], predictions['x_test']])
        all_y_pred = np.concatenate([predictions['y_train'], predictions['y_test']])

        # Add split information to city info
        for i in range(n_train):
            city_info[i]['split'] = 'train'
        for i in range(n_train, len(city_info)):
            city_info[i]['split'] = 'test'

        create_3d_error_plot(
            all_x_pred, all_y_pred,
            coordinates[:, 0], coordinates[:, 1],
            city_info, metrics,
            combined_output_path,
            title_suffix=" (Train + Test)"
        )

    # Save numerical results
    results = {
        'metrics': metrics,
        'config': config,
        'n_train': n_train,
        'n_test': n_test,
        'representation_shape': list(selected_repr.shape),
        'token_index': token_index,
        'layer_index': layer_index
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("Visualization Complete")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"  - Test set plot: prediction_error_3d_test.html")
    if config.get('include_train_plot', True):
        print(f"  - Training set plot: prediction_error_3d_train.html")
    if config.get('include_combined_plot', True):
        print(f"  - Combined plot: prediction_error_3d_combined.html")
    print(f"  - Results: results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create 3D prediction error visualization')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')

    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)