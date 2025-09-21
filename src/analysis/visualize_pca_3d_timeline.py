#!/usr/bin/env python3
"""
Create interactive 3D PCA visualization with timeline slider for multiple checkpoints.

This script loads representations from all available checkpoints and creates
an interactive 3D plot with a slider to navigate through training time.

Usage:
    python visualize_pca_3d_timeline.py configs/analysis/pca_3d_timeline.yaml
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
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import re

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory


def load_all_checkpoints(base_path):
    """
    Load representations from all checkpoint directories.

    Args:
        base_path: Path to the representations directory

    Returns:
        List of (step, representations, metadata) tuples sorted by step
    """
    base_path = Path(base_path)
    checkpoints = []

    # Find all checkpoint directories
    for checkpoint_dir in base_path.glob('checkpoint-*'):
        # Extract step number from directory name
        match = re.match(r'checkpoint-(\d+)', checkpoint_dir.name)
        if not match:
            continue

        step = int(match.group(1))

        # Load representations
        repr_path = checkpoint_dir / 'representations.pt'
        metadata_path = checkpoint_dir / 'metadata.json'

        if not repr_path.exists() or not metadata_path.exists():
            print(f"Warning: Missing files in {checkpoint_dir.name}, skipping")
            continue

        # Load data
        repr_data = torch.load(repr_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        checkpoints.append((step, repr_data, metadata))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])

    return checkpoints


def extract_representations(repr_data, token_index=-1, layer_index=-1):
    """
    Extract specific token and layer representations.

    Args:
        repr_data: Dictionary with 'representations' tensor
        token_index: Which token to use (-1 for all)
        layer_index: Which layer to use (-1 for all)

    Returns:
        2D array of shape (n_cities, features)
    """
    # Get the full representations tensor
    if isinstance(repr_data, dict):
        representations = repr_data['representations']
        if isinstance(representations, torch.Tensor):
            representations = representations.numpy()
    else:
        representations = repr_data.numpy() if isinstance(repr_data, torch.Tensor) else repr_data

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

    return representations


def create_pca_timeline_plot(checkpoints, token_index, layer_index, n_components,
                            output_path, axis_mapping=None, test_only=False, marker_size=4,
                            cities_df=None):
    """
    Create interactive 3D PCA plot with timeline slider.

    Uses PCA fitted on the LAST checkpoint to project all checkpoints,
    providing a consistent coordinate system to see evolution.

    Args:
        checkpoints: List of (step, repr_data, metadata) tuples
        token_index: Which token to use (-1 for all)
        layer_index: Which layer to use (-1 for all)
        n_components: Number of PCA components
        output_path: Path to save HTML plot
        axis_mapping: Dict with 'type' and axis mappings:
            - type: 'pca' - use PCA components (default)
            - type: 'mixed' - use x/y regression directions and residual PC
        test_only: If True, only show test cities
        marker_size: Size of the markers in the plot (default 4)
        cities_df: DataFrame with city coordinates (required for 'mixed' mode)
    """
    if not checkpoints:
        raise ValueError("No checkpoints found!")

    # Get city info from first checkpoint (should be same for all)
    _, _, first_metadata = checkpoints[0]
    city_info = first_metadata['city_info']
    n_train = first_metadata['n_train_cities']

    # Determine projection mode and axes
    projection_type = 'pca'  # default
    if axis_mapping and isinstance(axis_mapping, dict):
        projection_type = axis_mapping.get('type', 'pca')

    if projection_type == 'mixed' and cities_df is None:
        raise ValueError("cities_df required for mixed projection mode!")

    # Set up axis indices based on type
    if projection_type == 'pca':
        # Original PCA mode
        if axis_mapping and isinstance(axis_mapping, dict):
            x_pc = axis_mapping.get(1, 0)  # Default to PC1
            y_pc = axis_mapping.get(2, 1)  # Default to PC2
            z_pc = axis_mapping.get(3, 2)  # Default to PC3
            int_values = [v for k, v in axis_mapping.items() if isinstance(v, int)]
            if int_values:
                n_components = max(n_components, max(int_values) + 1)
        else:
            x_pc, y_pc, z_pc = 0, 1, 2
    else:
        # Mixed mode will be handled separately
        x_pc, y_pc, z_pc = 0, 1, 2  # Will be overridden

    # Filter cities if test_only
    if test_only:
        city_info = city_info[n_train:]
        city_indices = list(range(n_train, len(first_metadata['city_info'])))
    else:
        city_indices = list(range(len(city_info)))

    # Get unique regions and assign colors
    regions = sorted(set(c.get("region") for c in city_info if c.get("region") is not None))
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    region_to_color = {region: colors[i % len(colors)] for i, region in enumerate(regions)}

    # Get last checkpoint representations
    print("Loading final checkpoint...")
    last_step, last_repr_data, last_metadata = checkpoints[-1]
    last_representations = extract_representations(last_repr_data, token_index, layer_index)

    # Filter if test_only
    if test_only:
        last_representations = last_representations[n_train:]

    if projection_type == 'mixed':
        # MIXED MODE: Use x/y regression directions + residual PC
        print("Computing mixed projection (x/y regression + residual PC)...")

        # Get city coordinates from the dataframe
        # Match city IDs from metadata to dataframe
        city_ids = [c['row_id'] for c in city_info]

        # Get coordinates for these cities
        import numpy as np
        from sklearn.linear_model import LinearRegression

        x_coords = []
        y_coords = []
        for cid in city_ids:
            city_row = cities_df[cities_df['city_id'] == cid]
            if len(city_row) > 0:
                x_coords.append(city_row.iloc[0]['x'])
                y_coords.append(city_row.iloc[0]['y'])
            else:
                # Fallback to city_info if not found
                matching = [c for c in city_info if c['row_id'] == cid]
                if matching:
                    x_coords.append(matching[0]['x'])
                    y_coords.append(matching[0]['y'])
                else:
                    raise ValueError(f"Could not find coordinates for city {cid}")

        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        # Fit linear regression for X coordinate
        x_model = LinearRegression()
        x_model.fit(last_representations, x_coords)
        x_direction = x_model.coef_ / np.linalg.norm(x_model.coef_)  # Normalize

        # Fit linear regression for Y coordinate
        y_model = LinearRegression()
        y_model.fit(last_representations, y_coords)
        y_direction = y_model.coef_ / np.linalg.norm(y_model.coef_)  # Normalize

        # Project out x and y directions to get residual
        # Stack the two directions (row-wise)
        U = np.vstack([x_direction, y_direction])  # (2, d)

        # Project out both directions simultaneously
        # Since x and y directions may not be orthogonal, use general projection formula
        Ginv = np.linalg.pinv(U @ U.T)  # (2, 2) - handles non-orthonormal directions
        coeffs = (last_representations @ U.T) @ Ginv  # (n, 2)
        projection = coeffs @ U  # (n, d) - projection onto span(U)
        residual_repr = last_representations - projection  # residual with both directions removed

        # Fit PCA on residual to get top PC
        residual_pca = PCA(n_components=1)
        residual_pca.fit(residual_repr)
        z_direction = residual_pca.components_[0]

        # Create projection matrix from these 3 directions
        projection_matrix = np.vstack([x_direction, y_direction, z_direction])

        print(f"X direction R²: {x_model.score(last_representations, x_coords):.3f}")
        print(f"Y direction R²: {y_model.score(last_representations, y_coords):.3f}")
        print(f"Residual PC variance: {residual_pca.explained_variance_ratio_[0]:.1%}")

        variance_explained = [
            x_model.score(last_representations, x_coords),
            y_model.score(last_representations, y_coords),
            residual_pca.explained_variance_ratio_[0]
        ]
    else:
        # ORIGINAL PCA MODE
        print("Fitting PCA on final checkpoint...")
        pca = PCA(n_components=n_components)
        pca.fit(last_representations)
        variance_explained = pca.explained_variance_ratio_
        projection_matrix = pca.components_[:3]  # Use first 3 PCs

        print(f"PCA fitted on step {last_step}")
        print(f"Variance explained: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}, PC3={variance_explained[2]:.1%}")

    # PROJECT ALL CHECKPOINTS using the same projection
    all_frames_data = []

    for step, repr_data, metadata in checkpoints:
        # Extract representations
        representations = extract_representations(repr_data, token_index, layer_index)

        # Filter if test_only
        if test_only:
            representations = representations[n_train:]

        # Project using the fixed projection matrix
        repr_projected = representations @ projection_matrix.T  # (n_cities, 3)

        # Store frame data
        frame_data = {
            'step': step,
            'repr_projected': repr_projected,
            'variance_explained': variance_explained  # Same for all frames now
        }
        all_frames_data.append(frame_data)

    # Create figure with first frame
    fig = go.Figure()

    # Add traces for each region (for first frame)
    first_frame = all_frames_data[0]
    for region in regions:
        # Get indices for this region
        region_mask = [i for i, c in enumerate(city_info) if c.get('region') == region]

        if not region_mask:
            continue

        # Add scatter trace
        fig.add_trace(go.Scatter3d(
            x=first_frame['repr_projected'][region_mask, 0],  # Always use index 0,1,2 for mixed mode
            y=first_frame['repr_projected'][region_mask, 1],
            z=first_frame['repr_projected'][region_mask, 2],
            mode='markers',
            name=region,
            marker=dict(
                size=marker_size,
                color=region_to_color[region],
                opacity=0.8
            ),
            text=[city_info[i]['name'] for i in region_mask],
            hovertemplate='<b>%{text}</b><br>' +
                         f'Region: {region}<br>' +
                         ('X-pred: %{x:.2f}<br>' if projection_type == 'mixed' else f'PC{x_pc+1}: %{{x:.2f}}<br>') +
                         ('Y-pred: %{y:.2f}<br>' if projection_type == 'mixed' else f'PC{y_pc+1}: %{{y:.2f}}<br>') +
                         ('Residual: %{z:.2f}<br>' if projection_type == 'mixed' else f'PC{z_pc+1}: %{{z:.2f}}<br>') +
                         '<extra></extra>'
        ))

    # Create frames for animation
    frames = []
    for frame_data in all_frames_data:
        frame_traces = []
        for region in regions:
            region_mask = [i for i, c in enumerate(city_info) if c.get('region') == region]

            if not region_mask:
                continue

            frame_traces.append(go.Scatter3d(
                x=frame_data['repr_projected'][region_mask, 0],
                y=frame_data['repr_projected'][region_mask, 1],
                z=frame_data['repr_projected'][region_mask, 2],
                mode='markers',
                marker=dict(size=marker_size, color=region_to_color[region], opacity=0.8),
                text=[city_info[i]['name'] for i in region_mask],
                name=region
            ))

        if projection_type == 'mixed':
            frame_title = f"3D Mixed Projection - Step {frame_data['step']:,}<br>" + \
                         f"<sub>Projections fitted on final checkpoint (step {checkpoints[-1][0]:,})<br>" + \
                         f"X-coord R²={frame_data['variance_explained'][0]:.3f}, " + \
                         f"Y-coord R²={frame_data['variance_explained'][1]:.3f}, " + \
                         f"Residual variance={frame_data['variance_explained'][2]:.1%}</sub>"
        else:
            frame_title = f"3D PCA Evolution - Step {frame_data['step']:,}<br>" + \
                         f"<sub>PCA fitted on final checkpoint (step {checkpoints[-1][0]:,})<br>" + \
                         f"Variance explained: PC{x_pc+1}={frame_data['variance_explained'][x_pc]:.1%}, " + \
                         f"PC{y_pc+1}={frame_data['variance_explained'][y_pc]:.1%}, " + \
                         f"PC{z_pc+1}={frame_data['variance_explained'][z_pc]:.1%}</sub>"

        frames.append(go.Frame(
            data=frame_traces,
            name=str(frame_data['step']),
            layout=go.Layout(
                title=dict(
                    text=frame_title
                )
            )
        ))

    fig.frames = frames

    # Create slider
    steps = []
    for i, frame_data in enumerate(all_frames_data):
        step = dict(
            method="animate",
            args=[[str(frame_data['step'])],
                  {"frame": {"duration": 0, "redraw": True},
                   "mode": "immediate",
                   "transition": {"duration": 0}}],
            label=f"{frame_data['step']:,}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        yanchor="top",
        y=0.0,
        xanchor="left",
        x=0.1,
        transition={"duration": 0},
        pad={"b": 10, "t": 50},
        len=0.8,
        currentvalue={
            "font": {"size": 16},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "center"
        },
        steps=steps
    )]

    # Update layout
    first_var = all_frames_data[0]['variance_explained']

    if projection_type == 'mixed':
        title_text = f"3D Mixed Projection - Step {all_frames_data[0]['step']:,}<br>" + \
                    f"<sub>Projections fitted on final checkpoint (step {checkpoints[-1][0]:,})<br>" + \
                    f"X-coord R²={first_var[0]:.3f}, Y-coord R²={first_var[1]:.3f}, " + \
                    f"Residual variance={first_var[2]:.1%}</sub>"
        x_label = 'X-coordinate prediction'
        y_label = 'Y-coordinate prediction'
        z_label = 'Residual PC1'
    else:
        title_text = f"3D PCA Evolution - Step {all_frames_data[0]['step']:,}<br>" + \
                    f"<sub>PCA fitted on final checkpoint (step {checkpoints[-1][0]:,})<br>" + \
                    f"Variance explained: PC{x_pc+1}={first_var[x_pc]:.1%}, " + \
                    f"PC{y_pc+1}={first_var[y_pc]:.1%}, " + \
                    f"PC{z_pc+1}={first_var[z_pc]:.1%}</sub>"
        x_label = f'PC{x_pc+1}'
        y_label = f'PC{y_pc+1}'
        z_label = f'PC{z_pc+1}'

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        sliders=sliders,
        width=1200,
        height=800
    )

    # Add play/pause buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.0,
                x=0.0,
                xanchor="left",
                yanchor="top",
                pad={"r": 10, "t": 50},
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate"}]
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}]
                    )
                ]
            )
        ]
    )

    # Save plot with autoplay disabled
    fig.write_html(output_path, auto_play=False)
    print(f"Interactive timeline plot saved to {output_path}")

    return len(checkpoints)


def main():
    parser = argparse.ArgumentParser(description='Create 3D PCA visualization timeline')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')

    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract config values
    output_dir = Path(config['output_dir'])
    representations_base_path = Path(config['representations_base_path'])
    token_index = config.get('token_index', -1)
    layer_index = config.get('layer_index', -1)
    n_components = config.get('n_components', 3)
    axis_mapping = config.get('axis_mapping', None)
    test_only = config.get('test_only', False)
    marker_size = config.get('marker_size', 4)  # Default to 4 if not specified

    # Check if mixed mode requires cities_csv
    cities_df = None
    if axis_mapping and isinstance(axis_mapping, dict):
        if axis_mapping.get('type') == 'mixed':
            if 'cities_csv' not in config:
                raise ValueError("ERROR: axis_mapping.type='mixed' requires 'cities_csv' path in config!")
            cities_csv = Path(config['cities_csv'])
            if not cities_csv.exists():
                raise ValueError(f"ERROR: cities_csv file not found: {cities_csv}")
            # Load cities data
            cities_df = pd.read_csv(cities_csv)
            print(f"Loaded cities data from: {cities_csv}")

    # Initialize output directory
    output_dir = init_directory(output_dir, overwrite=args.overwrite)

    # Copy config to output
    shutil.copy(args.config_path, output_dir / 'config.yaml')

    print("="*60)
    print("3D PCA Timeline Visualization")
    print("="*60)
    print(f"Loading representations from: {representations_base_path}")
    print(f"Token index: {token_index} (-1 = all)")
    print(f"Layer index: {layer_index} (-1 = all)")
    print(f"Test only: {test_only}")

    # Load all checkpoints
    print("\nLoading checkpoints...")
    checkpoints = load_all_checkpoints(representations_base_path)
    print(f"Found {len(checkpoints)} checkpoints")

    if not checkpoints:
        print("Error: No checkpoints found!")
        sys.exit(1)

    # Print checkpoint steps
    steps = [step for step, _, _ in checkpoints]
    print(f"Steps: {steps}")

    # Create timeline plot
    print("\nCreating interactive timeline plot...")
    output_path = output_dir / 'pca_3d_timeline.html'

    n_checkpoints = create_pca_timeline_plot(
        checkpoints, token_index, layer_index, n_components,
        output_path, axis_mapping, test_only, marker_size, cities_df
    )

    print(f"\nSuccessfully created timeline with {n_checkpoints} checkpoints")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()