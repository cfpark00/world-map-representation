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
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import re

# Add parent directory to path for imports
project_root = Path('')
sys.path.insert(0, str(project_root))

from src.utils import init_directory


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):  # np.float32, np.int64, etc.
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

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


def compute_projection(train_representations, test_representations, projection_type,
                      all_city_info, train_indices, test_indices, cities_df=None, axis_mapping=None, n_components=3,
                      checkpoints=None):
    """
    Compute projection matrix and metadata based on projection type.

    Returns:
        dict with keys:
            - projection_matrix: (3, d) array for projecting representations
            - axis_labels: list of 3 strings for axis labels
            - variance_info: list of 3 values for variance/R² info
            - title_prefix: string for plot title
            - subtitle_template: format string for subtitle (with {step} placeholder)
    """
    d = train_representations.shape[1]

    if projection_type == 'temporal':
        # TEMPORAL MODE: Use PCA of temporal change vectors
        if checkpoints is None or len(checkpoints) < 2:
            raise ValueError("Temporal mode requires at least 2 checkpoints!")

        print("Computing temporal PCA (change vector analysis)...")
        print(f"  Training on {len(train_indices)} cities across {len(checkpoints)} timesteps")

        # Extract representations for all timesteps for training cities
        temporal_diffs = []

        for t in range(1, len(checkpoints)):
            # Get representations at t and t-1
            _, repr_data_prev, _ = checkpoints[t-1]
            _, repr_data_curr, _ = checkpoints[t]

            # Extract all representations
            # Use the same token_index and layer_index as provided to the function
            # Need to get these from parent scope
            import inspect
            frame = inspect.currentframe().f_back.f_locals
            token_idx = frame.get('token_index', -1)
            layer_idx = frame.get('layer_index', -1)

            all_reps_prev = extract_representations(repr_data_prev, token_idx, layer_idx)
            all_reps_curr = extract_representations(repr_data_curr, token_idx, layer_idx)

            # Get training representations
            train_reps_prev = all_reps_prev[train_indices]
            train_reps_curr = all_reps_curr[train_indices]

            # Compute differences
            diff = train_reps_curr - train_reps_prev
            temporal_diffs.append(diff)

        # Stack all temporal differences
        all_diffs = np.vstack(temporal_diffs)  # (n_train * (n_timesteps-1), d)

        # Apply PCA to temporal differences
        temporal_pca = PCA(n_components=3)
        temporal_pca.fit(all_diffs)
        projection_matrix = temporal_pca.components_

        # Calculate variance explained
        variance_explained = temporal_pca.explained_variance_ratio_

        return {
            'projection_matrix': projection_matrix,
            'axis_labels': ['Temporal PC1', 'Temporal PC2', 'Temporal PC3'],
            'axis_labels_short': ['TPC1', 'TPC2', 'TPC3'],
            'variance_info': variance_explained[:3].tolist(),
            'title_prefix': '3D Temporal PCA (Change Vectors)',
            'subtitle_template': 'PCA fitted on temporal changes between checkpoints<br>' +
                               f'Variance explained: TPC1={variance_explained[0]:.1%}, ' +
                               f'TPC2={variance_explained[1]:.1%}, ' +
                               f'TPC3={variance_explained[2]:.1%}'
        }

    elif projection_type == 'mixed':
        # MIXED MODE: Use x/y regression directions + residual PC
        print("Computing mixed projection (x/y regression + residual PC)...")
        print(f"  Training on {len(train_indices)} cities")

        # Get city coordinates for TRAINING cities
        train_city_info = [all_city_info[idx] for idx in train_indices]
        train_city_ids = [c['row_id'] for c in train_city_info]

        x_coords_train = []
        y_coords_train = []
        for cid in train_city_ids:
            city_row = cities_df[cities_df['city_id'] == cid]
            if len(city_row) > 0:
                x_coords_train.append(city_row.iloc[0]['x'])
                y_coords_train.append(city_row.iloc[0]['y'])
            else:
                matching = [c for c in train_city_info if c['row_id'] == cid]
                if matching:
                    x_coords_train.append(matching[0]['x'])
                    y_coords_train.append(matching[0]['y'])
                else:
                    raise ValueError(f"Could not find coordinates for city {cid}")

        x_coords_train = np.array(x_coords_train)
        y_coords_train = np.array(y_coords_train)

        # Fit regressions
        x_model = LinearRegression()
        x_model.fit(train_representations, x_coords_train)
        x_direction = x_model.coef_ / np.linalg.norm(x_model.coef_)

        y_model = LinearRegression()
        y_model.fit(train_representations, y_coords_train)
        y_direction = y_model.coef_ / np.linalg.norm(y_model.coef_)

        # Get residual PC
        U = np.vstack([x_direction, y_direction])
        Ginv = np.linalg.pinv(U @ U.T)
        coeffs_train = (train_representations @ U.T) @ Ginv
        projection_train = coeffs_train @ U
        residual_repr_train = train_representations - projection_train

        residual_pca = PCA(n_components=1)
        residual_pca.fit(residual_repr_train)
        z_direction = residual_pca.components_[0]

        projection_matrix = np.vstack([x_direction, y_direction, z_direction])

        # Calculate test R²
        test_city_info = [all_city_info[idx] for idx in test_indices]
        test_city_ids = [c['row_id'] for c in test_city_info]
        x_coords_test = []
        y_coords_test = []
        for cid in test_city_ids:
            matching = [c for c in all_city_info if c['row_id'] == cid]
            if matching:
                x_coords_test.append(matching[0]['x'])
                y_coords_test.append(matching[0]['y'])

        x_test_r2 = x_model.score(test_representations, np.array(x_coords_test))
        y_test_r2 = y_model.score(test_representations, np.array(y_coords_test))

        return {
            'projection_matrix': projection_matrix,
            'axis_labels': ['X-coordinate prediction', 'Y-coordinate prediction', 'Residual PC1'],
            'axis_labels_short': ['X-pred', 'Y-pred', 'Residual'],
            'variance_info': [x_test_r2, y_test_r2, residual_pca.explained_variance_ratio_[0]],
            'title_prefix': '3D Mixed Projection',
            'subtitle_template': 'Projections fitted on training set from final checkpoint (step {final_step:,})<br>' +
                               f'X-coord Test R²={x_test_r2:.3f}, Y-coord Test R²={y_test_r2:.3f}, ' +
                               f'Residual variance={residual_pca.explained_variance_ratio_[0]:.1%}'
        }

    elif projection_type == 'random':
        # RANDOM MODE: Use 3 random orthonormal directions
        print("Computing random orthonormal projection...")
        print(f"  Training on {len(train_indices)} cities")

        np.random.seed(42)
        random_matrix = np.random.randn(d, 3)
        Q, R = np.linalg.qr(random_matrix)
        projection_matrix = Q[:, :3].T

        # Compute variance captured
        train_projected = train_representations @ projection_matrix.T
        total_variance = np.var(train_representations)
        projected_variance = np.var(train_projected)
        variance_captured = projected_variance / total_variance

        return {
            'projection_matrix': projection_matrix,
            'axis_labels': ['Random Axis 1', 'Random Axis 2', 'Random Axis 3'],
            'axis_labels_short': ['Random Axis 1', 'Random Axis 2', 'Random Axis 3'],
            'variance_info': [variance_captured/3, variance_captured/3, variance_captured/3],
            'title_prefix': '3D Random Projection',
            'subtitle_template': 'Random orthonormal axes generated from final checkpoint (step {final_step:,})<br>' +
                               f'Total variance captured: {variance_captured:.1%}'
        }

    else:  # PCA mode
        print("Fitting PCA on training set...")
        print(f"  Training on {len(train_indices)} cities")

        pca = PCA(n_components=max(3, n_components))
        pca.fit(train_representations)
        projection_matrix = pca.components_[:3]
        variance_explained = pca.explained_variance_ratio_

        # Handle custom PC selection
        x_pc, y_pc, z_pc = 0, 1, 2
        if axis_mapping and isinstance(axis_mapping, dict):
            x_pc = axis_mapping.get(1, 0)
            y_pc = axis_mapping.get(2, 1)
            z_pc = axis_mapping.get(3, 2)
            # Reorder projection matrix rows
            projection_matrix = pca.components_[[x_pc, y_pc, z_pc]]

        return {
            'projection_matrix': projection_matrix,
            'axis_labels': [f'PC{x_pc+1}', f'PC{y_pc+1}', f'PC{z_pc+1}'],
            'axis_labels_short': [f'PC{x_pc+1}', f'PC{y_pc+1}', f'PC{z_pc+1}'],
            'variance_info': [variance_explained[x_pc], variance_explained[y_pc], variance_explained[z_pc]],
            'title_prefix': '3D PCA Evolution',
            'subtitle_template': 'PCA fitted on training set from final checkpoint (step {final_step:,})<br>' +
                               f'Variance explained: PC{x_pc+1}={variance_explained[x_pc]:.1%}, ' +
                               f'PC{y_pc+1}={variance_explained[y_pc]:.1%}, ' +
                               f'PC{z_pc+1}={variance_explained[z_pc]:.1%}',
            'x_pc': x_pc,
            'y_pc': y_pc,
            'z_pc': z_pc
        }


def save_projection_data(output_dir, checkpoints, all_frames_data, projection_info,
                        city_info, train_indices, test_indices,
                        token_index, layer_index, axis_mapping,
                        probe_train, probe_test, train_frac):
    """
    Save all data needed to recreate the visualization.

    Args:
        output_dir: Directory to save the data
        checkpoints: List of checkpoint info
        all_frames_data: Projected data for all frames
        projection_info: Projection matrix and metadata
        city_info: Information about cities being visualized
        train_indices: Indices of training cities
        test_indices: Indices of test cities
        token_index: Token extraction index
        layer_index: Layer extraction index
        axis_mapping: Axis mapping configuration
        probe_train: Training cities filter pattern
        probe_test: Test cities filter pattern
        train_frac: Fraction of cities used for training
    """
    import pickle
    import json

    # Create data package
    data_package = {
        'checkpoints_info': [(step, metadata) for step, _, metadata in checkpoints],
        'all_frames_data': all_frames_data,
        'projection_info': projection_info,
        'city_info': city_info,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'token_index': token_index,
        'layer_index': layer_index,
        'axis_mapping': axis_mapping,
        'probe_train': probe_train,
        'probe_test': probe_test,
        'train_frac': train_frac,
        'n_checkpoints': len(checkpoints)
    }

    # Save as pickle for complete data preservation
    pickle_path = output_dir / 'projection_data.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_package, f)
    print(f"  Saved complete projection data to {pickle_path}")

    # Also save key data as numpy arrays for easier access
    import numpy as np

    # Save projection matrix
    np.save(output_dir / 'projection_matrix.npy', projection_info['projection_matrix'])

    # Save all projected coordinates as a single array
    all_projections = np.array([frame['repr_projected'] for frame in all_frames_data])
    np.save(output_dir / 'all_projections.npy', all_projections)
    print(f"  Saved projection matrix and coordinates as .npy files")

    # Save metadata as JSON for easy inspection
    metadata = {
        'n_checkpoints': len(checkpoints),
        'n_cities_visualized': len(test_indices),
        'n_cities_train': len(train_indices),
        'projection_type': axis_mapping.get('type', 'pca') if axis_mapping else 'pca',
        'token_index': token_index,
        'layer_index': layer_index,
        'axis_labels': projection_info['axis_labels'],
        'axis_labels_short': projection_info['axis_labels_short'],
        'variance_info': projection_info['variance_info'],
        'title_prefix': projection_info['title_prefix'],
        'subtitle_template': projection_info['subtitle_template'],
        'probe_train': probe_train,
        'probe_test': probe_test,
        'train_frac': train_frac,
        'checkpoint_steps': [step for step, _ in data_package['checkpoints_info']]
    }

    json_path = output_dir / 'projection_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)
    print(f"  Saved metadata to {json_path}")

    # Save city info as CSV for easy inspection
    import pandas as pd
    city_df = pd.DataFrame(city_info)
    city_df['is_train'] = False
    city_df['is_test'] = False
    for idx in range(len(city_info)):
        if idx in train_indices:
            city_df.loc[idx, 'is_train'] = True
        if idx in test_indices:
            city_df.loc[idx, 'is_test'] = True

    csv_path = output_dir / 'city_info.csv'
    city_df.to_csv(csv_path, index=False)
    print(f"  Saved city info to {csv_path}")


def create_pca_timeline_plot(checkpoints, token_index, layer_index, n_components,
                            output_path, train_frac, axis_mapping=None, marker_size=4,
                            cities_df=None, probe_train=None, probe_test=None):
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
        train_frac: Fraction of probe_train cities to use for training (required)
        axis_mapping: Dict with 'type' and axis mappings:
            - type: 'pca' - use PCA components (default)
            - type: 'mixed' - use x/y regression directions and residual PC
            - type: 'random' - use 3 random orthonormal directions
        marker_size: Size of the markers in the plot (default 4)
        cities_df: DataFrame with city coordinates (required for 'mixed' mode)
        probe_train: Regex pattern for selecting training cities (optional)
        probe_test: Regex pattern for selecting test cities to visualize (optional)
    """
    if not checkpoints:
        raise ValueError("No checkpoints found!")

    # Get city info from first checkpoint (should be same for all)
    _, _, first_metadata = checkpoints[0]
    all_city_info = first_metadata['city_info']

    # Import proper filter function from utils
    import numpy as np
    import pandas as pd
    import sys
    project_root = Path('')
    sys.path.insert(0, str(project_root))
    from src.utils import filter_dataframe_by_pattern

    # Convert city_info list to DataFrame for filtering
    city_df = pd.DataFrame(all_city_info)

    # Ensure city_id column exists (might be row_id in metadata)
    if 'row_id' in city_df.columns and 'city_id' not in city_df.columns:
        city_df['city_id'] = city_df['row_id']

    # Step 1: Filter cities by probe_train pattern
    if probe_train:
        train_candidates = filter_dataframe_by_pattern(city_df, probe_train, column_name='region')
        train_candidate_indices = train_candidates.index.tolist()
    else:
        train_candidate_indices = list(range(len(city_df)))
    print(f"Cities matching probe_train pattern: {len(train_candidate_indices)}")

    # Step 2: Sample train_frac of the filtered cities for training
    n_train_to_sample = int(len(train_candidate_indices) * train_frac)
    if n_train_to_sample == 0:
        raise ValueError(f"train_frac {train_frac} results in 0 training cities!")

    # Set random seed for reproducibility
    np.random.seed(42)
    train_indices = np.random.choice(train_candidate_indices, size=n_train_to_sample, replace=False)
    train_indices = sorted(train_indices.tolist())
    print(f"Sampled {n_train_to_sample} cities for training (train_frac={train_frac})")

    # Step 3: Filter cities by probe_test pattern
    if probe_test:
        test_candidates = filter_dataframe_by_pattern(city_df, probe_test, column_name='region')
        test_candidate_indices = test_candidates.index.tolist()
    else:
        test_candidate_indices = list(range(len(city_df)))

    # Step 4: Remove training cities from test set
    test_indices = [idx for idx in test_candidate_indices if idx not in train_indices]
    print(f"Cities for visualization (after removing train): {len(test_indices)}")

    # Get city info for visualization (test cities only)
    city_info = [all_city_info[idx] for idx in test_indices]

    # Determine projection mode
    projection_type = 'pca'  # default
    if axis_mapping and isinstance(axis_mapping, dict):
        projection_type = axis_mapping.get('type', 'pca')

    if projection_type == 'mixed' and cities_df is None:
        raise ValueError("cities_df required for mixed projection mode!")

    # Get unique regions and assign colors
    regions = sorted(set(c.get("region") for c in city_info if c.get("region") is not None))
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    region_to_color = {region: colors[i % len(colors)] for i, region in enumerate(regions)}

    # Get last checkpoint representations
    print("Loading final checkpoint...")
    last_step, last_repr_data, last_metadata = checkpoints[-1]
    all_representations = extract_representations(last_repr_data, token_index, layer_index)

    # Extract representations for train and test sets
    train_representations = all_representations[train_indices]
    test_representations = all_representations[test_indices]

    # Compute projection using the unified function
    projection_info = compute_projection(
        train_representations, test_representations, projection_type,
        all_city_info, train_indices, test_indices, cities_df, axis_mapping, n_components,
        checkpoints=checkpoints
    )

    projection_matrix = projection_info['projection_matrix']
    axis_labels = projection_info['axis_labels']
    axis_labels_short = projection_info['axis_labels_short']
    variance_explained = projection_info['variance_info']
    title_prefix = projection_info['title_prefix']
    subtitle_template = projection_info['subtitle_template']
    final_step = checkpoints[-1][0]


    # PROJECT ALL CHECKPOINTS using the same projection
    all_frames_data = []

    for step, repr_data, metadata in checkpoints:
        # Extract ALL representations
        all_reps = extract_representations(repr_data, token_index, layer_index)

        # Get only TEST representations for visualization
        test_reps = all_reps[test_indices]

        # Project TEST representations using the fixed projection matrix
        repr_projected = test_reps @ projection_matrix.T  # (n_test, 3)

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
                         f'{axis_labels_short[0]}: %{{x:.2f}}<br>' +
                         f'{axis_labels_short[1]}: %{{y:.2f}}<br>' +
                         f'{axis_labels_short[2]}: %{{z:.2f}}<br>' +
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

        frame_title = f"{title_prefix} - Step {frame_data['step']:,}<br>" + \
                     f"<sub>{subtitle_template.format(final_step=final_step)}</sub>"

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
    title_text = f"{title_prefix} - Step {all_frames_data[0]['step']:,}<br>" + \
                f"<sub>{subtitle_template.format(final_step=final_step)}</sub>"
    x_label = axis_labels[0]
    y_label = axis_labels[1]
    z_label = axis_labels[2]

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

    # Save all data needed to recreate the plot
    save_projection_data(
        output_path.parent,  # Same directory as HTML
        checkpoints,
        all_frames_data,
        projection_info,
        city_info,
        train_indices,
        test_indices,
        token_index,
        layer_index,
        axis_mapping,
        probe_train,
        probe_test,
        train_frac
    )

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
    marker_size = config.get('marker_size', 4)  # Default to 4 if not specified

    # Required parameter
    if 'train_frac' not in config:
        raise ValueError("ERROR: 'train_frac' is required in config!")
    train_frac = config['train_frac']

    # Optional regex patterns
    probe_train = config.get('probe_train', None)
    probe_test = config.get('probe_test', None)

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
    print(f"Train fraction: {train_frac}")
    if probe_train:
        print(f"Probe train pattern: {probe_train}")
    if probe_test:
        print(f"Probe test pattern: {probe_test}")

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
        output_path, train_frac, axis_mapping, marker_size, cities_df,
        probe_train, probe_test
    )

    print(f"\nSuccessfully created timeline with {n_checkpoints} checkpoints")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()