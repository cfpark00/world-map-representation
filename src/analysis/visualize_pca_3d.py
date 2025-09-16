#!/usr/bin/env python3
"""
Create 3D PCA visualization of city representations colored by region.

This script loads saved representations, performs PCA to reduce to 3 dimensions,
and creates an interactive 3D plot with cities colored by their geographic regions.

Usage:
    python visualize_pca_3d.py configs/analysis/pca_3d.yaml
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

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory


def create_pca_3d_plot(representations, city_info, n_train, output_path,
                       title_suffix="", n_components=3, axis_mapping=None):
    """
    Create 3D PCA visualization of city representations.

    Args:
        representations: Array of shape (n_cities, feature_dim)
        city_info: List of city information dictionaries
        n_train: Number of training cities
        output_path: Path to save the HTML plot
        title_suffix: Additional text for the plot title
        n_components: Number of PCA components to compute
        axis_mapping: Dict mapping x/y/z to PC indices (0-based), e.g. {1: 1, 2: 2, 3: 3} for PC2,PC3,PC4
    """
    # Compute PCA with enough components
    if axis_mapping:
        # Need enough components for the highest requested PC
        max_pc = max(axis_mapping.values())
        n_components = max(n_components, max_pc + 1)

    pca = PCA(n_components=n_components)
    repr_pca = pca.fit_transform(representations)

    # Determine which PCs to use for each axis
    if axis_mapping:
        x_pc = axis_mapping.get(1, 0)  # Default to PC1
        y_pc = axis_mapping.get(2, 1)  # Default to PC2
        z_pc = axis_mapping.get(3, 2)  # Default to PC3
    else:
        x_pc, y_pc, z_pc = 0, 1, 2  # Default: PC1, PC2, PC3

    # Get all unique regions (excluding None)
    regions = sorted(set(c.get("region") for c in city_info if c.get("region") is not None))

    # Assign colors to regions
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    region_to_color = {region: colors[i % len(colors)] for i, region in enumerate(regions)}

    # Prepare data for each region
    region_indices = defaultdict(list)
    for i, c in enumerate(city_info):
        region = c.get("region")
        if region is not None:
            region_indices[region].append(i)

    # Create 3D scatter plot
    fig = go.Figure()

    # Track which cities are train vs test
    train_test_split = ['train' if i < n_train else 'test' for i in range(len(city_info))]

    for region in regions:
        idxs = region_indices[region]

        # Count train/test split for this region
        n_train_region = sum(1 for i in idxs if i < n_train)
        n_test_region = len(idxs) - n_train_region

        # Add hover text with city names and split info
        hover_text = []
        for i in idxs:
            city = city_info[i]
            split = train_test_split[i]
            hover_text.append(
                f"{city['name']}<br>"
                f"Region: {region}<br>"
                f"Split: {split}<br>"
                f"PC{x_pc+1}: {repr_pca[i, x_pc]:.2f}<br>"
                f"PC{y_pc+1}: {repr_pca[i, y_pc]:.2f}<br>"
                f"PC{z_pc+1}: {repr_pca[i, z_pc]:.2f}"
            )

        fig.add_trace(go.Scatter3d(
            x=[repr_pca[i, x_pc] for i in idxs],
            y=[repr_pca[i, y_pc] for i in idxs],
            z=[repr_pca[i, z_pc] for i in idxs],
            mode='markers',
            name=f'{region} ({n_train_region}/{n_test_region})',  # Show train/test counts
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                size=4,
                color=region_to_color[region],
                opacity=0.8
            )
        ))

    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_

    # Get variance for the selected components
    x_var = variance_explained[x_pc] if x_pc < len(variance_explained) else 0
    y_var = variance_explained[y_pc] if y_pc < len(variance_explained) else 0
    z_var = variance_explained[z_pc] if z_pc < len(variance_explained) else 0
    total_var = x_var + y_var + z_var

    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f'3D PCA of City Representations by Region{title_suffix}<br>'
                f'<sub>PC{x_pc+1}: {x_var:.1%} | PC{y_pc+1}: {y_var:.1%} | '
                f'PC{z_pc+1}: {z_var:.1%} | Total: {total_var:.1%}</sub>'
            ),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=f'PC{x_pc+1} ({x_var:.1%})',
            yaxis_title=f'PC{y_pc+1} ({y_var:.1%})',
            zaxis_title=f'PC{z_pc+1} ({z_var:.1%})',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
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
            borderwidth=1,
            title='Region (train/test)'
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    # Save plot
    fig.write_html(output_path)
    print(f"Saved 3D PCA plot to: {output_path}")

    return fig, variance_explained


def main(config_path, overwrite=False, debug=False):
    """Main function to create 3D PCA visualization."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config - fail fast!
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")
    if 'representation_path' not in config:
        raise ValueError("FATAL: 'representation_path' required in config")

    # Initialize output directory with safety checks
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Copy config to output directory for reproducibility
    shutil.copy(config_path, output_dir / 'config.yaml')

    # Get parameters from config
    repr_path = Path(config['representation_path'])
    token_index = config.get('token_index', 1)  # Default to 'c' token
    layer_index = config.get('layer_index', -1)  # Default to last layer
    n_components = config.get('n_components', 3)  # Number of PCA components
    axis_mapping = config.get('axis_mapping', None)  # PC mapping for each axis

    print("="*60)
    print("3D PCA Visualization")
    print("="*60)
    print(f"Representation path: {repr_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of PCA components: {n_components}")

    if axis_mapping:
        print(f"Axis mapping: X=PC{axis_mapping.get(1, 1)}, Y=PC{axis_mapping.get(2, 2)}, Z=PC{axis_mapping.get(3, 3)}")
    else:
        print("Using default axis mapping: X=PC1, Y=PC2, Z=PC3")

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
            # For old format, reshape if needed
            n_cities = metadata.get('n_cities', len(selected_repr))
            n_layers = len(metadata.get('layers', []))
            if n_layers > 0:
                # Assume 3 tokens, reshape to get middle token, last layer
                n_tokens = 3
                hidden_dim = selected_repr.shape[1] // (n_tokens * n_layers)
                reshaped = selected_repr.reshape(n_cities, n_tokens, n_layers, hidden_dim)
                selected_repr = reshaped[:, 1, -1, :]  # Middle token (c), last layer
            print("Using flat representations (old format, extracted token 1, last layer)")
            print(f"Representation shape: {selected_repr.shape}")
        else:
            raise ValueError("FATAL: Could not find representations in the loaded file")

    # Get city info and training split info
    city_info = metadata['city_info']
    n_train = metadata.get('n_train_cities', len(city_info) * 3 // 5)  # Default 60% train
    n_test = metadata.get('n_test_cities', len(city_info) - n_train)

    print(f"\nDataset split:")
    print(f"  Training cities: {n_train}")
    print(f"  Test cities: {n_test}")
    print(f"  Total cities: {len(city_info)}")

    # Get unique regions
    regions = sorted(set(c.get("region") for c in city_info if c.get("region") is not None))
    print(f"  Regions: {len(regions)}")

    # Create PCA visualization
    print(f"\nPerforming PCA with {n_components} components...")

    # Create plots for different views
    plots_created = []

    # 1. Combined train+test plot
    output_path = output_dir / 'pca_3d_all.html'
    fig, variance_explained = create_pca_3d_plot(
        selected_repr, city_info, n_train, output_path,
        title_suffix=" (All Cities)", n_components=n_components,
        axis_mapping=axis_mapping
    )
    plots_created.append('pca_3d_all.html')

    # 2. Test set only plot (optional)
    if config.get('include_test_only', True):
        test_repr = selected_repr[n_train:]
        test_city_info = city_info[n_train:]
        test_output_path = output_dir / 'pca_3d_test.html'

        # For test-only, all cities are "test" so n_train=0
        create_pca_3d_plot(
            test_repr, test_city_info, 0, test_output_path,
            title_suffix=" (Test Set Only)", n_components=n_components,
            axis_mapping=axis_mapping
        )
        plots_created.append('pca_3d_test.html')

    # 3. Training set only plot (optional)
    if config.get('include_train_only', False):
        train_repr = selected_repr[:n_train]
        train_city_info = city_info[:n_train]
        train_output_path = output_dir / 'pca_3d_train.html'

        # For train-only, all cities are "train" so n_train=len(train_city_info)
        create_pca_3d_plot(
            train_repr, train_city_info, len(train_city_info), train_output_path,
            title_suffix=" (Training Set Only)", n_components=n_components,
            axis_mapping=axis_mapping
        )
        plots_created.append('pca_3d_train.html')

    # Save numerical results
    results = {
        'n_components': n_components,
        'variance_explained': variance_explained.tolist(),
        'total_variance_explained': float(sum(variance_explained)),
        'n_train': n_train,
        'n_test': n_test,
        'n_regions': len(regions),
        'regions': regions,
        'representation_shape': list(selected_repr.shape),
        'token_index': token_index,
        'layer_index': layer_index,
        'config': config
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("PCA Visualization Complete")
    print("="*60)
    print(f"Output directory: {output_dir}")
    for plot_file in plots_created:
        print(f"  - {plot_file}")
    print(f"  - results.json")
    print(f"\nVariance explained:")
    print(f"  PC1: {variance_explained[0]:.1%}")
    print(f"  PC2: {variance_explained[1]:.1%}")
    print(f"  PC3: {variance_explained[2]:.1%}")
    print(f"  Total: {sum(variance_explained):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create 3D PCA visualization')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')

    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)