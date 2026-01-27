#!/usr/bin/env python3
"""
Recreate PCA timeline plots from saved projection data.

This script loads the saved projection data and recreates the interactive
3D visualization without needing the original representations.

Usage:
    python recreate_pca_plot_from_data.py <data_dir> [--output <output.html>]
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import sys

# Add parent directory to path for imports
project_root = Path('')
sys.path.insert(0, str(project_root))


def recreate_plot_from_data(data_dir, output_path=None):
    """
    Recreate the PCA timeline plot from saved data.

    Args:
        data_dir: Directory containing saved projection data
        output_path: Optional path for output HTML (defaults to original location)
    """
    data_dir = Path(data_dir)

    # Load the complete data package
    pickle_path = data_dir / 'projection_data.pkl'
    if not pickle_path.exists():
        raise FileNotFoundError(f"Data package not found at {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data_package = pickle.load(f)

    # Extract components
    all_frames_data = data_package['all_frames_data']
    projection_info = data_package['projection_info']
    city_info = data_package['city_info']
    test_indices = data_package['test_indices']
    axis_mapping = data_package['axis_mapping']

    # Get display parameters
    axis_labels = projection_info['axis_labels']
    axis_labels_short = projection_info['axis_labels_short']
    title_prefix = projection_info['title_prefix']
    subtitle_template = projection_info['subtitle_template']

    # Default output path if not specified
    if output_path is None:
        output_path = data_dir / 'pca_3d_timeline_recreated.html'
    else:
        output_path = Path(output_path)

    # Get unique regions and assign colors
    regions = sorted(set(c.get("region") for c in city_info if c.get("region") is not None))
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    region_to_color = {region: colors[i % len(colors)] for i, region in enumerate(regions)}

    # Get marker size (use default if not in metadata)
    marker_size = 4  # Default

    # Create figure with first frame
    fig = go.Figure()
    first_frame = all_frames_data[0]

    # Add traces for each region (for first frame)
    for region in regions:
        # Get indices for this region
        region_mask = [i for i, c in enumerate(city_info) if c.get('region') == region]

        if not region_mask:
            continue

        # Add scatter trace
        fig.add_trace(go.Scatter3d(
            x=first_frame['repr_projected'][region_mask, 0],
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

        # Get final step for subtitle
        final_step = data_package['checkpoints_info'][-1][0] if data_package['checkpoints_info'] else 0
        frame_title = f"{title_prefix} - Step {frame_data['step']:,}<br>" + \
                     f"<sub>{subtitle_template.format(final_step=final_step)}</sub>"

        frames.append(go.Frame(
            data=frame_traces,
            name=str(frame_data['step']),
            layout=go.Layout(
                title=dict(text=frame_title)
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
    final_step = data_package['checkpoints_info'][-1][0] if data_package['checkpoints_info'] else 0
    title_text = f"{title_prefix} - Step {all_frames_data[0]['step']:,}<br>" + \
                f"<sub>{subtitle_template.format(final_step=final_step)}</sub>"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
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

    # Save plot
    fig.write_html(output_path, auto_play=False)
    print(f"Recreated plot saved to {output_path}")

    # Print info about the data
    print(f"\nData summary:")
    print(f"  Checkpoints: {len(all_frames_data)}")
    print(f"  Cities visualized: {len(city_info)}")
    print(f"  Regions: {len(regions)}")
    print(f"  Projection type: {axis_mapping.get('type', 'pca') if axis_mapping else 'pca'}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Recreate PCA timeline plot from saved data')
    parser.add_argument('data_dir', type=str, help='Directory containing saved projection data')
    parser.add_argument('--output', type=str, help='Output HTML file path (optional)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        sys.exit(1)

    recreate_plot_from_data(data_dir, args.output)


if __name__ == "__main__":
    main()