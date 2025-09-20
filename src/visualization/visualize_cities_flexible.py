#!/usr/bin/env python3
"""
Flexible city visualization - plots cities on a map with configurable highlighting.
Supports multiple region groups with custom colors and sizes via YAML config.
"""

import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import init_directory

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize city dataset with flexible highlighting')
    parser.add_argument('config', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths with init_directory
    dataset_path = Path(config['dataset_path'])
    cities_csv = dataset_path / 'cities.csv'
    output_dir = init_directory(config['output_dir'], overwrite=args.overwrite)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(cities_csv)
    print(f"Loaded {len(df):,} cities")

    # Apply regex filter if specified
    if 'region_filter' in config:
        filter_str = config['region_filter']
        if filter_str.startswith('region:'):
            pattern = filter_str.replace('region:', '')
            mask = df['region'].str.contains(pattern, regex=True, na=False)
            df = df[mask]
            print(f"After filtering with '{pattern}': {len(df):,} cities")

    # Create figure
    fig, ax = plt.subplots(figsize=tuple(config['figsize']))

    # Set background
    ax.set_facecolor(config.get('background_color', '#e6f2ff'))

    # Get highlight groups configuration
    highlight_groups = config.get('highlight_groups', [])

    # Track which cities have been plotted
    plotted_mask = pd.Series(False, index=df.index)

    # Plot each highlight group
    for group in highlight_groups:
        # Create mask for this group
        if 'regions' in group:
            # Match specific regions
            if isinstance(group['regions'], list):
                group_mask = df['region'].isin(group['regions'])
            else:
                # Single region or regex pattern
                group_mask = df['region'].str.contains(group['regions'], regex=True, na=False)
        elif 'pattern' in group:
            # General regex pattern on region column
            group_mask = df['region'].str.contains(group['pattern'], regex=True, na=False)
        else:
            print(f"Warning: highlight group '{group.get('label', 'unnamed')}' has no regions or pattern")
            continue

        # Exclude already plotted cities
        group_mask = group_mask & ~plotted_mask
        group_cities = df[group_mask]

        if len(group_cities) > 0:
            ax.scatter(group_cities['x'] / 10, group_cities['y'] / 10,
                      c=group.get('color', 'blue'),
                      s=group.get('size', 20),
                      alpha=group.get('alpha', 0.7),
                      edgecolors=group.get('edgecolor', 'none'),
                      linewidths=group.get('linewidth', 0.5),
                      marker=group.get('marker', 'o'),
                      label=f"{group.get('label', 'Group')} (n={len(group_cities):,})")

            plotted_mask |= group_mask
            print(f"Plotted {len(group_cities):,} cities for group '{group.get('label', 'unnamed')}'")

    # Plot remaining cities as default
    default_config = config.get('default_cities', {})
    remaining_cities = df[~plotted_mask]

    if len(remaining_cities) > 0:
        ax.scatter(remaining_cities['x'] / 10, remaining_cities['y'] / 10,
                  c=default_config.get('color', 'gray'),
                  s=default_config.get('size', 5),
                  alpha=default_config.get('alpha', 0.5),
                  edgecolors=default_config.get('edgecolor', 'none'),
                  linewidths=default_config.get('linewidth', 0),
                  marker=default_config.get('marker', 'o'),
                  label=f"{default_config.get('label', 'Other cities')} (n={len(remaining_cities):,})")
        print(f"Plotted {len(remaining_cities):,} remaining cities as default")

    # Set proper bounds (degrees)
    xlim = config.get('xlim', [-180, 180])
    ylim = config.get('ylim', [-90, 90])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Grid and labels
    ax.grid(True, alpha=config.get('grid_alpha', 0.3), linestyle=config.get('grid_style', '--'))
    ax.set_xlabel(config.get('xlabel', 'Longitude (°)'), fontsize=config.get('label_fontsize', 14))
    ax.set_ylabel(config.get('ylabel', 'Latitude (°)'), fontsize=config.get('label_fontsize', 14))
    ax.set_title(config.get('title', 'World Cities Distribution'),
                fontsize=config.get('title_fontsize', 16),
                fontweight='bold')

    # Legend
    legend_config = config.get('legend', {})
    ax.legend(loc=legend_config.get('loc', 'upper left'),
             fontsize=legend_config.get('fontsize', 11),
             framealpha=legend_config.get('framealpha', 0.9))

    # Save
    plt.tight_layout()
    output_filename = config.get('output_filename', 'cities_map.png')
    output_path = figures_dir / output_filename
    plt.savefig(output_path, dpi=config.get('dpi', 300), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Save config copy for reproducibility
    import shutil
    config_copy_path = output_dir / 'config.yaml'
    shutil.copy(args.config, config_copy_path)
    print(f"Config saved to: {config_copy_path}")

if __name__ == "__main__":
    main()