#!/usr/bin/env python3
"""
Simple city visualization - plots cities on a map.
Red dots for normal cities, blue (bigger) for Atlantis.
"""

import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import init_directory

def main():
    parser = argparse.ArgumentParser(description='Visualize city dataset')
    parser.add_argument('config', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")
    if 'dataset_path' not in config:
        raise ValueError("FATAL: 'dataset_path' required in config")

    dataset_path = Path(config['dataset_path'])
    cities_csv = dataset_path / 'cities.csv'
    output_dir = init_directory(config['output_dir'], overwrite=args.overwrite)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    df = pd.read_csv(cities_csv)
    print(f"Loaded {len(df):,} cities")

    # Apply filter if specified
    if 'region_filter' in config:
        filter_str = config['region_filter']
        if filter_str.startswith('region:'):
            pattern = filter_str.replace('region:', '')
            mask = df['region'].str.contains(pattern, regex=True, na=False)
            df = df[mask]
            print(f"After filtering with '{pattern}': {len(df):,} cities")

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_facecolor(config.get('background_color', '#e6f2ff'))

    # Separate Atlantis from regular cities
    atlantis_mask = df['region'].str.contains('Atlantis', na=False)
    regular_cities = df[~atlantis_mask]
    atlantis_cities = df[atlantis_mask]

    # Plot regular cities (red, small)
    if len(regular_cities) > 0:
        ax.scatter(regular_cities['x'] / 10, regular_cities['y'] / 10,
                  c='red', s=8, alpha=0.6, edgecolors='none')

    # Plot Atlantis cities (blue, bigger)
    if len(atlantis_cities) > 0:
        ax.scatter(atlantis_cities['x'] / 10, atlantis_cities['y'] / 10,
                  c='blue', s=30, alpha=0.8, edgecolors='white', linewidths=0.5)

    ax.set_xlim([-140, 180])
    ax.set_ylim([-60, 75])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=20, pad=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    output_path = figures_dir / 'cities_map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
