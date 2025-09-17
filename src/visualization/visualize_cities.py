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
import re

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import init_directory

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize city dataset')
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
    ax.set_facecolor('#e6f2ff')
    
    # Separate Atlantis from regular cities
    atlantis_mask = df['region'].str.contains('Atlantis', na=False)
    regular_cities = df[~atlantis_mask]
    atlantis_cities = df[atlantis_mask]
    
    # Plot regular cities (red, small)
    # Note: coordinates are scaled by 10 in the dataset
    if len(regular_cities) > 0:
        ax.scatter(regular_cities['x'] / 10, regular_cities['y'] / 10,
                  c=config['regular_color'], 
                  s=config['regular_size'],
                  alpha=0.6,
                  edgecolors='none',
                  label=f'Cities (n={len(regular_cities):,})')
    
    # Plot Atlantis cities (blue, bigger)
    if len(atlantis_cities) > 0:
        ax.scatter(atlantis_cities['x'] / 10, atlantis_cities['y'] / 10,
                  c=config['atlantis_color'],
                  s=config['atlantis_size'],
                  alpha=0.8,
                  edgecolors='white',
                  linewidths=0.5,
                  label=f'Atlantis (n={len(atlantis_cities):,})')
    
    # Set proper bounds (degrees)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    
    # Grid and labels
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Longitude (°)', fontsize=14)
    ax.set_ylabel('Latitude (°)', fontsize=14)
    ax.set_title('World Cities Distribution', fontsize=16, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', fontsize=11)
    
    # Save
    plt.tight_layout()
    output_path = figures_dir / 'cities_map.png'
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()