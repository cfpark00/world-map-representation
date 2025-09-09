#!/usr/bin/env python3
"""
Plot cities from a CSV file on a map with optional region highlighting.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append('.')  # Add root to path

def plot_cities(csv_path, output_path=None, highlight_region=None, exclude_region=None, figsize=(20, 12)):
    """
    Plot cities from CSV with optional region highlighting or exclusion.
    
    Args:
        csv_path: Path to CSV file with cities
        output_path: Where to save the figure (default: outputs/figures/cities_map.png)
        highlight_region: Region name to highlight in different color (e.g., 'Atlantis_0')
        exclude_region: Region name to exclude from plot (e.g., 'Atlantis')
        figsize: Figure size tuple
    """
    # Load cities
    print(f"Loading cities from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['x', 'y']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must have columns: {required_cols}")
        sys.exit(1)
    
    # Apply exclusion filter if specified
    if exclude_region and 'region' in df.columns:
        df = df[df['region'] != exclude_region].copy()
        print(f"Excluded {exclude_region} region from plot")
    
    # Set default output path
    if output_path is None:
        output_dir = Path('outputs/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'cities_map.png'
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set world map bounds (coordinates are scaled by 10)
    ax.set_xlim(-1800, 1800)
    ax.set_ylim(-900, 900)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot cities
    if highlight_region and 'region' in df.columns:
        # Split into highlighted and non-highlighted
        mask_highlight = df['region'] == highlight_region
        df_highlight = df[mask_highlight]
        df_normal = df[~mask_highlight]
        
        # Plot normal cities (no label here)
        if len(df_normal) > 0:
            ax.scatter(df_normal['x'], df_normal['y'], 
                      s=1, c='red', alpha=0.6, marker='.')
        
        # Plot highlighted region (no label here)
        if len(df_highlight) > 0:
            ax.scatter(df_highlight['x'], df_highlight['y'], 
                      s=5, c='blue', alpha=0.8, marker='.')
            print(f"Highlighted {len(df_highlight)} cities from {highlight_region}")
        else:
            print(f"Warning: No cities found for region '{highlight_region}'")
    else:
        # Plot all cities the same
        ax.scatter(df['x'], df['y'], 
                  s=1, c='red', alpha=0.6, marker='.')
    
    # Labels and title
    ax.set_xlabel('X (Longitude)', fontsize=20, labelpad=10)
    ax.set_ylabel('Y (Latitude)', fontsize=20, labelpad=10)
    
    title = f'City Distribution (n={len(df):,})'
    if highlight_region:
        title += f' - Highlighting {highlight_region}'
    if exclude_region:
        title += f' - Excluding {exclude_region}'
    ax.set_title(title, fontsize=24, pad=20)
    
    # Axis ticks (scaled by 10)
    ax.set_xticks(np.arange(-1800, 1801, 300))
    ax.set_yticks(np.arange(-900, 901, 300))
    ax.tick_params(axis='both', labelsize=18)
    
    # Make it look like a map
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f8ff')  # Light blue for oceans
    
    # Add legend if highlighting - use empty scatter plots with desired legend size
    if highlight_region and 'region' in df.columns and len(df_highlight) > 0:
        # Create invisible scatter plots just for legend - both using '.' marker
        ax.scatter([], [], s=100, c='red', alpha=0.6, marker='.', label='World Cities')
        ax.scatter([], [], s=100, c='blue', alpha=0.8, marker='.', label=highlight_region)
        ax.legend(loc='upper right', fontsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Map saved to {output_path}")
    plt.close()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total cities: {len(df):,}")
    print(f"X range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"Y range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    
    if 'region' in df.columns:
        print(f"\nRegions ({df['region'].nunique()}):")
        region_counts = df['region'].value_counts().head(10)
        for region, count in region_counts.items():
            print(f"  {region}: {count:,} cities")
        if len(df['region'].unique()) > 10:
            print(f"  ... and {df['region'].nunique() - 10} more regions")

def main():
    parser = argparse.ArgumentParser(description='Plot cities from CSV on a map')
    parser.add_argument('csv', type=str, help='Path to CSV file with cities')
    parser.add_argument('--output', type=str, default=None,
                       help='Output figure path (default: outputs/figures/cities_map.png)')
    parser.add_argument('--highlight-region', type=str, default=None,
                       help='Region to highlight in different color (e.g., Atlantis_0)')
    parser.add_argument('--exclude-region', type=str, default=None,
                       help='Region to exclude from plot (e.g., Atlantis)')
    parser.add_argument('--width', type=int, default=20,
                       help='Figure width in inches (default: 20)')
    parser.add_argument('--height', type=int, default=12,
                       help='Figure height in inches (default: 12)')
    
    args = parser.parse_args()
    
    plot_cities(
        csv_path=args.csv,
        output_path=args.output,
        highlight_region=args.highlight_region,
        exclude_region=args.exclude_region,
        figsize=(args.width, args.height)
    )

if __name__ == '__main__':
    main()