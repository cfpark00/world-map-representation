#!/usr/bin/env python3
"""Basic city scatter plot - all cities including Atlantis, colored by region."""

import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import init_directory

# Colors matching plotly qualitative (Plotly + D3 + G10) used in PCA visualizations
REGION_COLORS = {
    "Africa": "#636EFA",
    "Atlantis": "#EF553B",
    "Central Asia": "#00CC96",
    "China": "#AB63FA",
    "Eastern Europe": "#FFA15A",
    "India": "#19D3F3",
    "Japan": "#FF6692",
    "Korea": "#B6E880",
    "Middle East": "#FF97FF",
    "North America": "#FECB52",
    "Oceania": "#1F77B4",
    "South America": "#FF7F0E",
    "Southeast Asia": "#2CA02C",
    "Western Europe": "#D62728",
}


def main(config_path, overwrite=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")
    if 'cities_csv' not in config:
        raise ValueError("FATAL: 'cities_csv' required in config")

    output_dir = init_directory(config['output_dir'], overwrite=overwrite)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config['cities_csv'])
    print(f"Loaded {len(df):,} cities")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each region with its color
    for region in sorted(df['region'].unique()):
        mask = df['region'] == region
        color = REGION_COLORS.get(region, '#888888')
        ax.scatter(df.loc[mask, 'x'] / 10, df.loc[mask, 'y'] / 10,
                   c=color, s=6, alpha=0.7)

    ax.set_xlim([-165, 185])
    ax.set_ylim([-65, 80])
    ax.set_aspect('equal')

    # Thick spines and ticks, no axis labels
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=6)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Tick labels in actual scale (multiply by 10), bold and bigger
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_xticklabels(['-1500', '-1000', '-500', '0', '500', '1000', '1500'],
                       fontsize=16, fontweight='bold')
    ax.set_yticks([-50, 0, 50])
    ax.set_yticklabels(['-500', '0', '500'], fontsize=16, fontweight='bold')

    plt.tight_layout()
    output_path = figures_dir / 'cities_basic.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite)
