#!/usr/bin/env python3
"""Plot linear probe predicted Atlantis locations vs true locations."""

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
    if 'predictions_csv' not in config:
        raise ValueError("FATAL: 'predictions_csv' required in config")

    output_dir = init_directory(config['output_dir'], overwrite=overwrite)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load all cities
    df_cities = pd.read_csv(config['cities_csv'])
    print(f"Loaded {len(df_cities):,} cities")

    # Load predictions
    df_pred = pd.read_csv(config['predictions_csv'])
    print(f"Loaded {len(df_pred):,} predictions")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all cities by region (excluding Atlantis)
    for region in sorted(df_cities['region'].unique()):
        if region == 'Atlantis':
            continue
        mask = df_cities['region'] == region
        color = REGION_COLORS.get(region, '#888888')
        ax.scatter(df_cities.loc[mask, 'x'] / 10, df_cities.loc[mask, 'y'] / 10,
                   c=color, s=12, alpha=0.7)

    # Plot true Atlantis locations (black crosses)
    ax.scatter(df_pred['x_true'] / 10, df_pred['y_true'] / 10,
               c='black', s=120, alpha=0.9, marker='x',
               linewidths=3, label='Atlantis (true)', zorder=10)

    # Plot predicted Atlantis locations (red circles)
    ax.scatter(df_pred['x_pred'] / 10, df_pred['y_pred'] / 10,
               c=REGION_COLORS['Atlantis'], s=80, alpha=0.9, marker='o',
               edgecolors='white', linewidths=0.5, label='Atlantis (predicted)', zorder=11)

    ax.set_xlim([-90, 25])
    ax.set_ylim([5, 62.5])
    ax.set_aspect('equal')

    # Thick spines and ticks, no axis labels
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=6)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Tick labels in actual scale (multiply by 10), bold and bigger
    ax.set_xticks([-75, -50, -25, 0, 25])
    ax.set_xticklabels(['-750', '-500', '-250', '0', '250'],
                       fontsize=16, fontweight='bold')
    ax.set_yticks([10, 30, 50])
    ax.set_yticklabels(['100', '300', '500'], fontsize=16, fontweight='bold')

    plt.tight_layout()
    output_path = figures_dir / 'probe_atlantis_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite)
