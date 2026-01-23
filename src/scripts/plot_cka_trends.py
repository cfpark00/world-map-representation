#!/usr/bin/env python3
"""
Plot CKA trends across PT1 (1-task) → PT2 (2-task) → PT3 (3-task).
Creates separate plots for each seed configuration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_cka_stats(df, prefix, layer, seed_filter=None, exclude_overlap=True):
    """Calculate mean and SEM for a given prefix, layer, and optional seed."""
    data = df[(df['prefix'] == prefix) & (df['layer'] == layer)]

    if exclude_overlap:
        data = data[~data['training_overlap']]

    if seed_filter is not None:
        # For seed filter, we want comparisons where both experiments have that seed
        data = data[(data['seed1'] == seed_filter) & (data['seed2'] == seed_filter)]

    values = data['final_cka'].values
    mean = np.mean(values) if len(values) > 0 else np.nan
    sem = np.std(values) / np.sqrt(len(values)) if len(values) > 0 else np.nan
    n = len(values)

    return mean, sem, n, values

def plot_cka_trends_by_seed(df, output_dir):
    """Create CKA trends plot separated by seed."""

    # Determine which seeds are available for PT1
    pt1_data = df[df['prefix'] == 'pt1']
    available_seeds = sorted(pt1_data['seed1'].unique())

    # Map seeds to labels
    seed_labels = {
        42: 'Original (seed 42)',
        1: 'Seed 1',
        2: 'Seed 2',
        3: 'Seed 3',
        4: 'Seed 4',
    }

    layers = [3, 4, 5, 6]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    fig, axes = plt.subplots(1, len(available_seeds), figsize=(6 * len(available_seeds), 6), squeeze=False)
    axes = axes[0]  # Get first row

    line_width = 4.5  # 1.5x thicker than 3

    for seed_idx, seed in enumerate(available_seeds):
        ax = axes[seed_idx]

        x_positions = np.arange(3)  # PT1, PT2, PT3

        for layer_idx, layer in enumerate(layers):
            y_values = []
            y_errors = []

            for prefix in ['pt1', 'pt2', 'pt3']:
                mean, sem, n, values = calculate_cka_stats(df, prefix, layer, seed_filter=seed)
                y_values.append(mean)
                y_errors.append(sem)

            # Plot line with error bars
            ax.errorbar(x_positions, y_values,
                       yerr=y_errors,
                       marker=markers[layer_idx],
                       color=colors[layer_idx],
                       linewidth=line_width,
                       markersize=10,
                       label=f'Layer {layer}',
                       alpha=0.9,
                       capsize=6,
                       capthick=line_width)

        # Formatting - no xlabel, no title
        ax.set_xticks(x_positions)
        ax.set_xticklabels([])
        if seed_idx == 0:
            ax.set_ylabel('CKA Similarity', fontsize=18, fontweight='bold')  # Bigger ylabel
        ax.set_ylim([0, 1])

        # Bold y-axis labels, thicker ticks
        ax.tick_params(axis='y', which='major', labelsize=28, pad=8, width=3, length=9)
        ax.tick_params(axis='x', which='major', labelsize=28, pad=8, width=3, length=9)

        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Remove top and right spines, thicken bottom and left
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3.75)
        ax.spines['left'].set_linewidth(3.75)

        ax.legend(fontsize=12)

    plt.tight_layout()

    output_file = output_dir / 'cka_trends_by_seed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_cka_trends_aggregated(df, output_dir):
    """Create aggregated CKA trends plot (all seeds combined)."""

    layers = [3, 4, 5, 6]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(3)  # PT1, PT2, PT3
    jitter = 0.045  # 0.03 * 1.5

    line_width = 4.5  # 1.5x thicker than 3

    for layer_idx, layer in enumerate(layers):
        y_values = []
        y_errors = []

        for prefix in ['pt1', 'pt2', 'pt3']:
            mean, sem, n, values = calculate_cka_stats(df, prefix, layer, seed_filter=None)
            y_values.append(mean)
            y_errors.append(sem)

            # Plot individual points with jitter
            if len(values) > 0:
                x_jitter = x_positions[len(y_values)-1] + np.random.normal(0, jitter, len(values))
                ax.scatter(x_jitter, values, color=colors[layer_idx], alpha=0.15, s=20, zorder=1)

        # Plot line with error bars
        ax.errorbar(x_positions, y_values,
                    yerr=y_errors,
                    marker=markers[layer_idx],
                    color=colors[layer_idx],
                    linewidth=line_width,
                    markersize=10,
                    label=f'Layer {layer}',
                    alpha=0.9,
                    capsize=6,
                    capthick=line_width,
                    zorder=2)

    # Formatting - no title, no x labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])
    ax.set_ylim([0, 1])

    # Bold y-axis labels, thicker ticks
    ax.tick_params(axis='y', which='major', labelsize=28, pad=8, width=3, length=9)
    ax.tick_params(axis='x', which='major', labelsize=28, pad=8, width=3, length=9)

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove top and right spines, thicken bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3.75)  # 1.5x thicker than 2.5
    ax.spines['left'].set_linewidth(3.75)

    plt.tight_layout()

    output_file = output_dir / 'cka_trends_aggregated.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_cka_trends_single_seed(df, seed, seed_label, output_dir):
    """Create CKA trends plot for a single seed."""

    layers = [3, 4, 5, 6]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(3)  # PT1, PT2, PT3
    jitter = 0.036  # 0.03 * 1.2

    line_width = 4.5  # 1.5x thicker than 3

    for layer_idx, layer in enumerate(layers):
        y_values = []
        y_errors = []

        for prefix in ['pt1', 'pt2', 'pt3']:
            mean, sem, n, values = calculate_cka_stats(df, prefix, layer, seed_filter=seed)
            y_values.append(mean)
            y_errors.append(sem)

            # Plot individual points with jitter
            if len(values) > 0:
                x_jitter = x_positions[len(y_values)-1] + np.random.normal(0, jitter, len(values))
                ax.scatter(x_jitter, values, color=colors[layer_idx], alpha=0.15, s=20, zorder=1)

        # Plot line with error bars
        ax.errorbar(x_positions, y_values,
                    yerr=y_errors,
                    marker=markers[layer_idx],
                    color=colors[layer_idx],
                    linewidth=line_width,
                    markersize=10,
                    label=f'Layer {layer}',
                    alpha=0.9,
                    capsize=6,
                    capthick=line_width,
                    zorder=2)

    # Formatting - no title, no x labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])
    ax.set_ylim([0, 1])

    # Bold y-axis labels, thicker ticks
    ax.tick_params(axis='y', which='major', labelsize=28, pad=8, width=3, length=9)
    ax.tick_params(axis='x', which='major', labelsize=28, pad=8, width=3, length=9)

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove top and right spines, thicken bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3.75)  # 1.5x thicker than 2.5
    ax.spines['left'].set_linewidth(3.75)

    plt.tight_layout()

    # Create seed-specific filename
    seed_suffix = 'orig' if seed == 42 else f'seed{seed}'
    output_file = output_dir / f'cka_trends_{seed_suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    base_path = Path(__file__).resolve().parents[2]
    data_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_trends'

    # Load data
    csv_file = data_dir / 'cka_summary.csv'
    if not csv_file.exists():
        print(f"Error: {csv_file} not found. Run collect_cka_trends_data.py first.")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} CKA comparisons")

    # Create plots
    print("\nCreating aggregated plot...")
    plot_cka_trends_aggregated(df, data_dir)

    # Create individual seed plots (orig, seed1, seed2 only)
    print("\nCreating individual seed plots...")
    seeds_to_plot = [42, 1, 2]  # Only orig, seed1, seed2
    seed_labels = {
        42: 'Original (seed 42)',
        1: 'Seed 1',
        2: 'Seed 2',
    }

    for seed in seeds_to_plot:
        label = seed_labels[seed]
        print(f"  Creating plot for {label}...")
        plot_cka_trends_single_seed(df, seed, label, data_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()
