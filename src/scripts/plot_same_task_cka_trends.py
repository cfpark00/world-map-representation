#!/usr/bin/env python3
"""
Plot CKA trends for same-task, different-seed comparisons.
Shows that multi-task training increases alignment even for the same task.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_same_task_trends(df_same, df_cross, output_dir):
    """Plot CKA trends for same-task vs cross-task comparisons across PT1 → PT2 → PT3."""

    layer = 5  # Only layer 5
    color_same = '#2ca02c'
    color_cross = 'gray'
    marker = '^'

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(3)  # PT1, PT2, PT3
    jitter = 0.045

    line_width = 4.5

    # Plot same-task (solid line)
    y_values = []
    y_errors = []

    for prefix in ['pt1', 'pt2', 'pt3']:
        data = df_same[(df_same['prefix'] == prefix) & (df_same['layer'] == layer)]
        values = data['final_cka'].values

        mean = np.mean(values) if len(values) > 0 else np.nan
        sem = np.std(values) / np.sqrt(len(values)) if len(values) > 0 else np.nan

        y_values.append(mean)
        y_errors.append(sem)

        # Plot individual points with jitter
        if len(values) > 0:
            x_jitter = x_positions[len(y_values)-1] + np.random.normal(0, jitter, len(values))
            ax.scatter(x_jitter, values, color=color_same, alpha=0.15, s=20, zorder=1)

    # Plot solid line with error bars (same-task)
    ax.errorbar(x_positions, y_values,
                yerr=y_errors,
                marker=marker,
                color=color_same,
                linewidth=line_width,
                markersize=10,
                alpha=0.9,
                capsize=6,
                capthick=line_width,
                zorder=2,
                label='Same-task (different seeds)')

    # Plot cross-task (dotted line)
    y_values_cross = []
    y_errors_cross = []

    for prefix in ['pt1', 'pt2', 'pt3']:
        data = df_cross[(df_cross['prefix'] == prefix) & (df_cross['layer'] == layer)]
        values = data['final_cka'].values

        mean = np.mean(values) if len(values) > 0 else np.nan
        sem = np.std(values) / np.sqrt(len(values)) if len(values) > 0 else np.nan

        y_values_cross.append(mean)
        y_errors_cross.append(sem)

        # Plot individual points with jitter
        if len(values) > 0:
            x_jitter = x_positions[len(y_values_cross)-1] + np.random.normal(0, jitter, len(values))
            ax.scatter(x_jitter, values, color=color_cross, alpha=0.15, s=20, zorder=1)

    # Plot dotted line with error bars (cross-task)
    ax.errorbar(x_positions, y_values_cross,
                yerr=y_errors_cross,
                marker=marker,
                color=color_cross,
                linewidth=line_width,
                markersize=10,
                alpha=0.9,
                capsize=6,
                capthick=line_width,
                linestyle='--',
                zorder=2,
                label='Cross-task (different tasks)')

    # Formatting - no title, no x labels, no y label
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
    ax.spines['bottom'].set_linewidth(3.75)
    ax.spines['left'].set_linewidth(3.75)

    plt.tight_layout()

    output_file = output_dir / 'same_task_cka_trends_with_cross.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_same_task_only(df_same, output_dir):
    """Plot CKA trends for same-task only (without cross-task comparison)."""

    layer = 5  # Only layer 5
    color_same = '#2ca02c'
    marker = '^'

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(3)  # PT1, PT2, PT3
    jitter = 0.045

    line_width = 4.5

    # Plot same-task (solid line)
    y_values = []
    y_errors = []

    for prefix in ['pt1', 'pt2', 'pt3']:
        data = df_same[(df_same['prefix'] == prefix) & (df_same['layer'] == layer)]
        values = data['final_cka'].values

        mean = np.mean(values) if len(values) > 0 else np.nan
        sem = np.std(values) / np.sqrt(len(values)) if len(values) > 0 else np.nan

        y_values.append(mean)
        y_errors.append(sem)

        # Plot individual points with jitter
        if len(values) > 0:
            x_jitter = x_positions[len(y_values)-1] + np.random.normal(0, jitter, len(values))
            ax.scatter(x_jitter, values, color=color_same, alpha=0.15, s=20, zorder=1)

    # Plot solid line with error bars (same-task)
    ax.errorbar(x_positions, y_values,
                yerr=y_errors,
                marker=marker,
                color=color_same,
                linewidth=line_width,
                markersize=10,
                alpha=0.9,
                capsize=6,
                capthick=line_width,
                zorder=2)

    # Formatting - no title, no x labels, no y label
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
    ax.spines['bottom'].set_linewidth(3.75)
    ax.spines['left'].set_linewidth(3.75)

    plt.tight_layout()

    output_file = output_dir / 'same_task_cka_trends.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    base_path = Path(__file__).resolve().parents[2]
    data_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_trends'

    # Load same-task data
    csv_file = data_dir / 'same_task_cka_summary.csv'
    if not csv_file.exists():
        print(f"Error: {csv_file} not found. Run collect_same_task_cka_trends.py first.")
        return

    df_same = pd.read_csv(csv_file)
    print(f"Loaded {len(df_same)} same-task CKA comparisons")

    # Load cross-task data
    csv_file_cross = data_dir / 'cka_summary.csv'
    if not csv_file_cross.exists():
        print(f"Error: {csv_file_cross} not found. Run collect_cka_trends_data.py first.")
        return

    df_cross = pd.read_csv(csv_file_cross)
    # Filter to non-overlapping cross-task comparisons only
    df_cross = df_cross[~df_cross['training_overlap']]
    print(f"Loaded {len(df_cross)} cross-task CKA comparisons (non-overlapping)")

    # Filter to layer 5 only
    df_same = df_same[df_same['layer'] == 5]
    df_cross = df_cross[df_cross['layer'] == 5]
    print(f"Filtered to layer 5: {len(df_same)} same-task, {len(df_cross)} cross-task")

    # Print summary statistics
    print("\nSummary statistics (Layer 5):")
    print("=" * 60)
    print("\nSame-task (different seeds):")
    for prefix in ['pt1', 'pt2', 'pt3']:
        prefix_data = df_same[df_same['prefix'] == prefix]
        if len(prefix_data) == 0:
            continue

        mean_cka = prefix_data['final_cka'].mean()
        sem_cka = prefix_data['final_cka'].std() / np.sqrt(len(prefix_data))
        print(f"{prefix.upper()}: {mean_cka:.4f} ± {sem_cka:.4f} (n={len(prefix_data)})")

    print("\nCross-task (different tasks, non-overlapping):")
    for prefix in ['pt1', 'pt2', 'pt3']:
        prefix_data = df_cross[df_cross['prefix'] == prefix]
        if len(prefix_data) == 0:
            print(f"{prefix.upper()}: No data")
            continue

        mean_cka = prefix_data['final_cka'].mean()
        sem_cka = prefix_data['final_cka'].std() / np.sqrt(len(prefix_data))
        print(f"{prefix.upper()}: {mean_cka:.4f} ± {sem_cka:.4f} (n={len(prefix_data)})")

    # Create plots
    print("\nCreating plots...")
    plot_same_task_trends(df_same, df_cross, data_dir)
    plot_same_task_only(df_same, data_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
