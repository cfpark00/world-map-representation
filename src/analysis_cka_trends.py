#!/usr/bin/env python3
"""
Generate CKA trends plot across different pretraining configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

# Define training data for each model
TRAINING_DATA = {
    'pt1': {
        1: {'distance'},
        2: {'trianglearea'},
        3: {'angle'},
        4: {'compass'},
        5: {'inside'},
        6: {'perimeter'},
        7: {'crossing'}
    },
    'pt2': {
        1: {'distance', 'trianglearea'},
        2: {'angle', 'compass'},
        3: {'inside', 'perimeter'},
        4: {'crossing', 'distance'},
        5: {'trianglearea', 'angle'},
        6: {'compass', 'inside'},
        7: {'perimeter', 'crossing'},
        8: {'distance', 'angle'}  # FIXED: Was incorrectly same as pt2-1
    },
    'pt3': {
        1: {'distance', 'angle', 'inside'},
        2: {'compass', 'perimeter', 'crossing'},
        3: {'crossing', 'trianglearea', 'distance'},
        4: {'angle', 'compass', 'inside'},
        5: {'perimeter', 'crossing', 'trianglearea'},
        6: {'trianglearea', 'distance', 'angle'},
        7: {'inside', 'perimeter', 'compass'},
        8: {'distance', 'trianglearea', 'compass'}  # FIXED: Was incorrectly same as pt3-1
    }
}

def has_overlap(prefix, exp1_num, exp2_num):
    """Check if two experiments have overlapping training data."""
    tasks1 = TRAINING_DATA[prefix][exp1_num]
    tasks2 = TRAINING_DATA[prefix][exp2_num]
    return bool(tasks1 & tasks2) or tasks1 == tasks2

def has_distance_task(prefix, exp_num):
    """Check if an experiment includes 'distance' in its training data."""
    return 'distance' in TRAINING_DATA[prefix][exp_num]

def calculate_cka_statistics(summary_df):
    """Calculate CKA averages and standard errors for each configuration and layer."""

    results = {
        'pt1_all': {'means': {}, 'stds': {}, 'counts': {}, 'values': {}},
        'pt2_non_overlap': {'means': {}, 'stds': {}, 'counts': {}, 'values': {}},
        'pt3_non_overlap': {'means': {}, 'stds': {}, 'counts': {}, 'values': {}}
    }

    layers = [3, 4, 5, 6]

    for layer in layers:
        # PT1: All off-diagonal pairs (including pt1-7)
        pt1_df = summary_df[(summary_df['prefix'] == 'pt1') &
                            (summary_df['layer'] == layer)]
        pt1_values = pt1_df['final_cka'].values
        results['pt1_all']['means'][layer] = np.mean(pt1_values)
        results['pt1_all']['stds'][layer] = np.std(pt1_values)
        results['pt1_all']['counts'][layer] = len(pt1_values)
        results['pt1_all']['values'][layer] = pt1_values

        # PT2: Non-overlapping pairs only
        pt2_df = summary_df[(summary_df['prefix'] == 'pt2') &
                            (summary_df['layer'] == layer)]
        pt2_non_overlap = []
        for _, row in pt2_df.iterrows():
            exp1_num = int(row['exp1'].split('-')[1])
            exp2_num = int(row['exp2'].split('-')[1])
            if not has_overlap('pt2', exp1_num, exp2_num):
                pt2_non_overlap.append(row['final_cka'])

        if pt2_non_overlap:
            results['pt2_non_overlap']['means'][layer] = np.mean(pt2_non_overlap)
            results['pt2_non_overlap']['stds'][layer] = np.std(pt2_non_overlap)
            results['pt2_non_overlap']['counts'][layer] = len(pt2_non_overlap)
            results['pt2_non_overlap']['values'][layer] = np.array(pt2_non_overlap)
        else:
            results['pt2_non_overlap']['means'][layer] = 0
            results['pt2_non_overlap']['stds'][layer] = 0
            results['pt2_non_overlap']['counts'][layer] = 0
            results['pt2_non_overlap']['values'][layer] = np.array([])

        # PT3: Non-overlapping pairs only
        pt3_df = summary_df[(summary_df['prefix'] == 'pt3') &
                            (summary_df['layer'] == layer)]
        pt3_non_overlap = []
        for _, row in pt3_df.iterrows():
            exp1_num = int(row['exp1'].split('-')[1])
            exp2_num = int(row['exp2'].split('-')[1])
            if not has_overlap('pt3', exp1_num, exp2_num):
                pt3_non_overlap.append(row['final_cka'])

        if pt3_non_overlap:
            results['pt3_non_overlap']['means'][layer] = np.mean(pt3_non_overlap)
            results['pt3_non_overlap']['stds'][layer] = np.std(pt3_non_overlap)
            results['pt3_non_overlap']['counts'][layer] = len(pt3_non_overlap)
            results['pt3_non_overlap']['values'][layer] = np.array(pt3_non_overlap)
        else:
            results['pt3_non_overlap']['means'][layer] = 0
            results['pt3_non_overlap']['stds'][layer] = 0
            results['pt3_non_overlap']['counts'][layer] = 0
            results['pt3_non_overlap']['values'][layer] = np.array([])

    return results

def plot_cka_trends(results, output_dir):
    """Create the CKA trends plot with all individual points and error bars."""

    fig, ax = plt.subplots(figsize=(10, 6))

    layers = [3, 4, 5, 6]
    x_positions = np.arange(3)  # Only 3 configurations now
    x_labels = ['PT1', 'PT2', 'PT3']  # Removed asterisk since we include all

    # Colors for each layer
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # Jitter width for scatter points
    jitter = 0.03

    for i, layer in enumerate(layers):
        y_values = [
            results['pt1_all']['means'][layer],
            results['pt2_non_overlap']['means'][layer],
            results['pt3_non_overlap']['means'][layer]
        ]

        # Calculate standard errors
        y_errors = [
            results['pt1_all']['stds'][layer] / np.sqrt(results['pt1_all']['counts'][layer]),
            results['pt2_non_overlap']['stds'][layer] / np.sqrt(results['pt2_non_overlap']['counts'][layer]),
            results['pt3_non_overlap']['stds'][layer] / np.sqrt(results['pt3_non_overlap']['counts'][layer])
        ]

        # Plot individual points with jitter
        configs = ['pt1_all', 'pt2_non_overlap', 'pt3_non_overlap']
        for j, config in enumerate(configs):
            values = results[config]['values'][layer]
            if len(values) > 0:
                # Add small random jitter to x position for visibility
                x_jittered = np.full(len(values), x_positions[j]) + np.random.normal(0, jitter, len(values))
                ax.scatter(x_jittered, values,
                          color=colors[i],
                          alpha=0.3,
                          s=20,
                          zorder=1)

        # Plot mean line with error bars
        ax.errorbar(x_positions, y_values,
                    yerr=y_errors,
                    marker=markers[i],
                    color=colors[i],
                    linewidth=2,
                    markersize=8,
                    label=f'Layer {layer}',
                    alpha=0.9,
                    capsize=5,
                    capthick=2,
                    zorder=2)

    # Formatting
    ax.set_xlabel('Pretraining Configuration', fontsize=12)
    ax.set_ylabel('CKA (Final Checkpoint)', fontsize=12)
    ax.set_title('CKA Trends Across Pretraining Configurations\n(Non-overlapping pairs only, mean ± SE)',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='best', framealpha=0.9)

    # Set tick label size and padding
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)

    # Remove top and right spines, thicken bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # No annotation needed since we include all pairs

    # Save the plot
    plot_path = output_dir / 'cka_trends.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save without labels
    fig_nolabel, ax_nolabel = plt.subplots(figsize=(10, 6))
    for i, layer in enumerate(layers):
        y_values = [
            results['pt1_all']['means'][layer],
            results['pt2_non_overlap']['means'][layer],
            results['pt3_non_overlap']['means'][layer]
        ]

        y_errors = [
            results['pt1_all']['stds'][layer] / np.sqrt(results['pt1_all']['counts'][layer]),
            results['pt2_non_overlap']['stds'][layer] / np.sqrt(results['pt2_non_overlap']['counts'][layer]),
            results['pt3_non_overlap']['stds'][layer] / np.sqrt(results['pt3_non_overlap']['counts'][layer])
        ]

        # Plot individual points with jitter (no labels version)
        configs = ['pt1_all', 'pt2_non_overlap', 'pt3_non_overlap']
        for j, config in enumerate(configs):
            values = results[config]['values'][layer]
            if len(values) > 0:
                x_jittered = np.full(len(values), x_positions[j]) + np.random.normal(0, jitter, len(values))
                ax_nolabel.scatter(x_jittered, values,
                                  color=colors[i],
                                  alpha=0.3,
                                  s=20,
                                  zorder=1)

        ax_nolabel.errorbar(x_positions, y_values,
                           yerr=y_errors,
                           marker=markers[i],
                           color=colors[i],
                           linewidth=2,
                           markersize=8,
                           alpha=0.9,
                           capsize=5,
                           capthick=2,
                           zorder=2)

    # Keep ticks and labels in nolabel version, just remove titles
    ax_nolabel.set_xticks(x_positions)
    ax_nolabel.set_xticklabels(x_labels, fontsize=14)
    ax_nolabel.set_ylim(0, 1)

    # Set tick label size and padding
    ax_nolabel.tick_params(axis='both', which='major', labelsize=14, pad=8)

    # Remove top and right spines, thicken bottom and left
    ax_nolabel.spines['top'].set_visible(False)
    ax_nolabel.spines['right'].set_visible(False)
    ax_nolabel.spines['bottom'].set_linewidth(1.5)
    ax_nolabel.spines['left'].set_linewidth(1.5)

    plot_nolabel_path = output_dir / 'cka_trends_nolabel.png'
    plt.tight_layout()
    plt.savefig(plot_nolabel_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path, plot_nolabel_path

def main():
    # Load config
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'input_dir': 'scratch/cka_analysis_clean',
            'output_dir': 'data/cka_matrices'
        }

    # Setup paths
    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    summary_df = pd.read_csv(input_dir / 'cka_summary.csv')

    print("Calculating CKA statistics...")
    results = calculate_cka_statistics(summary_df)

    # Print the values for verification
    print("\nCKA Statistics (Final Checkpoint):")
    print("-" * 50)
    for config_name in ['pt1_all', 'pt2_non_overlap', 'pt3_non_overlap']:
        config_label = config_name.replace('_all', '').replace('_non_overlap', '')
        print(f"\n{config_label}:")
        for layer in sorted(results[config_name]['means'].keys()):
            mean = results[config_name]['means'][layer]
            std = results[config_name]['stds'][layer]
            count = results[config_name]['counts'][layer]
            se = std / np.sqrt(count) if count > 0 else 0
            print(f"  Layer {layer}: {mean:.4f} ± {se:.4f} (n={count})")

    # Create the plot
    print("\nGenerating plots...")
    plot_path, plot_nolabel_path = plot_cka_trends(results, output_dir)

    print(f"\nPlots saved to:")
    print(f"  - {plot_path}")
    print(f"  - {plot_nolabel_path}")

if __name__ == '__main__':
    main()