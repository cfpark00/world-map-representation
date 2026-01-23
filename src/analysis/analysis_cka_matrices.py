#!/usr/bin/env python3
"""
CKA matrix analysis module for creating final vs max CKA matrices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def get_overlap_type(prefix, exp1_num, exp2_num):
    """Determine the type of overlap between two experiments."""
    if exp1_num == exp2_num:
        return 'full'  # Diagonal - same model

    tasks1 = TRAINING_DATA[prefix][exp1_num]
    tasks2 = TRAINING_DATA[prefix][exp2_num]

    # Check if sets are identical (e.g., pt2-1 and pt2-8)
    if tasks1 == tasks2:
        return 'full'

    # Check for partial overlap
    if tasks1 & tasks2:  # Intersection exists
        return 'partial'

    return 'none'

def create_dual_cka_matrix(prefix, summary_df, checkpoint_df, layer):
    """Create CKA matrix with final values in lower triangle and max values in upper triangle."""

    # Filter summary for final values
    summary = summary_df[(summary_df['prefix'] == prefix) &
                         (summary_df['layer'] == layer)].copy()

    # Filter checkpoints for max values
    checkpoints = checkpoint_df[(checkpoint_df['prefix'] == prefix) &
                               (checkpoint_df['layer'] == layer)].copy()

    if summary.empty or checkpoints.empty:
        print(f"No data for {prefix} layer {layer}")
        return None, None, None

    # Get number of experiments
    n_exps = 8 if prefix in ['pt2', 'pt3'] else 7

    # Initialize matrices
    final_matrix = np.eye(n_exps)  # Diagonal is 1.0 (self-similarity)
    max_matrix = np.eye(n_exps)
    overlap_matrix = np.zeros((n_exps, n_exps), dtype=object)

    # Fill diagonal with 'full'
    for i in range(n_exps):
        overlap_matrix[i, i] = 'full'

    # Fill the matrices with final values from summary
    for _, row in summary.iterrows():
        exp1_num = int(row['exp1'].split('-')[1])
        exp2_num = int(row['exp2'].split('-')[1])
        final_cka = row['final_cka']

        # Convert to 0-indexed
        i, j = exp1_num - 1, exp2_num - 1

        # Fill final values
        final_matrix[i, j] = final_cka
        final_matrix[j, i] = final_cka

        # Determine overlap type
        overlap_type = get_overlap_type(prefix, exp1_num, exp2_num)
        overlap_matrix[i, j] = overlap_type
        overlap_matrix[j, i] = overlap_type

    # Calculate max values from checkpoint data
    max_values = checkpoints.groupby(['exp1', 'exp2'])['cka'].max()
    for (exp1, exp2), max_cka in max_values.items():
        exp1_num = int(exp1.split('-')[1])
        exp2_num = int(exp2.split('-')[1])
        i, j = exp1_num - 1, exp2_num - 1
        max_matrix[i, j] = max_cka
        max_matrix[j, i] = max_cka

    return final_matrix, max_matrix, overlap_matrix

def plot_dual_cka_matrix(prefix, final_matrix, max_matrix, overlap_matrix, layer, ax=None, with_labels=True):
    """Plot CKA matrix with final values (symmetric)."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    n = final_matrix.shape[0]

    # Use final_matrix directly (it's already symmetric)
    display_matrix = final_matrix.copy()
    annotations = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal
                annotations[i, j] = '1.00'
            else:
                # Show final CKA values
                annotations[i, j] = f'{final_matrix[i, j]:.2f}'

    # Plot heatmap - always show numbers with bigger font
    sns.heatmap(display_matrix,
                annot=annotations,  # Always show annotations
                fmt='',
                cmap='magma',
                vmin=0, vmax=1,
                square=True,
                cbar=with_labels,
                cbar_kws={'label': 'CKA'} if with_labels else {},
                ax=ax,
                linewidths=0.5,
                linecolor='gray',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    # Make colorbar tick labels and label bigger if colorbar exists
    if with_labels:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('CKA', fontsize=14)

    # Add overlap markers on both triangles for symmetry
    for i in range(n):
        for j in range(n):
            overlap = overlap_matrix[i, j]

            if overlap == 'full' and i != j:
                # Full overlap (e.g., pt2-1 and pt2-8) - red box
                ax.add_patch(plt.Rectangle((j + 0.85, i + 0.85), 0.1, 0.1,
                                          fill=True, color='red', zorder=10))
            elif overlap == 'partial':
                # Partial overlap - red triangle
                triangle = plt.Polygon([(j + 0.85, i + 0.95),
                                      (j + 0.95, i + 0.95),
                                      (j + 0.9, i + 0.85)],
                                     color='red', zorder=10)
                ax.add_patch(triangle)

    if with_labels:

        # Set labels
        labels = [f'{prefix}-{i+1}' for i in range(n)]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(f'{prefix.upper()} - Layer {layer} - Final CKA\n'
                    f'Red■=Full overlap | Red▲=Partial overlap',
                    fontsize=13)
    else:
        # No labels version
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    return ax

def main():
    # Load config
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'input_dir': 'scratch/cka_analysis_clean',
            'output_dir': 'data/cka_matrices',
            'layers': [3, 4, 5, 6]
        }

    # Setup paths
    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    summary_df = pd.read_csv(input_dir / 'cka_summary.csv')
    checkpoint_df = pd.read_csv(input_dir / 'cka_checkpoints.csv')

    print(f"Loaded {len(summary_df)} summary measurements")
    print(f"Loaded {len(checkpoint_df)} checkpoint measurements")

    prefixes = ['pt1', 'pt2', 'pt3']
    layers = config.get('layers', [3, 4, 5, 6])

    # Create matrices for all layers - both labeled and unlabeled versions
    for layer in layers:
        print(f"\nProcessing Layer {layer}...")

        # Create combined figure with labels
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(f'CKA Matrices - Layer {layer} - Final Checkpoint',
                    fontsize=16, fontweight='bold')

        # Create combined figure without labels
        fig_nolabel, axes_nolabel = plt.subplots(1, 3, figsize=(24, 7))

        for idx, prefix in enumerate(prefixes):
            final_matrix, max_matrix, overlap_matrix = create_dual_cka_matrix(
                prefix, summary_df, checkpoint_df, layer)

            if final_matrix is not None:
                # With labels
                ax = axes[idx]
                plot_dual_cka_matrix(prefix, final_matrix, max_matrix, overlap_matrix, layer, ax, with_labels=True)

                # Without labels
                ax_nolabel = axes_nolabel[idx]
                plot_dual_cka_matrix(prefix, final_matrix, max_matrix, overlap_matrix, layer, ax_nolabel, with_labels=False)

                # Save individual matrices
                for with_labels, suffix in [(True, ''), (False, '_nolabel')]:
                    individual_path = output_dir / f'cka_matrix_{prefix}_l{layer}{suffix}.png'
                    fig_individual, ax_individual = plt.subplots(figsize=(10, 8))
                    plot_dual_cka_matrix(prefix, final_matrix, max_matrix, overlap_matrix,
                                       layer, ax_individual, with_labels=with_labels)
                    plt.tight_layout()
                    plt.savefig(individual_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_individual)

        # Save combined figures
        plt.figure(fig.number)
        plt.tight_layout()
        combined_path = output_dir / f'cka_matrices_all_l{layer}_combined.png'
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(fig_nolabel.number)
        plt.tight_layout()
        combined_nolabel_path = output_dir / f'cka_matrices_all_l{layer}_combined_nolabel.png'
        plt.savefig(combined_nolabel_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {combined_path.name} and {combined_nolabel_path.name}")

    print(f"\nAll matrices saved to: {output_dir}/")

if __name__ == '__main__':
    main()