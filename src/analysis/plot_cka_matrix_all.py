#!/usr/bin/env python3
"""
Plot CKA matrices for pt1, pt2, and pt3 experiments.
Supports 7x7 for pt1 and 8x8 for pt2/pt3.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml
import re

# Task mappings for each experiment group
TASK_MAPPINGS = {
    'pt1': {
        1: 'distance',
        2: 'trianglearea',
        3: 'angle',
        4: 'compass',
        5: 'inside',
        6: 'perimeter',
        7: 'crossing'
    },
    'pt2': {
        1: 'distance',
        2: 'angle',
        3: 'inside',
        4: 'crossing',
        5: 'trianglearea',
        6: 'compass',
        7: 'perimeter',
        8: 'distance'
    },
    'pt3': {
        1: 'distance',
        2: 'compass',
        3: 'crossing',
        4: 'angle',
        5: 'perimeter',
        6: 'trianglearea',
        7: 'inside',
        8: 'distance'
    }
}

def get_non_overlapping_pairs(prefix):
    """
    Get pairs of experiments that don't share any training data.
    Returns a set of (i, j) tuples where i and j are 1-indexed experiment numbers.
    """
    # Define which datasets each experiment contains
    dataset_mappings = {
        'pt2': {
            1: {'distance', 'trianglearea'},
            2: {'angle', 'compass'},
            3: {'inside', 'perimeter'},
            4: {'crossing', 'distance'},
            5: {'trianglearea', 'angle'},
            6: {'compass', 'inside'},
            7: {'perimeter', 'crossing'},
            8: {'distance', 'angle'}
        },
        'pt3': {
            1: {'distance', 'trianglearea', 'angle'},
            2: {'compass', 'inside', 'perimeter'},
            3: {'crossing', 'distance', 'trianglearea'},
            4: {'angle', 'compass', 'inside'},
            5: {'perimeter', 'crossing', 'distance'},
            6: {'trianglearea', 'angle', 'compass'},
            7: {'inside', 'perimeter', 'crossing'},
            8: {'distance', 'trianglearea', 'angle'}
        }
    }

    if prefix not in dataset_mappings:
        return set()

    datasets = dataset_mappings[prefix]
    non_overlap_pairs = set()

    for i in range(1, len(datasets) + 1):
        for j in range(i + 1, len(datasets) + 1):
            # Check if they share any datasets
            shared = datasets[i] & datasets[j]
            if not shared:
                non_overlap_pairs.add((i, j))

    return non_overlap_pairs


def get_task_mapping(prefix):
    """
    Get task mapping for a given prefix by reading data generation configs.
    Falls back to hardcoded mappings if configs not found.
    """
    try:
        task_map = {}
        for i in range(1, 9):  # Check up to 8
            config_path = Path(f'/configs/data_generation/ftset/combine_{prefix}-{i}.yaml')
            if not config_path.exists():
                break

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                first_dataset = config['datasets'][0]['path']
                task = re.search(r'/([^/]+)_1M', first_dataset).group(1)
                task_map[i] = task

        if task_map:
            return task_map
    except Exception as e:
        print(f"Could not read configs for {prefix}: {e}")

    # Fall back to hardcoded mappings
    return TASK_MAPPINGS.get(prefix, {})

def plot_cka_matrix(prefix='pt1', output_dir=None, show_plot=True, layer=5):
    """
    Plot CKA matrix for a given experiment prefix.

    Args:
        prefix: 'pt1', 'pt2', or 'pt3'
        output_dir: Directory to save output (defaults to scratch/cka_analysis_{prefix})
        show_plot: Whether to display the plot
        layer: Which layer to analyze (default: 5)
    """

    # Get task mapping
    task_mapping = get_task_mapping(prefix)
    n_models = len(task_mapping)

    if n_models == 0:
        print(f"No models found for {prefix}")
        return

    models = [f'{prefix}-{i}' for i in range(1, n_models + 1)]
    tasks = [task_mapping[i] for i in range(1, n_models + 1)]

    print(f"\n{'='*50}")
    print(f"Creating {n_models}x{n_models} CKA matrix for {prefix}")
    print(f"{'='*50}")
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")

    # Base directory for CKA results
    base_dir = Path(f'/data/experiments/cka_analysis_{prefix}')
    if not base_dir.exists():
        base_dir = Path('/data/experiments/cka_analysis')  # Fallback for pt1

    # Initialize matrix
    cka_matrix = np.ones((n_models, n_models))  # Diagonal is 1 by definition

    # Load CKA values for each pair
    missing_pairs = []
    for i in range(n_models):
        for j in range(i+1, n_models):  # Only upper triangle
            model1 = models[i]
            model2 = models[j]

            # Try to load CKA results
            results_dir = base_dir / f'{model1}_vs_{model2}_l{layer}'
            results_file = results_dir / 'cka_results.json'

            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    # Use final CKA value
                    cka_value = data.get('final_cka', np.nan)
                    cka_matrix[i, j] = cka_value
                    cka_matrix[j, i] = cka_value  # Symmetric
                    print(f"  Loaded {model1} vs {model2}: CKA = {cka_value:.4f}")
            else:
                print(f"  Missing: {model1} vs {model2}")
                missing_pairs.append((model1, model2))
                cka_matrix[i, j] = np.nan
                cka_matrix[j, i] = np.nan

    if missing_pairs:
        print(f"\nWarning: {len(missing_pairs)} pairs missing!")

    # Create the plot
    fig, ax = plt.subplots(figsize=(11 if n_models == 8 else 10, 9 if n_models == 8 else 8))

    # Create heatmap
    sns.heatmap(cka_matrix,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'CKA'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray')

    # Set labels
    ax.set_xticklabels([f'{m}\n({t})' for m, t in zip(models, tasks)], rotation=45, ha='right')
    ax.set_yticklabels([f'{m}\n({t})' for m, t in zip(models, tasks)], rotation=0)

    # Set title
    title = f'CKA Matrix ({prefix.upper()}): Final Checkpoint Similarity\n'
    title += f'Layer {layer} Representations (All tokens)'
    ax.set_title(title, fontsize=14, pad=20)

    # Add text for missing values
    for i in range(n_models):
        for j in range(n_models):
            if np.isnan(cka_matrix[i, j]) and i != j:
                ax.text(j + 0.5, i + 0.5, 'N/A',
                       ha='center', va='center',
                       color='red', fontweight='bold')

    plt.tight_layout()

    # Save the plot
    if output_dir is None:
        output_dir = Path(f'/scratch/cka_analysis_{prefix}')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'cka_matrix_{prefix}_l{layer}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if show_plot:
        plt.show()

    # Print summary statistics
    print(f"\n{prefix.upper()} CKA Matrix Summary:")
    off_diagonal = cka_matrix[np.triu_indices(n_models, k=1)]

    # Handle NaN values properly
    valid_values = off_diagonal[~np.isnan(off_diagonal)]
    if len(valid_values) > 0:
        print(f"  Mean CKA (off-diagonal): {np.mean(valid_values):.4f}")
        print(f"  Min CKA: {np.min(valid_values):.4f}")
        print(f"  Max CKA: {np.max(valid_values):.4f}")
        print(f"  Std CKA: {np.std(valid_values):.4f}")
        print(f"  Valid pairs: {len(valid_values)}/{len(off_diagonal)}")
    else:
        print(f"  No valid CKA values computed yet!")

    # Compute special statistic for non-overlapping training data
    if prefix == 'pt1':
        # For pt1, all off-diagonal pairs have non-overlapping training (each uses single task)
        if len(valid_values) > 0:
            print(f"\n  SPECIAL: Mean CKA for non-overlapping training sets: {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}")
            print(f"           (ALL {len(valid_values)} off-diagonal pairs - each pt1 uses a single distinct task)")
        else:
            print(f"\n  SPECIAL: No CKA values computed yet")
            print(f"           (For pt1, all off-diagonal pairs have non-overlapping training)")
    elif prefix in ['pt2', 'pt3']:
        non_overlap_pairs = get_non_overlapping_pairs(prefix)
        non_overlap_cka_values = []

        for i in range(n_models):
            for j in range(i+1, n_models):
                if (i+1, j+1) in non_overlap_pairs:
                    val = cka_matrix[i, j]
                    if not np.isnan(val):
                        non_overlap_cka_values.append(val)

        if non_overlap_cka_values:
            print(f"\n  SPECIAL: Mean CKA for non-overlapping training sets: {np.mean(non_overlap_cka_values):.4f} ± {np.std(non_overlap_cka_values):.4f}")
            print(f"           ({len(non_overlap_cka_values)} pairs with no shared training data)")
        else:
            print(f"\n  SPECIAL: No CKA values available for non-overlapping pairs yet")
            print(f"           ({len(non_overlap_pairs)} pairs identified with no shared training data)")

    # Find most and least similar pairs
    min_val = np.inf
    max_val = -np.inf
    min_pair = None
    max_pair = None

    for i in range(n_models):
        for j in range(i+1, n_models):
            val = cka_matrix[i, j]
            if not np.isnan(val):
                if val < min_val:
                    min_val = val
                    min_pair = (models[i], models[j], tasks[i], tasks[j])
                if val > max_val:
                    max_val = val
                    max_pair = (models[i], models[j], tasks[i], tasks[j])

    if min_pair:
        print(f"  Least similar: {min_pair[0]} ({min_pair[2]}) vs {min_pair[1]} ({min_pair[3]}) - CKA = {min_val:.4f}")
    if max_pair:
        print(f"  Most similar: {max_pair[0]} ({max_pair[2]}) vs {max_pair[1]} ({max_pair[3]}) - CKA = {max_val:.4f}")

    return cka_matrix, models, tasks

def main():
    parser = argparse.ArgumentParser(description='Plot CKA matrix for experiments')
    parser.add_argument('--prefix', type=str, default='all',
                       choices=['pt1', 'pt2', 'pt3', 'all'],
                       help='Which experiment group to plot (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: scratch/cka_analysis_{prefix})')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--layer', type=int, default=5,
                       help='Which layer to analyze (default: 5)')

    args = parser.parse_args()

    if args.prefix == 'all':
        # Plot all three
        for prefix in ['pt1', 'pt2', 'pt3']:
            try:
                plot_cka_matrix(prefix, args.output_dir, not args.no_show, args.layer)
            except Exception as e:
                print(f"Error plotting {prefix}: {e}")
                continue
    else:
        plot_cka_matrix(args.prefix, args.output_dir, not args.no_show, args.layer)

if __name__ == "__main__":
    main()