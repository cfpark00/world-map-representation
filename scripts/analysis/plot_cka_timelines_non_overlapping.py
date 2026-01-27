#!/usr/bin/env python3
"""
Plot CKA timelines for non-overlapping training set pairs.
Creates 12 plots: 3 prefixes (pt1, pt2, pt3) x 4 layers (3, 4, 5, 6).
Each plot shows CKA evolution over training steps for pairs with no shared training data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

# Define which pairs have non-overlapping training data
NON_OVERLAPPING_PAIRS = {
    'pt1': 'all',  # All 21 pairs - each model trained on single distinct task
    'pt2': [  # 18 pairs - models trained on 2 tasks each
        (1, 2), (1, 3), (1, 6), (1, 7),  # pt2-1 (distance, trianglearea) doesn't overlap with these
        (2, 3), (2, 4), (2, 7),  # pt2-2 (angle, compass)
        (3, 4), (3, 5),  # pt2-3 (inside, perimeter)
        (4, 5), (4, 6),  # pt2-4 (crossing, distance)
        (5, 6), (5, 7),  # pt2-5 (trianglearea, angle)
        (6, 7), (6, 8),  # pt2-6 (compass, inside)
        (7, 8),  # pt2-7 (perimeter, crossing)
        (2, 5),  # Additional non-overlapping pairs
        (3, 6),
    ],
    'pt3': [  # 9 pairs - models trained on 3 tasks each
        (1, 2),  # pt3-1 (distance, angle, inside) vs pt3-2 (compass, perimeter, crossing)
        (1, 7),  # pt3-1 vs pt3-7 (inside, perimeter, compass)
        (2, 3),  # pt3-2 vs pt3-3 (crossing, trianglearea, distance)
        (2, 6),  # pt3-2 vs pt3-6 (trianglearea, distance, angle)
        (3, 4),  # pt3-3 vs pt3-4 (angle, compass, inside)
        (3, 7),  # pt3-3 vs pt3-7
        (4, 5),  # pt3-4 vs pt3-5 (perimeter, crossing, distance)
        (5, 6),  # pt3-5 vs pt3-6
        (5, 7),  # pt3-5 vs pt3-7 (added based on checking the sets)
    ]
}

# Correct the non-overlapping pairs based on actual training data
TRAINING_DATA = {
    'pt2': {
        1: {'distance', 'trianglearea'},
        2: {'angle', 'compass'},
        3: {'inside', 'perimeter'},
        4: {'crossing', 'distance'},
        5: {'trianglearea', 'angle'},
        6: {'compass', 'inside'},
        7: {'perimeter', 'crossing'},
        8: {'distance', 'trianglearea'}  # Same as pt2-1
    },
    'pt3': {
        1: {'distance', 'angle', 'inside'},
        2: {'compass', 'perimeter', 'crossing'},
        3: {'crossing', 'trianglearea', 'distance'},
        4: {'angle', 'compass', 'inside'},
        5: {'perimeter', 'crossing', 'trianglearea'},
        6: {'trianglearea', 'distance', 'angle'},
        7: {'inside', 'perimeter', 'compass'},
        8: {'distance', 'angle', 'inside'}  # Same as pt3-1
    }
}

def get_non_overlapping_pairs(prefix):
    """Get pairs with non-overlapping training data."""
    if prefix == 'pt1':
        # All pairs for pt1 (each trained on single task)
        return [(i, j) for i in range(1, 8) for j in range(i+1, 8)]

    # Calculate non-overlapping pairs dynamically
    datasets = TRAINING_DATA[prefix]
    non_overlap = []
    n = 8 if prefix in ['pt2', 'pt3'] else 7

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if not (datasets[i] & datasets[j]):  # No intersection
                non_overlap.append((i, j))

    return non_overlap

def plot_timeline_for_config(prefix, layer, checkpoint_df, output_dir):
    """Create a timeline plot for a specific prefix-layer combination."""

    # Filter data for this prefix and layer
    df = checkpoint_df[(checkpoint_df['prefix'] == prefix) &
                       (checkpoint_df['layer'] == layer)].copy()

    if df.empty:
        print(f"No data for {prefix} layer {layer}")
        return

    # Filter for non-overlapping pairs only (use the training_overlap column)
    non_overlap_df = df[~df['training_overlap']].copy()

    if non_overlap_df.empty:
        print(f"No non-overlapping pairs for {prefix} layer {layer}")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique pairs and assign colors
    unique_pairs = non_overlap_df[['exp1', 'exp2']].drop_duplicates()
    n_pairs = len(unique_pairs)
    colors = cm.get_cmap('tab20' if n_pairs <= 20 else 'hsv')(np.linspace(0, 1, n_pairs))

    # Plot each pair
    for idx, (_, pair_row) in enumerate(unique_pairs.iterrows()):
        exp1, exp2 = pair_row['exp1'], pair_row['exp2']

        # Get data for this pair
        pair_data = non_overlap_df[(non_overlap_df['exp1'] == exp1) &
                                   (non_overlap_df['exp2'] == exp2)]

        # Sort by checkpoint
        pair_data = pair_data.sort_values('checkpoint')

        # Plot with label showing the tasks
        label = f"{exp1} vs {exp2}"
        if 'task1' in pair_data.columns and 'task2' in pair_data.columns:
            task1 = pair_data['task1'].iloc[0]
            task2 = pair_data['task2'].iloc[0]
            label = f"{exp1} ({task1}) vs {exp2} ({task2})"

        ax.plot(pair_data['checkpoint'], pair_data['cka'],
               label=label, color=colors[idx], linewidth=1.5, alpha=0.8)

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Training Steps (log scale)', fontsize=12)
    ax.set_ylabel('CKA', fontsize=12)
    ax.set_title(f'{prefix.upper()} - Layer {layer}\nCKA Evolution for Non-Overlapping Training Set Pairs',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)

    # Add legend
    if n_pairs <= 10:
        ax.legend(loc='best', fontsize=9, ncol=1)
    else:
        # For many pairs, put legend outside
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        plt.tight_layout()

    # Add text annotation for number of pairs
    ax.text(0.02, 0.98, f'{n_pairs} non-overlapping pairs',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save plot
    output_path = output_dir / f'cka_timeline_{prefix}_l{layer}_non_overlap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    # Print summary statistics
    final_cka = non_overlap_df.groupby(['exp1', 'exp2'])['cka'].last()
    print(f"  {prefix.upper()} L{layer}: {n_pairs} pairs, "
          f"mean final CKA = {final_cka.mean():.4f} ± {final_cka.std():.4f}")

def main():
    # Load checkpoint data
    data_dir = Path('/scratch/cka_analysis_clean')
    checkpoint_df = pd.read_csv(data_dir / 'cka_checkpoints.csv')

    print(f"Loaded {len(checkpoint_df)} checkpoint measurements")
    print(f"Unique experiments: {checkpoint_df['prefix'].unique()}")
    print(f"Layers: {sorted(checkpoint_df['layer'].unique())}")
    print()

    # Create output directory
    output_dir = Path('/scratch/cka_analysis_clean/timelines_non_overlap')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create grid plot for overview
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('CKA Evolution for Non-Overlapping Training Set Pairs', fontsize=16, fontweight='bold')

    prefixes = ['pt1', 'pt2', 'pt3']
    layers = [3, 4, 5, 6]

    for i, prefix in enumerate(prefixes):
        for j, layer in enumerate(layers):
            ax = axes[i, j]

            # Get non-overlapping pairs
            non_overlap_pairs = get_non_overlapping_pairs(prefix)

            # Filter data
            df = checkpoint_df[(checkpoint_df['prefix'] == prefix) &
                              (checkpoint_df['layer'] == layer)].copy()

            if not df.empty:
                # Use the training_overlap column directly
                non_overlap_df = df[~df['training_overlap']]

                if not non_overlap_df.empty:
                    # Plot all pairs with transparency
                    for (exp1, exp2), group in non_overlap_df.groupby(['exp1', 'exp2']):
                        group = group.sort_values('checkpoint')
                        ax.plot(group['checkpoint'], group['cka'],
                               linewidth=1, alpha=0.5)

                    # Add mean line
                    mean_cka = non_overlap_df.groupby('checkpoint')['cka'].mean()
                    ax.plot(mean_cka.index, mean_cka.values,
                           'k-', linewidth=2, label='Mean', alpha=0.8)

                    n_pairs = len(non_overlap_df[['exp1', 'exp2']].drop_duplicates())
                    ax.set_title(f'{prefix.upper()} L{layer} ({n_pairs} pairs)', fontsize=10)
                else:
                    ax.set_title(f'{prefix.upper()} L{layer} (No data)', fontsize=10)
            else:
                ax.set_title(f'{prefix.upper()} L{layer} (No data)', fontsize=10)

            ax.set_xscale('log')
            ax.set_xlabel('Steps (log)' if i == 2 else '', fontsize=9)
            ax.set_ylabel('CKA' if j == 0 else '', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    overview_path = output_dir / 'cka_timelines_overview.png'
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved overview: {overview_path}")

    # Create individual plots for each prefix-layer combination
    print("\nCreating individual timeline plots...")
    for prefix in prefixes:
        for layer in layers:
            plot_timeline_for_config(prefix, layer, checkpoint_df, output_dir)

    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == '__main__':
    main()