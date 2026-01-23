#!/usr/bin/env python3
"""
Generate 21×21 CKA matrix for PT3 layer 5.
21 experiments: 7 models × 3 seeds (orig, seed1, seed2)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.colors import FuncNorm

def three_slope_mapping(x, breakpoint1=0.4, breakpoint2=0.6):
    """
    Piecewise linear mapping with three segments:
    - 0.0 to breakpoint1: compressed (maps to 0.0-0.2)
    - breakpoint1 to breakpoint2: medium (maps to 0.2-0.5)
    - breakpoint2 to 1.0: expanded (maps to 0.5-1.0)
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # Segment 1: 0.0 to 0.4 -> 0.0 to 0.2
    mask1 = x <= breakpoint1
    result[mask1] = (x[mask1] / breakpoint1) * 0.2

    # Segment 2: 0.4 to 0.6 -> 0.2 to 0.5
    mask2 = (x > breakpoint1) & (x <= breakpoint2)
    result[mask2] = 0.2 + ((x[mask2] - breakpoint1) / (breakpoint2 - breakpoint1)) * 0.3

    # Segment 3: 0.6 to 1.0 -> 0.5 to 1.0
    mask3 = x > breakpoint2
    result[mask3] = 0.5 + ((x[mask3] - breakpoint2) / (1.0 - breakpoint2)) * 0.5

    return result

def three_slope_inverse(x, breakpoint1=0.4, breakpoint2=0.6):
    """Inverse mapping for three-slope normalization."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # Inverse segment 1: 0.0 to 0.2 -> 0.0 to 0.4
    mask1 = x <= 0.2
    result[mask1] = (x[mask1] / 0.2) * breakpoint1

    # Inverse segment 2: 0.2 to 0.5 -> 0.4 to 0.6
    mask2 = (x > 0.2) & (x <= 0.5)
    result[mask2] = breakpoint1 + ((x[mask2] - 0.2) / 0.3) * (breakpoint2 - breakpoint1)

    # Inverse segment 3: 0.5 to 1.0 -> 0.6 to 1.0
    mask3 = x > 0.5
    result[mask3] = breakpoint2 + ((x[mask3] - 0.5) / 0.5) * (1.0 - breakpoint2)

    return result

def load_cka_matrix(base_path):
    """Load CKA values from all PT3 pairs."""
    cka_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all'

    # Define experiments
    variants = list(range(1, 8))  # pt3-1 through pt3-7
    seeds = ['orig', 'seed1', 'seed2']

    experiments = []
    for var in variants:
        for seed in seeds:
            if seed == 'orig':
                exp_name = f'pt3-{var}'
            else:
                exp_name = f'pt3-{var}_{seed}'
            experiments.append(exp_name)

    n = len(experiments)
    matrix = np.full((n, n), np.nan)

    # Fill diagonal with 1.0
    np.fill_diagonal(matrix, 1.0)

    # Load CKA values
    for i, exp1 in enumerate(experiments):
        for j, exp2 in enumerate(experiments):
            if i >= j:
                continue

            # Try both orderings
            pair_name = f'pt3-{exp1.split("-")[1]}_vs_pt3-{exp2.split("-")[1]}'
            pair_dir = cka_dir / pair_name / 'layer5'

            summary_file = pair_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                cka_value = data['final_cka']
                matrix[i, j] = cka_value
                matrix[j, i] = cka_value

    return matrix, experiments

def plot_21x21_matrix(matrix, experiments, output_dir):
    """Plot the 21×21 CKA matrix."""

    # Create labels (variant-seed format)
    labels = []
    for exp in experiments:
        if '_seed' in exp:
            var, seed = exp.rsplit('_seed', 1)
            var_num = var.split('-')[1]
            labels.append(f'{var_num}-s{seed}')
        else:
            var_num = exp.split('-')[1]
            labels.append(f'{var_num}-o')

    fig, ax = plt.subplots(figsize=(14, 12))

    # Use non-linear normalization like exp4
    norm = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)

    # Plot heatmap
    im = ax.imshow(matrix, cmap='magma', norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=14)

    # Set ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    ax.set_title('PT3 CKA Matrix (21×21, Layer 5)', fontsize=16, fontweight='bold', pad=20)

    # Add grid lines to separate models
    for i in range(3, 21, 3):
        ax.axhline(i - 0.5, color='white', linewidth=2)
        ax.axvline(i - 0.5, color='white', linewidth=2)

    plt.tight_layout()

    output_file = output_dir / 'pt3_cka_21x21_l5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Save matrix as CSV
    csv_file = output_dir / 'pt3_cka_21x21_l5.csv'
    np.savetxt(csv_file, matrix, delimiter=',', fmt='%.6f')
    print(f"Saved: {csv_file}")

def plot_7x7_averaged(matrix, experiments, output_dir):
    """Average across seeds to create 7×7 matrix."""

    # Group by variant
    variants = list(range(1, 8))
    n_var = len(variants)

    avg_matrix = np.zeros((n_var, n_var))
    sem_matrix = np.zeros((n_var, n_var))

    for i, var1 in enumerate(variants):
        for j, var2 in enumerate(variants):
            # Get all CKA values for this variant pair (across seed combinations)
            values = []

            for seed1_idx in range(3):
                for seed2_idx in range(3):
                    exp1_idx = (var1 - 1) * 3 + seed1_idx
                    exp2_idx = (var2 - 1) * 3 + seed2_idx

                    val = matrix[exp1_idx, exp2_idx]
                    if not np.isnan(val):
                        values.append(val)

            if len(values) > 0:
                avg_matrix[i, j] = np.mean(values)
                sem_matrix[i, j] = np.std(values) / np.sqrt(len(values))

    # Plot without SEM
    fig, ax = plt.subplots(figsize=(10, 9))

    labels = [str(i) for i in variants]

    norm = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)

    im = ax.imshow(avg_matrix, cmap='magma', norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=14)

    # Add annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{avg_matrix[i, j]:.3f}',
                          ha="center", va="center", color="white" if avg_matrix[i, j] > 0.5 else "black",
                          fontsize=24)

    # Set ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_yticklabels(labels, fontsize=24)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_yticklabels(labels, fontsize=24)

    plt.tight_layout()

    output_file = output_dir / 'pt3_cka_7x7_averaged_l5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Plot with SEM
    fig, ax = plt.subplots(figsize=(10, 9))

    # Create annotations with mean ± sem
    annot_array = np.empty_like(avg_matrix, dtype=object)
    for i in range(n_var):
        for j in range(n_var):
            annot_array[i, j] = f'{avg_matrix[i, j]:.3f}\n±{sem_matrix[i, j]:.3f}'

    norm = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)

    im = ax.imshow(avg_matrix, cmap='magma', norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=14)

    # Add annotations with SEM
    for i in range(n_var):
        for j in range(n_var):
            text = ax.text(j, i, annot_array[i, j],
                          ha="center", va="center", color="white" if avg_matrix[i, j] > 0.5 else "black",
                          fontsize=18)

    # Set ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_yticklabels(labels, fontsize=24)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_yticklabels(labels, fontsize=24)

    plt.tight_layout()

    output_file = output_dir / 'pt3_cka_7x7_averaged_with_sem_l5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_intra_vs_inter_bar(matrix, experiments, output_dir):
    """Bar plot comparing intra-task vs inter-task CKA."""

    variants = list(range(1, 8))

    intra_task_values = []
    inter_task_values = []

    for i, var1 in enumerate(variants):
        for j, var2 in enumerate(variants):
            if i >= j:
                continue

            # Get all CKA values for this variant pair
            values = []
            for seed1_idx in range(3):
                for seed2_idx in range(3):
                    exp1_idx = (var1 - 1) * 3 + seed1_idx
                    exp2_idx = (var2 - 1) * 3 + seed2_idx

                    val = matrix[exp1_idx, exp2_idx]
                    if not np.isnan(val):
                        values.append(val)

            if var1 == var2:
                intra_task_values.extend(values)
            else:
                inter_task_values.extend(values)

    # Calculate statistics
    intra_mean = np.mean(intra_task_values)
    intra_sem = np.std(intra_task_values) / np.sqrt(len(intra_task_values))
    inter_mean = np.mean(inter_task_values)
    inter_sem = np.std(inter_task_values) / np.sqrt(len(inter_task_values))

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(intra_task_values, inter_task_values, equal_var=False)

    # Determine significance
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Intra-task\n(same model,\ndifferent seeds)',
                  'Inter-task\n(different models)']
    means = [intra_mean, inter_mean]
    sems = [intra_sem, inter_sem]

    bars = ax.bar(categories, means, yerr=sems, capsize=10,
                   color=['#1f77b4', '#ff7f0e'], alpha=0.8, width=0.6)

    # Add value labels on bars
    for bar, mean, sem in zip(bars, means, sems):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sem + 0.02,
                f'{mean:.3f}±{sem:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add significance bracket
    y_max = max(means[0] + sems[0], means[1] + sems[1])
    bracket_height = y_max + 0.08
    ax.plot([0, 0, 1, 1], [bracket_height, bracket_height + 0.02, bracket_height + 0.02, bracket_height],
            'k-', linewidth=1.5)
    ax.text(0.5, bracket_height + 0.03, sig, ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('CKA Similarity', fontsize=14)
    ax.set_title(f'PT3 Layer 5: Intra-task vs Inter-task CKA\n(p={p_value:.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / 'pt3_cka_intra_vs_inter_l5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    print(f"\nStatistics:")
    print(f"  Intra-task: {intra_mean:.3f} ± {intra_sem:.3f} (n={len(intra_task_values)})")
    print(f"  Inter-task: {inter_mean:.3f} ± {inter_sem:.3f} (n={len(inter_task_values)})")
    print(f"  p-value: {p_value:.4f} ({sig})")

def main():
    base_path = Path(__file__).resolve().parents[2]
    output_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PT3 CKA matrix...")
    matrix, experiments = load_cka_matrix(base_path)

    # Check for missing values
    n_missing = np.sum(np.isnan(matrix)) - 21  # Exclude diagonal
    if n_missing > 0:
        print(f"Warning: {n_missing} CKA values are missing!")

    print("\nGenerating plots...")
    plot_21x21_matrix(matrix, experiments, output_dir)
    plot_7x7_averaged(matrix, experiments, output_dir)
    plot_intra_vs_inter_bar(matrix, experiments, output_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()
