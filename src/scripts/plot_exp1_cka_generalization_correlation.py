#!/usr/bin/env python3
"""
Scatter plot showing correlation between CKA scores and generalization performance for exp1.

X-axis: CKA score between pt1-X and pt1-Y (from exp4, layer 5, averaged across seeds, excluding crossing)
Y-axis: FTWB1-X's performance on task Y after Atlantis fine-tuning (generalization, excluding crossing)

This tests whether representation similarity (CKA) between single-task models
predicts how well they generalize after fine-tuning on OOD data (Atlantis).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats
import math

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
EXP1_BASE = BASE_DIR / 'data/experiments/revision/exp1'
EXP4_CKA = BASE_DIR / 'data/experiments/revision/exp4/cka_analysis'
OUTPUT_DIR = EXP1_BASE / 'plots'

# Task mapping (excluding crossing=7)
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter"]
TASK_NUMS = [1, 2, 3, 4, 5, 6]  # Exclude 7 (crossing)

# Task acronyms for annotations
TASK_ACRONYMS = {
    'distance': 'D',
    'trianglearea': 'T',
    'angle': 'A',
    'compass': 'Co',
    'inside': 'I',
    'perimeter': 'P',
}

# Accuracy-based tasks (higher is better)
ACCURACY_TASKS = ['inside', 'compass']

# FTWB1 task mapping
FTWB1_TASKS = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
}

def load_cka_matrix_exp4():
    """Load 7x7 averaged CKA matrix from exp4, exclude crossing (row/col 7)."""
    print("Loading CKA matrix from exp4...")

    # Load full 7x7 matrix (skip first row and first column which are headers)
    csv_path = EXP4_CKA / 'cka_matrix_7x7_averaged_layer5.csv'
    df = pd.read_csv(csv_path, index_col=0)
    full_matrix = df.values

    # Extract 6x6 submatrix (excluding crossing = index 6)
    matrix_6x6 = full_matrix[:6, :6]

    print(f"  Loaded 6x6 CKA matrix (excluded crossing)")
    return matrix_6x6

def load_baselines(seed):
    """Load PT1 baseline performance for a specific seed."""
    baselines = {}
    baselines_atlantis = {}

    if seed == 'original':
        base_exp = Path("/data/experiments/pt1")
    else:
        base_exp = EXP1_BASE / f"pt1_seed{seed}"

    for task in TASKS:
        # Standard task baseline
        json_path = base_exp / "evals" / task / "eval_data" / "evaluation_results.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
            checkpoints = sorted(data.keys(), key=lambda x: int(x.split('-')[1]))
            last_checkpoint = checkpoints[-1]
            baselines[task] = data[last_checkpoint][f'eval_{task}_metric_mean']

        # Atlantis task baseline
        atlantis_path = base_exp / "evals" / f"atlantis_{task}" / "eval_data" / "evaluation_results.json"
        with open(atlantis_path, 'r') as f:
            data = json.load(f)
            checkpoints = sorted(data.keys(), key=lambda x: int(x.split('-')[1]))
            last_checkpoint = checkpoints[-1]
            baselines_atlantis[task] = data[last_checkpoint][f'eval_{task}_metric_mean']

    return baselines, baselines_atlantis

def load_ftwb1_performance(seed):
    """Load FTWB1 performance for 6 single-task models (excluding crossing)."""
    performance_dict = {}

    for ftwb1_num in range(1, 7):  # 1-6 only
        if seed == 'original':
            exp_name = f"pt1_ftwb1-{ftwb1_num}"
            exp_path = Path("/data/experiments") / exp_name
        else:
            exp_name = f"pt1_seed{seed}_ftwb1-{ftwb1_num}"
            exp_path = EXP1_BASE / exp_name

        performance_dict[ftwb1_num] = {}

        for task in TASKS:
            # Load Atlantis task performance
            atlantis_path = exp_path / "evals" / f"atlantis_{task}" / "eval_data" / "evaluation_results.json"
            with open(atlantis_path, 'r') as f:
                data = json.load(f)
                checkpoints = sorted(data.keys(), key=lambda x: int(x.split('-')[1]))
                last_checkpoint = checkpoints[-1]
                performance_dict[ftwb1_num][task] = data[last_checkpoint][f'eval_{task}_metric_mean']

    return performance_dict

def normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy=False):
    """Normalize metrics to 0-1 scale where 0=no improvement, 1=standard level."""
    if is_accuracy:
        if baseline_standard <= baseline_atlantis:
            return 0.0
        normalized = (value_atlantis - baseline_atlantis) / (baseline_standard - baseline_atlantis)
    else:
        # Lower is better (error metric) - use log-ratio
        if baseline_standard >= baseline_atlantis or baseline_atlantis <= 0 or value_atlantis <= 0:
            if value_atlantis >= baseline_atlantis:
                return 0.0
            elif value_atlantis <= baseline_standard:
                return 1.0
            else:
                normalized = (baseline_atlantis - value_atlantis) / (baseline_atlantis - baseline_standard)
        else:
            numerator = math.log(baseline_atlantis / value_atlantis)
            denominator = math.log(baseline_atlantis / baseline_standard)
            if denominator == 0:
                return 0.0
            normalized = numerator / denominator

    return max(0.0, min(1.5, normalized))

def create_generalization_matrix(seed):
    """Create normalized generalization matrix (6 trained x 6 eval tasks)."""
    print(f"Loading generalization matrix for {seed}...")

    baselines, baselines_atlantis = load_baselines(seed)
    ftwb1_perf = load_ftwb1_performance(seed)

    matrix = np.zeros((6, 6))

    for ftwb1_num in range(1, 7):
        trained_task = FTWB1_TASKS[ftwb1_num]
        trained_idx = TASKS.index(trained_task)

        for eval_idx, eval_task in enumerate(TASKS):
            value_atlantis = ftwb1_perf[ftwb1_num][eval_task]
            baseline_atlantis = baselines_atlantis[eval_task]
            baseline_standard = baselines[eval_task]
            is_accuracy = eval_task in ACCURACY_TASKS

            normalized = normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy)
            matrix[trained_idx, eval_idx] = normalized

    print(f"  Loaded generalization matrix")
    return matrix

def create_scatter_plot(seed='original'):
    """Create scatter plot of CKA vs generalization for a specific seed."""
    seed_label = "Original (seed 42)" if seed == 'original' else f"Seed {seed}"
    print(f"\n{'='*70}")
    print(f"Creating CKA vs Generalization plot for {seed_label}")
    print(f"{'='*70}")

    # Load data
    cka_matrix = load_cka_matrix_exp4()  # 6x6 from exp4 (layer 5, averaged, no crossing)
    gen_matrix = create_generalization_matrix(seed)  # 6x6 from exp1

    # Collect data points (off-diagonal only)
    x_cka = []
    y_performance = []
    colors = []
    annotations = []

    # Color map for different TRAINING tasks
    task_colors = {
        'distance': '#D86756',
        'trianglearea': '#925CB1',
        'angle': '#5F9FD9',
        'compass': '#E9A947',
        'inside': '#4C5B6C',
        'perimeter': '#58B99D',
    }

    # For each training task X
    for x_idx in range(6):
        x_task = TASKS[x_idx]

        # For each evaluation task Y
        for y_idx in range(6):
            if x_idx == y_idx:
                continue  # Skip diagonal

            y_task = TASKS[y_idx]

            # Get CKA score
            cka = cka_matrix[x_idx, y_idx]

            # Get generalization performance
            perf = gen_matrix[x_idx, y_idx]

            x_cka.append(cka)
            y_performance.append(perf)
            colors.append(task_colors[x_task])

            # Create annotation
            annotation = f"{TASK_ACRONYMS[x_task]}→{TASK_ACRONYMS[y_task]}"
            annotations.append(annotation)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Scatter plot
    scatter = ax.scatter(x_cka, y_performance, c=colors, alpha=0.8, s=200,
                        edgecolors='black', linewidth=1)

    # Add annotations
    for i, (x, y, annotation) in enumerate(zip(x_cka, y_performance, annotations)):
        ax.annotate(annotation, (x, y),
                   fontsize=14, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points', weight='bold')

    # Add trend line
    z = np.polyfit(x_cka, y_performance, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(x_cka), max(x_cka), 100)
    ax.plot(x_trend, p(x_trend), "k:", alpha=0.7, linewidth=3)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(x_cka, y_performance)
    r_squared = correlation ** 2
    slope = z[0]
    intercept = z[1]

    # Title with equation and R^2
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}    $R^2$ = {r_squared:.3f}'
    ax.set_title(equation_text, fontsize=20, weight='bold', pad=20)

    # Labels
    ax.set_xlabel('CKA Score (PT1-X vs PT1-Y, Layer 5, Averaged Across Seeds)',
                 fontsize=18, weight='bold', labelpad=15)
    ax.set_ylabel(f'FTWB1-X Performance on Task Y ({seed_label})\n(0=no improvement, 1=standard level)',
                 fontsize=18, weight='bold', labelpad=15)

    # Style
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='both', which='major', labelsize=20, pad=10, width=2.5, length=8)

    plt.tight_layout()

    # Save
    seed_suffix = "original" if seed == 'original' else f"seed{seed}"
    output_path = OUTPUT_DIR / f'{seed_suffix}_cka_generalization_correlation_layer5.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Print statistics
    print(f"\n{'='*70}")
    print("CORRELATION STATISTICS")
    print(f"{'='*70}")
    print(f"Seed: {seed_label}")
    print(f"Number of data points: {len(x_cka)} (6×6 off-diagonal)")
    print(f"CKA score range: [{min(x_cka):.3f}, {max(x_cka):.3f}]")
    print(f"Performance range: [{min(y_performance):.3f}, {max(y_performance):.3f}]")
    print(f"Pearson correlation: r={correlation:.3f}, p={p_value:.4f}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"Equation: y = {slope:.2f}x + {intercept:.2f}")

    plt.close()

    return correlation, r_squared

def create_aggregated_scatter_plot():
    """Create scatter plot using averaged generalization across all seeds."""
    print(f"\n{'='*70}")
    print(f"Creating AGGREGATED CKA vs Generalization plot")
    print(f"{'='*70}")

    # Load CKA matrix (already averaged across seeds)
    cka_matrix = load_cka_matrix_exp4()

    # Load generalization matrices for all seeds and average them
    gen_matrices = []
    for seed in ['original', 1, 2, 3]:
        gen_matrix = create_generalization_matrix(seed)
        gen_matrices.append(gen_matrix)

    # Average across seeds
    avg_gen_matrix = np.mean(gen_matrices, axis=0)
    print(f"  Averaged generalization across 4 seeds")

    # Collect data points (off-diagonal only)
    x_cka = []
    y_performance = []
    colors = []
    annotations = []

    # Color map for different TRAINING tasks
    task_colors = {
        'distance': '#D86756',
        'trianglearea': '#925CB1',
        'angle': '#5F9FD9',
        'compass': '#E9A947',
        'inside': '#4C5B6C',
        'perimeter': '#58B99D',
    }

    # For each training task X
    for x_idx in range(6):
        x_task = TASKS[x_idx]

        # For each evaluation task Y
        for y_idx in range(6):
            if x_idx == y_idx:
                continue  # Skip diagonal

            y_task = TASKS[y_idx]

            # Get CKA score
            cka = cka_matrix[x_idx, y_idx]

            # Get averaged generalization performance
            perf = avg_gen_matrix[x_idx, y_idx]

            x_cka.append(cka)
            y_performance.append(perf)
            colors.append(task_colors[x_task])

            # Create annotation
            annotation = f"{TASK_ACRONYMS[x_task]}→{TASK_ACRONYMS[y_task]}"
            annotations.append(annotation)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Scatter plot - 1.5x bigger dots
    scatter = ax.scatter(x_cka, y_performance, c=colors, alpha=0.8, s=300,
                        edgecolors='black', linewidth=1)

    # Add annotations - 2x bigger text
    for i, (x, y, annotation) in enumerate(zip(x_cka, y_performance, annotations)):
        ax.annotate(annotation, (x, y),
                   fontsize=28, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points', weight='bold')

    # Add trend line - thicker dotted line
    z = np.polyfit(x_cka, y_performance, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(x_cka), max(x_cka), 100)
    ax.plot(x_trend, p(x_trend), "k:", alpha=0.7, linewidth=6)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(x_cka, y_performance)
    r_squared = correlation ** 2
    slope = z[0]
    intercept = z[1]

    # Title with equation, R^2, and p-value
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}    $R^2$ = {r_squared:.3f}    p = {p_value:.4f}'
    ax.set_title(equation_text, fontsize=20, weight='bold', pad=20)

    # No labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Style
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(5.0)  # 2x thicker
    ax.spines['left'].set_linewidth(5.0)  # 2x thicker
    ax.set_xlim(0.3, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='both', which='major', labelsize=40, pad=10, width=5.0, length=16)  # 2x bigger labels, 2x thicker ticks

    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()

    # Save with text
    output_path = OUTPUT_DIR / 'aggregated_cka_generalization_correlation_layer5.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Save without text version (remove title and annotations)
    ax.set_title('')
    # Remove all text annotations
    for txt in ax.texts:
        txt.set_visible(False)
    output_path_notext = OUTPUT_DIR / 'aggregated_cka_generalization_correlation_layer5_notext.png'
    plt.savefig(output_path_notext, dpi=300, bbox_inches='tight')
    print(f"Saved (no text): {output_path_notext}")

    # Print statistics
    print(f"\n{'='*70}")
    print("CORRELATION STATISTICS (AGGREGATED)")
    print(f"{'='*70}")
    print(f"Number of data points: {len(x_cka)} (6×6 off-diagonal)")
    print(f"CKA score range: [{min(x_cka):.3f}, {max(x_cka):.3f}]")
    print(f"Performance range: [{min(y_performance):.3f}, {max(y_performance):.3f}]")
    print(f"Pearson correlation: r={correlation:.3f}, p={p_value:.4f}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"Equation: y = {slope:.2f}x + {intercept:.2f}")

    plt.close()

    return correlation, r_squared

def main():
    print("="*70)
    print("EXP1: CKA vs GENERALIZATION CORRELATION ANALYSIS")
    print("="*70)
    print("\nUsing:")
    print("  - CKA: exp4 layer 5 averaged across seeds (no crossing)")
    print("  - Generalization: exp1 FTWB1 performance (no crossing)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create plots for original and each seed
    results = {}
    for seed in ['original', 1, 2, 3]:
        correlation, r_squared = create_scatter_plot(seed=seed)
        results[seed] = {'r': correlation, 'r2': r_squared}

    # Create aggregated plot
    agg_r, agg_r2 = create_aggregated_scatter_plot()
    results['aggregated'] = {'r': agg_r, 'r2': agg_r2}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS SEEDS")
    print(f"{'='*70}")
    for seed, res in results.items():
        if seed == 'aggregated':
            seed_label = "Aggregated (avg)"
        elif seed == 'original':
            seed_label = "Original"
        else:
            seed_label = f"Seed {seed}"
        print(f"{seed_label:20s}: r={res['r']:.3f}, R²={res['r2']:.3f}")

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
