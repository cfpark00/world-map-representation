#!/usr/bin/env python3
"""
Create FTWB2 evaluation heatmaps for revision/exp1.
Generates 3 plots (one per seed) showing 21x7 heatmap of normalized performance.

Normalized performance metric:
- 0.0 = no improvement over atlantis baseline (untrained performance)
- 1.0 = reaches standard baseline (fully trained performance)
- Values can exceed 1.0 if performance is better than standard baseline

For error metrics (distance, trianglearea, angle, perimeter):
  - Lower is better
  - Uses log-ratio: log(baseline_atlantis / value) / log(baseline_atlantis / baseline_standard)

For accuracy metrics (crossing, inside, compass):
  - Higher is better
  - Uses linear: (value - baseline_atlantis) / (baseline_standard - baseline_atlantis)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import math

# Base paths
EXP_BASE = Path("/n/home12/cfpark00/datadir/WM_1/data/experiments/revision/exp1")
OUTPUT_DIR = Path("/n/home12/cfpark00/datadir/WM_1/data/experiments/revision/exp1/plots")

# All tasks in order
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]

# Accuracy-based tasks (higher is better)
ACCURACY_TASKS = ["crossing", "inside", "compass"]

# Training data for FTWB2 experiments (2 tasks each)
TRAINING_DATA_2TASK = {
    1: ["distance", "trianglearea"],
    2: ["angle", "compass"],
    3: ["inside", "perimeter"],
    4: ["crossing", "distance"],
    5: ["trianglearea", "angle"],
    6: ["compass", "inside"],
    7: ["perimeter", "crossing"],
    8: ["angle", "distance"],
    9: ["compass", "trianglearea"],
    10: ["angle", "inside"],
    11: ["compass", "perimeter"],
    12: ["crossing", "inside"],
    13: ["distance", "perimeter"],
    14: ["crossing", "trianglearea"],
    15: ["compass", "distance"],
    16: ["inside", "trianglearea"],
    17: ["angle", "perimeter"],
    18: ["compass", "crossing"],
    19: ["distance", "inside"],
    20: ["perimeter", "trianglearea"],
    21: ["angle", "crossing"],
}

def load_baselines(seed):
    """Load PT1 baseline performance for a specific seed.

    Args:
        seed: int (1,2,3) for revision seeds, or 'original' for original pt1
    """
    baselines = {}
    baselines_atlantis = {}

    if seed == 'original':
        base_exp = Path("/n/home12/cfpark00/datadir/WM_1/data/experiments/pt1")
    else:
        base_exp = EXP_BASE / f"pt1_seed{seed}"

    for task in TASKS:
        # Standard task baseline
        json_path = base_exp / "evals" / task / "eval_data" / "evaluation_results.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Get last checkpoint
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

def load_ftwb2_performance(seed):
    """Load FTWB2 performance for all 21 experiments.

    Args:
        seed: int (1,2,3) for revision seeds, or 'original' for original pt1
    """
    performance_dict = {}

    for exp_num in range(1, 22):
        if seed == 'original':
            exp_name = f"pt1_ftwb2-{exp_num}"
            exp_path = Path("/n/home12/cfpark00/datadir/WM_1/data/experiments") / exp_name
        else:
            exp_name = f"pt1_seed{seed}_ftwb2-{exp_num}"
            exp_path = EXP_BASE / exp_name

        performance_dict[exp_num] = {}

        for task in TASKS:
            # Load Atlantis task performance
            atlantis_path = exp_path / "evals" / f"atlantis_{task}" / "eval_data" / "evaluation_results.json"
            with open(atlantis_path, 'r') as f:
                data = json.load(f)
                checkpoints = sorted(data.keys(), key=lambda x: int(x.split('-')[1]))
                last_checkpoint = checkpoints[-1]
                performance_dict[exp_num][task] = data[last_checkpoint][f'eval_{task}_metric_mean']

    return performance_dict

def normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy=False):
    """
    Normalize metrics to 0-1 scale where:
    - 0 = atlantis baseline (no improvement)
    - 1 = standard baseline (fully trained)

    Uses log-ratio for error metrics, linear for accuracy.
    """
    if is_accuracy:
        # Higher is better (accuracy)
        if baseline_standard <= baseline_atlantis:
            return 0.0
        else:
            normalized = (value_atlantis - baseline_atlantis) / (baseline_standard - baseline_atlantis)
    else:
        # Lower is better (error metric)
        if baseline_standard >= baseline_atlantis or baseline_atlantis <= 0 or value_atlantis <= 0:
            # Handle edge cases
            if value_atlantis >= baseline_atlantis:
                return 0.0
            elif value_atlantis <= baseline_standard:
                return 1.0
            else:
                normalized = (baseline_atlantis - value_atlantis) / (baseline_atlantis - baseline_standard)
        else:
            # Normal case: use log-ratio
            numerator = math.log(baseline_atlantis / value_atlantis)
            denominator = math.log(baseline_atlantis / baseline_standard)

            if denominator == 0:
                return 0.0
            else:
                normalized = numerator / denominator

    # Clip to reasonable range
    return max(0.0, min(1.5, normalized))

def create_performance_matrix(performance_dict, baselines, baselines_atlantis):
    """Create normalized performance matrix (21 experiments × 7 tasks).

    TRANSPOSED: rows = experiments (trained on), cols = tasks (evaluated on)
    """
    n_tasks = len(TASKS)
    n_exps = len(performance_dict)
    matrix = np.zeros((n_exps, n_tasks))

    for exp_num in range(1, n_exps + 1):
        for task_idx, task in enumerate(TASKS):
            value_atlantis = performance_dict[exp_num][task]
            baseline_atlantis = baselines_atlantis[task]
            baseline_standard = baselines[task]
            is_accuracy = task in ACCURACY_TASKS

            normalized = normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy)
            matrix[exp_num - 1, task_idx] = normalized

    return matrix

def plot_ftwb2_heatmap(seed):
    """Create heatmap for a specific seed.

    Args:
        seed: int (1,2,3) for revision seeds, or 'original' for original pt1
    """
    seed_label = "Original (seed 42)" if seed == 'original' else f"Seed {seed}"
    print(f"\nProcessing {seed_label}...")

    # Load baselines and performance
    baselines, baselines_atlantis = load_baselines(seed)
    ftwb2_perf = load_ftwb2_performance(seed)

    # Create performance matrix
    matrix = create_performance_matrix(ftwb2_perf, baselines, baselines_atlantis)

    # Create figure - adjusted for transposed matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))

    # Create heatmap - no cell lines, bigger annotation font, NO colorbar
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='RdYlGn',
                vmin=0.0, vmax=1.0, square=True,
                cbar=False,  # Remove colorbar
                xticklabels=[f"{i+1}" for i in range(7)],
                yticklabels=[f"{i+1}" for i in range(21)],
                ax=ax, linewidths=0, linecolor='none',
                annot_kws={"fontsize": 13.2, "fontweight": "bold"})  # 11 * 1.2 = 13.2

    # Remove all tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, top=False)

    # Add 'T' to mark training tasks (top-left corner)
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task in trained_tasks:
            task_idx = TASKS.index(task)
            ax.text(task_idx + 0.15, exp_idx + 0.15, 'T', fontsize=10, fontweight='bold',
                   color='black', ha='center', va='center')

    # Save figure
    filename = "original_ftwb2_evaluation_heatmap.png" if seed == 'original' else f"seed{seed}_ftwb2_evaluation_heatmap.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Print summary statistics
    print(f"\n  {seed_label} Statistics:")

    # Trained task performance (after transpose: matrix[exp_idx, task_idx])
    trained_values = []
    for exp_idx in range(21):
        for task in TRAINING_DATA_2TASK[exp_idx + 1]:
            task_idx = TASKS.index(task)
            trained_values.append(matrix[exp_idx, task_idx])
    print(f"    Trained tasks: {np.mean(trained_values):.3f} ± {np.std(trained_values):.3f}")

    # Transfer performance
    transfer_values = []
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task_idx, task in enumerate(TASKS):
            if task not in trained_tasks:
                transfer_values.append(matrix[exp_idx, task_idx])
    print(f"    Transfer tasks: {np.mean(transfer_values):.3f} ± {np.std(transfer_values):.3f}")

    plt.close()

def plot_aggregated_ftwb2_heatmap():
    """Create aggregated heatmap averaging across all seeds."""
    print(f"\nProcessing Aggregated (average across all seeds)...")

    # Collect matrices from all seeds
    matrices = []
    for seed in ['original', 1, 2, 3]:
        baselines, baselines_atlantis = load_baselines(seed)
        ftwb2_perf = load_ftwb2_performance(seed)
        matrix = create_performance_matrix(ftwb2_perf, baselines, baselines_atlantis)
        matrices.append(matrix)

    # Average across seeds
    avg_matrix = np.mean(matrices, axis=0)
    print(f"  Averaged across 4 seeds")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))

    # Create heatmap - NO colorbar
    sns.heatmap(avg_matrix, annot=True, fmt=".2f", cmap='RdYlGn',
                vmin=0.0, vmax=1.0, square=True,
                cbar=False,  # Remove colorbar
                xticklabels=[f"{i+1}" for i in range(7)],
                yticklabels=[f"{i+1}" for i in range(21)],
                ax=ax, linewidths=0, linecolor='none',
                annot_kws={"fontsize": 13.2, "fontweight": "bold"})

    # Remove all tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, top=False)

    # Add 'T' to mark training tasks
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task in trained_tasks:
            task_idx = TASKS.index(task)
            ax.text(task_idx + 0.15, exp_idx + 0.15, 'T', fontsize=10, fontweight='bold',
                   color='black', ha='center', va='center')

    # Save figure
    output_path = OUTPUT_DIR / "aggregated_ftwb2_evaluation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Print summary statistics
    print(f"\n  Aggregated Statistics:")
    trained_values = []
    for exp_idx in range(21):
        for task in TRAINING_DATA_2TASK[exp_idx + 1]:
            task_idx = TASKS.index(task)
            trained_values.append(avg_matrix[exp_idx, task_idx])
    print(f"    Trained tasks: {np.mean(trained_values):.3f} ± {np.std(trained_values):.3f}")

    transfer_values = []
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task_idx, task in enumerate(TASKS):
            if task not in trained_tasks:
                transfer_values.append(avg_matrix[exp_idx, task_idx])
    print(f"    Transfer tasks: {np.mean(transfer_values):.3f} ± {np.std(transfer_values):.3f}")

    plt.close()

def main():
    print("="*60)
    print("Creating FTWB2 Evaluation Heatmaps for Revision Exp1")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create plots for original and each seed
    for seed in ['original', 1, 2, 3]:
        plot_ftwb2_heatmap(seed)

    # Create aggregated plot
    plot_aggregated_ftwb2_heatmap()

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - original_ftwb2_evaluation_heatmap.png")
    print("  - seed1_ftwb2_evaluation_heatmap.png")
    print("  - seed2_ftwb2_evaluation_heatmap.png")
    print("  - seed3_ftwb2_evaluation_heatmap.png")
    print("  - aggregated_ftwb2_evaluation_heatmap.png")

if __name__ == "__main__":
    main()
