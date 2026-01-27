#!/usr/bin/env python3
"""
Create FTWB1 evaluation heatmaps for revision/exp1.
Generates 4 plots (original + seeds 1,2,3) showing 7x7 heatmap of normalized performance.

Each FTWB1 model is trained on a single task and evaluated on all 7 tasks.
Rows = trained task (what the model was fine-tuned on)
Cols = evaluated task (what task we're testing on)

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
EXP_BASE = Path("/data/experiments/revision/exp1")
OUTPUT_DIR = Path("/data/experiments/revision/exp1/plots")

# All tasks in order
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]

# Accuracy-based tasks (higher is better)
ACCURACY_TASKS = ["crossing", "inside", "compass"]

# FTWB1 task mapping
FTWB1_TASKS = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

def load_baselines(seed):
    """Load PT1 baseline performance for a specific seed.

    Args:
        seed: int (1,2,3) for revision seeds, or 'original' for original pt1
    """
    baselines = {}
    baselines_atlantis = {}

    if seed == 'original':
        base_exp = Path("/data/experiments/pt1")
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

def load_ftwb1_performance(seed):
    """Load FTWB1 performance for all 7 single-task models.

    Args:
        seed: int (1,2,3) for revision seeds, or 'original' for original pt1

    Returns:
        dict: {ftwb1_num: {task: performance}}
    """
    performance_dict = {}

    for ftwb1_num in range(1, 8):
        if seed == 'original':
            exp_name = f"pt1_ftwb1-{ftwb1_num}"
            exp_path = Path("/data/experiments") / exp_name
        else:
            exp_name = f"pt1_seed{seed}_ftwb1-{ftwb1_num}"
            exp_path = EXP_BASE / exp_name

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
    """Create normalized performance matrix (7 trained tasks × 7 eval tasks)."""
    n_tasks = len(TASKS)
    matrix = np.zeros((n_tasks, n_tasks))

    for ftwb1_num in range(1, 8):
        trained_task = FTWB1_TASKS[ftwb1_num]
        trained_idx = TASKS.index(trained_task)

        for eval_idx, eval_task in enumerate(TASKS):
            value_atlantis = performance_dict[ftwb1_num][eval_task]
            baseline_atlantis = baselines_atlantis[eval_task]
            baseline_standard = baselines[eval_task]
            is_accuracy = eval_task in ACCURACY_TASKS

            normalized = normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy)
            matrix[trained_idx, eval_idx] = normalized

    return matrix

def plot_ftwb1_heatmap(seed):
    """Create heatmap for a specific seed.

    Args:
        seed: int (1,2,3) for revision seeds, or 'original' for original pt1
    """
    seed_label = "Original (seed 42)" if seed == 'original' else f"Seed {seed}"
    print(f"\nProcessing {seed_label}...")

    # Load baselines and performance
    baselines, baselines_atlantis = load_baselines(seed)
    ftwb1_perf = load_ftwb1_performance(seed)

    # Create performance matrix
    matrix = create_performance_matrix(ftwb1_perf, baselines, baselines_atlantis)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create heatmap - no cell lines, bigger annotation font, cbar aspect ratio
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='RdYlGn',
                vmin=0.0, vmax=1.0, square=True,
                cbar_kws={"shrink": 1.0, "aspect": 20},  # aspect=20 is the magic number
                xticklabels=[f"{i+1}" for i in range(7)],
                yticklabels=[f"{i+1}" for i in range(7)],
                ax=ax, linewidths=0, linecolor='none',
                annot_kws={"fontsize": 13.2, "fontweight": "bold"})  # 11 * 1.2 = 13.2

    # Remove all tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, top=False)

    # Add 'T' to mark training tasks (on diagonal)
    for i in range(7):
        ax.text(i + 0.15, i + 0.15, 'T', fontsize=10, fontweight='bold',
               color='black', ha='center', va='center')

    # Save figure
    filename = "original_ftwb1_evaluation_heatmap.png" if seed == 'original' else f"seed{seed}_ftwb1_evaluation_heatmap.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Print summary statistics
    print(f"\n  {seed_label} Statistics:")

    # Diagonal (trained task performance)
    diagonal_values = np.diag(matrix)
    print(f"    Trained task (diagonal): {np.mean(diagonal_values):.3f} ± {np.std(diagonal_values):.3f}")

    # Off-diagonal (transfer performance)
    off_diagonal_mask = ~np.eye(7, dtype=bool)
    off_diagonal_values = matrix[off_diagonal_mask]
    print(f"    Transfer (off-diagonal): {np.mean(off_diagonal_values):.3f} ± {np.std(off_diagonal_values):.3f}")

    plt.close()

def plot_aggregated_ftwb1_heatmap():
    """Create aggregated heatmap averaging across all seeds."""
    print(f"\nProcessing Aggregated (average across all seeds)...")

    # Collect matrices from all seeds
    matrices = []
    for seed in ['original', 1, 2, 3]:
        baselines, baselines_atlantis = load_baselines(seed)
        ftwb1_perf = load_ftwb1_performance(seed)
        matrix = create_performance_matrix(ftwb1_perf, baselines, baselines_atlantis)
        matrices.append(matrix)

    # Average across seeds
    avg_matrix = np.mean(matrices, axis=0)
    print(f"  Averaged across 4 seeds")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create heatmap
    sns.heatmap(avg_matrix, annot=True, fmt=".2f", cmap='RdYlGn',
                vmin=0.0, vmax=1.0, square=True,
                cbar_kws={"shrink": 1.0, "aspect": 20},
                xticklabels=[f"{i+1}" for i in range(7)],
                yticklabels=[f"{i+1}" for i in range(7)],
                ax=ax, linewidths=0, linecolor='none',
                annot_kws={"fontsize": 13.2, "fontweight": "bold"})

    # Remove all tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, top=False)

    # Add 'T' to mark training tasks (on diagonal)
    for i in range(7):
        ax.text(i + 0.15, i + 0.15, 'T', fontsize=10, fontweight='bold',
               color='black', ha='center', va='center')

    # Save figure
    output_path = OUTPUT_DIR / "aggregated_ftwb1_evaluation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Print summary statistics
    print(f"\n  Aggregated Statistics:")
    diagonal_values = np.diag(avg_matrix)
    print(f"    Trained task (diagonal): {np.mean(diagonal_values):.3f} ± {np.std(diagonal_values):.3f}")
    off_diagonal_mask = ~np.eye(7, dtype=bool)
    off_diagonal_values = avg_matrix[off_diagonal_mask]
    print(f"    Transfer (off-diagonal): {np.mean(off_diagonal_values):.3f} ± {np.std(off_diagonal_values):.3f}")

    plt.close()

def main():
    print("="*60)
    print("Creating FTWB1 Evaluation Heatmaps for Revision Exp1")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create plots for original and each seed
    for seed in ['original', 1, 2, 3]:
        plot_ftwb1_heatmap(seed)

    # Create aggregated plot
    plot_aggregated_ftwb1_heatmap()

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - original_ftwb1_evaluation_heatmap.png")
    print("  - seed1_ftwb1_evaluation_heatmap.png")
    print("  - seed2_ftwb1_evaluation_heatmap.png")
    print("  - seed3_ftwb1_evaluation_heatmap.png")
    print("  - aggregated_ftwb1_evaluation_heatmap.png")

if __name__ == "__main__":
    main()
