#!/usr/bin/env python3
"""
Create FTWB2 vs FTWB1 prediction plots for revision/exp1.
Generates 4 plots (original + seeds 1,2,3), each with two panels:
- Top: Actual FTWB2 performance heatmap
- Bottom: Difference from predicted (actual - predicted) using RdBu colormap

The prediction is based on maximum performance from single-task FTWB1 models
from the SAME seed (e.g., seed1 FTWB2 predicted from seed1 FTWB1).
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
    """Load PT1 baseline performance for a specific seed."""
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
    """Load FTWB1 performance for all 7 single-task models."""
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

def load_ftwb2_performance(seed):
    """Load FTWB2 performance for all 21 experiments."""
    performance_dict = {}

    for exp_num in range(1, 22):
        if seed == 'original':
            exp_name = f"pt1_ftwb2-{exp_num}"
            exp_path = Path("/data/experiments") / exp_name
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
    """Normalize metrics to 0-1 scale using log-ratio for errors, linear for accuracy."""
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

def create_actual_matrix(ftwb2_perf, baselines, baselines_atlantis):
    """Create normalized actual performance matrix (21 experiments × 7 tasks).

    TRANSPOSED: rows = experiments (trained on), cols = tasks (evaluated on)
    """
    n_tasks = len(TASKS)
    n_exps = 21
    matrix = np.zeros((n_exps, n_tasks))

    for exp_num in range(1, n_exps + 1):
        for task_idx, task in enumerate(TASKS):
            value_atlantis = ftwb2_perf[exp_num][task]
            baseline_atlantis = baselines_atlantis[task]
            baseline_standard = baselines[task]
            is_accuracy = task in ACCURACY_TASKS

            normalized = normalize_metric(value_atlantis, baseline_atlantis, baseline_standard, is_accuracy)
            matrix[exp_num - 1, task_idx] = normalized

    return matrix

def create_prediction_matrix(ftwb1_perf, baselines, baselines_atlantis):
    """Create prediction matrix based on max performance from single-task FTWB1 models.

    TRANSPOSED: rows = experiments (trained on), cols = tasks (evaluated on)
    """
    n_tasks = len(TASKS)
    n_exps = 21
    matrix = np.zeros((n_exps, n_tasks))

    task_to_idx = {task: idx for idx, task in enumerate(TASKS)}

    for exp_num in range(1, n_exps + 1):
        trained_tasks = TRAINING_DATA_2TASK[exp_num]

        for task_idx, eval_task in enumerate(TASKS):
            # Find best performance from FTWB1 models trained on any of the trained tasks
            best_value = None
            for trained_task in trained_tasks:
                # Get the FTWB1 model number for this trained task
                ftwb1_num = task_to_idx[trained_task] + 1
                value = ftwb1_perf[ftwb1_num][eval_task]

                # Keep the best value
                if best_value is None:
                    best_value = value
                elif eval_task in ACCURACY_TASKS and value > best_value:
                    best_value = value
                elif eval_task not in ACCURACY_TASKS and value < best_value:
                    best_value = value

            # Normalize the best value
            baseline_atlantis = baselines_atlantis[eval_task]
            baseline_standard = baselines[eval_task]
            is_accuracy = eval_task in ACCURACY_TASKS

            normalized = normalize_metric(best_value, baseline_atlantis, baseline_standard, is_accuracy)
            matrix[exp_num - 1, task_idx] = normalized

    return matrix

def plot_ftwb2_vs_ftwb1(seed):
    """Create difference plot: actual - predicted from FTWB1."""
    seed_label = "Original (seed 42)" if seed == 'original' else f"Seed {seed}"
    print(f"\nProcessing {seed_label}...")

    # Load baselines
    baselines, baselines_atlantis = load_baselines(seed)

    # Load FTWB1 and FTWB2 performance
    ftwb1_perf = load_ftwb1_performance(seed)
    ftwb2_perf = load_ftwb2_performance(seed)

    # Create matrices
    actual_matrix = create_actual_matrix(ftwb2_perf, baselines, baselines_atlantis)
    predicted_matrix = create_prediction_matrix(ftwb1_perf, baselines, baselines_atlantis)
    diff_matrix = actual_matrix - predicted_matrix

    # Create figure - adjusted for transposed matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))

    # Difference from predicted (Actual - Predicted) - NO colorbar
    sns.heatmap(diff_matrix, annot=True, fmt=".2f", cmap='RdBu',
                center=0, vmin=-0.5, vmax=0.5, square=True,
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

    # Add 'T' to mark training tasks
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task in trained_tasks:
            task_idx = TASKS.index(task)
            ax.text(task_idx + 0.15, exp_idx + 0.15, 'T', fontsize=10, fontweight='bold',
                   color='black', ha='center', va='center')

    # Save figure
    filename = "original_ftwb2_vs_ftwb1.png" if seed == 'original' else f"seed{seed}_ftwb2_vs_ftwb1.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Print summary statistics
    print(f"\n  {seed_label} Statistics:")

    # Trained task performance (after transpose: matrix[exp_idx, task_idx])
    trained_actual = []
    trained_diff = []
    for exp_idx in range(21):
        for task in TRAINING_DATA_2TASK[exp_idx + 1]:
            task_idx = TASKS.index(task)
            trained_actual.append(actual_matrix[exp_idx, task_idx])
            trained_diff.append(diff_matrix[exp_idx, task_idx])
    print(f"    Trained tasks actual: {np.mean(trained_actual):.3f} ± {np.std(trained_actual):.3f}")
    print(f"    Trained tasks diff: {np.mean(trained_diff):.3f} ± {np.std(trained_diff):.3f}")

    # Transfer performance
    transfer_actual = []
    transfer_diff = []
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task_idx, task in enumerate(TASKS):
            if task not in trained_tasks:
                transfer_actual.append(actual_matrix[exp_idx, task_idx])
                transfer_diff.append(diff_matrix[exp_idx, task_idx])
    print(f"    Transfer tasks actual: {np.mean(transfer_actual):.3f} ± {np.std(transfer_actual):.3f}")
    print(f"    Transfer tasks diff: {np.mean(transfer_diff):.3f} ± {np.std(transfer_diff):.3f}")

    # Overall difference statistics
    print(f"\n    Overall difference mean: {np.mean(diff_matrix):.3f}")
    print(f"    Overall difference std: {np.std(diff_matrix):.3f}")

    plt.close()

def plot_aggregated_ftwb2_vs_ftwb1():
    """Create aggregated difference plot averaging across all seeds."""
    print(f"\nProcessing Aggregated (average across all seeds)...")

    # Collect matrices from all seeds
    actual_matrices = []
    diff_matrices = []

    for seed in ['original', 1, 2, 3]:
        baselines, baselines_atlantis = load_baselines(seed)
        ftwb1_perf = load_ftwb1_performance(seed)
        ftwb2_perf = load_ftwb2_performance(seed)
        actual_matrix = create_actual_matrix(ftwb2_perf, baselines, baselines_atlantis)
        predicted_matrix = create_prediction_matrix(ftwb1_perf, baselines, baselines_atlantis)
        diff_matrix = actual_matrix - predicted_matrix

        actual_matrices.append(actual_matrix)
        diff_matrices.append(diff_matrix)

    # Average across seeds
    avg_actual = np.mean(actual_matrices, axis=0)
    avg_diff = np.mean(diff_matrices, axis=0)
    print(f"  Averaged across 4 seeds")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 16))

    # Difference heatmap
    sns.heatmap(avg_diff, annot=True, fmt=".2f", cmap='RdBu',
                center=0, vmin=-0.5, vmax=0.5, square=True,
                cbar=False,
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

    # Add 'T' markers
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task in trained_tasks:
            task_idx = TASKS.index(task)
            ax.text(task_idx + 0.15, exp_idx + 0.15, 'T', fontsize=10, fontweight='bold',
                   color='black', ha='center', va='center')

    # Save figure
    output_path = OUTPUT_DIR / "aggregated_ftwb2_vs_ftwb1.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Print summary statistics
    print(f"\n  Aggregated Statistics:")
    trained_actual = []
    trained_diff = []
    for exp_idx in range(21):
        for task in TRAINING_DATA_2TASK[exp_idx + 1]:
            task_idx = TASKS.index(task)
            trained_actual.append(avg_actual[exp_idx, task_idx])
            trained_diff.append(avg_diff[exp_idx, task_idx])
    print(f"    Trained tasks actual: {np.mean(trained_actual):.3f} ± {np.std(trained_actual):.3f}")
    print(f"    Trained tasks diff: {np.mean(trained_diff):.3f} ± {np.std(trained_diff):.3f}")

    transfer_actual = []
    transfer_diff = []
    for exp_idx in range(21):
        trained_tasks = TRAINING_DATA_2TASK[exp_idx + 1]
        for task_idx, task in enumerate(TASKS):
            if task not in trained_tasks:
                transfer_actual.append(avg_actual[exp_idx, task_idx])
                transfer_diff.append(avg_diff[exp_idx, task_idx])
    print(f"    Transfer tasks actual: {np.mean(transfer_actual):.3f} ± {np.std(transfer_actual):.3f}")
    print(f"    Transfer tasks diff: {np.mean(transfer_diff):.3f} ± {np.std(transfer_diff):.3f}")

    print(f"\n    Overall difference mean: {np.mean(avg_diff):.3f}")
    print(f"    Overall difference std: {np.std(avg_diff):.3f}")

    plt.close()

def main():
    print("="*60)
    print("Creating FTWB2 vs FTWB1 Plots for Revision Exp1")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create plots for original and each seed
    for seed in ['original', 1, 2, 3]:
        plot_ftwb2_vs_ftwb1(seed)

    # Create aggregated plot
    plot_aggregated_ftwb2_vs_ftwb1()

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - original_ftwb2_vs_ftwb1.png")
    print("  - seed1_ftwb2_vs_ftwb1.png")
    print("  - seed2_ftwb2_vs_ftwb1.png")
    print("  - seed3_ftwb2_vs_ftwb1.png")
    print("  - aggregated_ftwb2_vs_ftwb1.png")

if __name__ == "__main__":
    main()
