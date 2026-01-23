#!/usr/bin/env python3
"""
Create histogram comparing delta (actual - predicted) for FTWB2 experiments
that involve the distance task versus those that don't.

Delta = Actual FTWB2 performance - Predicted from FTWB1 specialists
Aggregated across all 4 seeds (original, seed1, seed2, seed3).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
        base_exp = Path("/n/home12/cfpark00/datadir/WM_1/data/experiments/pt1")
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
            exp_path = Path("/n/home12/cfpark00/datadir/WM_1/data/experiments") / exp_name
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
    """Create normalized actual performance matrix (21 experiments × 7 tasks)."""
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
    """Create prediction matrix based on max performance from single-task FTWB1 models."""
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

def collect_deltas_by_distance():
    """Collect all delta values across all seeds, separated by distance involvement."""
    deltas_with_distance = []
    deltas_without_distance = []

    for seed in ['original', 1, 2, 3]:
        baselines, baselines_atlantis = load_baselines(seed)
        ftwb1_perf = load_ftwb1_performance(seed)
        ftwb2_perf = load_ftwb2_performance(seed)

        actual_matrix = create_actual_matrix(ftwb2_perf, baselines, baselines_atlantis)
        predicted_matrix = create_prediction_matrix(ftwb1_perf, baselines, baselines_atlantis)
        diff_matrix = actual_matrix - predicted_matrix

        # Iterate through all experiments and tasks
        for exp_num in range(1, 22):
            exp_idx = exp_num - 1
            trained_tasks = TRAINING_DATA_2TASK[exp_num]
            involves_distance = "distance" in trained_tasks

            for task_idx, task in enumerate(TASKS):
                delta = diff_matrix[exp_idx, task_idx]

                if involves_distance:
                    deltas_with_distance.append(delta)
                else:
                    deltas_without_distance.append(delta)

    return deltas_with_distance, deltas_without_distance

def main():
    print("="*60)
    print("Creating FTWB2 Distance vs Non-Distance Delta Histogram")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect deltas
    print("\nCollecting delta values across all seeds...")
    deltas_with, deltas_without = collect_deltas_by_distance()

    print(f"  With distance: {len(deltas_with)} values")
    print(f"  Without distance: {len(deltas_without)} values")
    print(f"\n  With distance mean: {np.mean(deltas_with):.3f} ± {np.std(deltas_with):.3f}")
    print(f"  Without distance mean: {np.mean(deltas_without):.3f} ± {np.std(deltas_without):.3f}")

    # Create histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.linspace(-0.5, 0.5, 24)

    ax.hist(deltas_without, bins=bins, alpha=0.6, label='Without Distance', color='blue', edgecolor='black')
    ax.hist(deltas_with, bins=bins, alpha=0.6, label='With Distance', color='red', edgecolor='black')

    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    # Remove labels, legend, grid
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines thicker
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    # Make tick labels bigger and bold, push away from plot
    ax.tick_params(axis='both', which='major', labelsize=20, width=3, pad=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Save figure
    output_path = OUTPUT_DIR / "ftwb2_distance_delta_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
