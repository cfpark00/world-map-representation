#!/usr/bin/env python3
"""
Create histogram comparing probe generalization distance errors:
- Atlantis (OOD) vs Baseline (held-out non-Atlantis)
across all 21 ftwb2 models, pooling ALL individual city samples.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Base paths
EXP_BASE = Path("/data/experiments/revision/exp1")
EXP5_BASE = Path("/data/experiments/revision/exp5/pt1_with_atlantis")
OUTPUT_DIR = Path("/data/experiments/revision/exp1/plots")

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

def has_distance_task(ftwb2_id):
    """Check if ftwb2 model was trained with distance task."""
    return "distance" in TRAINING_DATA_2TASK[ftwb2_id]


def load_all_predictions(ftwb2_id, seed='seed1'):
    """Load all individual predictions for a specific ftwb2 model and seed."""
    if seed == 'original':
        base_path = Path("/data/experiments") / f"pt1_ftwb2-{ftwb2_id}" / "probe_generalization" / "atlantis"
    else:
        base_path = EXP_BASE / f"pt1_{seed}_ftwb2-{ftwb2_id}" / "probe_generalization" / "atlantis"

    test_path = base_path / "test_predictions.csv"
    baseline_path = base_path / "baseline_predictions.csv"

    if not test_path.exists():
        return None, None

    test_df = pd.read_csv(test_path)

    # Baseline predictions might not exist for older runs
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
    else:
        baseline_df = None

    return test_df, baseline_df


def main():
    print("=" * 60)
    print("Creating Probe Generalization Histogram (Pooled Samples)")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect ALL individual errors from all ftwb2 models across ALL seeds
    all_atlantis_errors = []
    all_baseline_errors = []
    atlantis_with_distance = []  # from ftwb2 models trained WITH distance task
    atlantis_without_distance = []  # from ftwb2 models trained WITHOUT distance task
    missing_models = []
    models_without_baseline = []

    seeds = ['original', 'seed1', 'seed2', 'seed3']

    for seed in seeds:
        for ftwb2_id in range(1, 22):
            test_df, baseline_df = load_all_predictions(ftwb2_id, seed=seed)

            if test_df is None:
                missing_models.append(f"{seed}_ftwb2-{ftwb2_id}")
                continue

            # Pool all individual Atlantis errors
            errors = test_df['dist_error'].tolist()
            all_atlantis_errors.extend(errors)

            # Split by whether model was trained with distance task
            if has_distance_task(ftwb2_id):
                atlantis_with_distance.extend(errors)
            else:
                atlantis_without_distance.extend(errors)

            # Pool all individual baseline errors
            if baseline_df is not None:
                all_baseline_errors.extend(baseline_df['dist_error'].tolist())
            else:
                models_without_baseline.append(f"{seed}_ftwb2-{ftwb2_id}")

    print(f"\nLoaded data from {84 - len(missing_models)} models (4 seeds × 21 ftwb2)")
    if missing_models:
        print(f"Missing models: {missing_models}")
    if models_without_baseline:
        print(f"Models without baseline predictions: {models_without_baseline}")

    all_atlantis_errors = np.array(all_atlantis_errors)
    all_baseline_errors = np.array(all_baseline_errors)
    atlantis_with_distance = np.array(atlantis_with_distance)
    atlantis_without_distance = np.array(atlantis_without_distance)

    print(f"\nAtlantis samples: {len(all_atlantis_errors)}")
    print(f"  - With distance task: {len(atlantis_with_distance)}")
    print(f"  - Without distance task: {len(atlantis_without_distance)}")
    print(f"Baseline samples: {len(all_baseline_errors)}")
    print(f"\nAtlantis (OOD): {np.mean(all_atlantis_errors):.1f} ± {np.std(all_atlantis_errors):.1f}")
    print(f"  - With distance: {np.mean(atlantis_with_distance):.1f} ± {np.std(atlantis_with_distance):.1f}")
    print(f"  - Without distance: {np.mean(atlantis_without_distance):.1f} ± {np.std(atlantis_without_distance):.1f}")
    print(f"Baseline (held-out): {np.mean(all_baseline_errors):.1f} ± {np.std(all_baseline_errors):.1f}")

    # Load exp5 mean (PT1 trained with Atlantis)
    exp5_test_path = EXP5_BASE / "probe_generalization" / "atlantis" / "test_predictions.csv"
    exp5_mean = None
    if exp5_test_path.exists():
        exp5_df = pd.read_csv(exp5_test_path)
        exp5_mean = exp5_df['dist_error'].mean()
        print(f"\nExp5 (PT1 with Atlantis): {exp5_mean:.1f} (n={len(exp5_df)})")
    else:
        print(f"\nWARNING: Exp5 data not found at {exp5_test_path}")

    # Create histogram with 3 groups: baseline, atlantis with distance, atlantis without distance
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Use log-spaced bins for log x-axis
    all_errors = np.concatenate([all_atlantis_errors, all_baseline_errors])
    bins = np.logspace(np.log10(max(1, all_errors.min())), np.log10(all_errors.max() * 1.1), 40)

    ax.hist(all_baseline_errors, bins=bins, alpha=0.6, label='Baseline (non-Atlantis)',
            color='orange', edgecolor='black')
    ax.hist(atlantis_without_distance, bins=bins, alpha=0.6, label='Atlantis (no distance task)',
            color='blue', edgecolor='black')
    ax.hist(atlantis_with_distance, bins=bins, alpha=0.6, label='Atlantis (with distance task)',
            color='red', edgecolor='black')

    # Add mean lines
    ax.axvline(np.mean(all_baseline_errors), color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(np.mean(atlantis_without_distance), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(np.mean(atlantis_with_distance), color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Add exp5 mean line if available
    if exp5_mean is not None:
        ax.axvline(exp5_mean, color='#38C50D', linestyle='-', linewidth=4, alpha=0.9)

    # Set log scale for x-axis only
    ax.set_xscale('log')
    ax.set_xlim(left=2)

    # Style to match existing plots
    ax.set_xlabel('')
    ax.set_ylabel('')


    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines thicker
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    # Make tick labels bigger and bold
    ax.tick_params(axis='both', which='major', labelsize=20, width=3, pad=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Save figure
    output_path = OUTPUT_DIR / "probe_generalization_histogram_with_exp5.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()

    # Also save summary statistics
    summary = {
        'n_models': 21 - len(missing_models),
        'missing_models': missing_models,
        'models_without_baseline': models_without_baseline,
        'atlantis': {
            'n_samples': len(all_atlantis_errors),
            'mean': float(np.mean(all_atlantis_errors)),
            'std': float(np.std(all_atlantis_errors)),
            'median': float(np.median(all_atlantis_errors)),
            'min': float(np.min(all_atlantis_errors)),
            'max': float(np.max(all_atlantis_errors)),
        },
        'atlantis_with_distance': {
            'n_samples': len(atlantis_with_distance),
            'mean': float(np.mean(atlantis_with_distance)),
            'std': float(np.std(atlantis_with_distance)),
            'median': float(np.median(atlantis_with_distance)),
            'min': float(np.min(atlantis_with_distance)),
            'max': float(np.max(atlantis_with_distance)),
        },
        'atlantis_without_distance': {
            'n_samples': len(atlantis_without_distance),
            'mean': float(np.mean(atlantis_without_distance)),
            'std': float(np.std(atlantis_without_distance)),
            'median': float(np.median(atlantis_without_distance)),
            'min': float(np.min(atlantis_without_distance)),
            'max': float(np.max(atlantis_without_distance)),
        },
        'baseline': {
            'n_samples': len(all_baseline_errors),
            'mean': float(np.mean(all_baseline_errors)),
            'std': float(np.std(all_baseline_errors)),
            'median': float(np.median(all_baseline_errors)),
            'min': float(np.min(all_baseline_errors)),
            'max': float(np.max(all_baseline_errors)),
        },
    }

    if exp5_mean is not None:
        summary['exp5_mean'] = float(exp5_mean)

    summary_path = OUTPUT_DIR / "probe_generalization_summary_with_exp5.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
