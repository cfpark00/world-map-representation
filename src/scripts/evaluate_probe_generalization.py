#!/usr/bin/env python3
"""
Evaluate linear probe generalization from training cities to test cities.

Trains a linear probe on representations from training cities (e.g., non-Atlantis)
to predict x,y coordinates, then evaluates prediction error on test cities (e.g., Atlantis).

Usage:
    python src/scripts/evaluate_probe_generalization.py configs/probe_eval/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path('')
sys.path.insert(0, str(project_root))

from src.utils import init_directory


def load_representations(repr_dir: Path, checkpoint: str = "final"):
    """Load representations and metadata from a checkpoint directory.

    Args:
        repr_dir: Path to representations directory (e.g., .../analysis_higher/.../representations)
        checkpoint: Which checkpoint to load ("final", "last", or specific step number)

    Returns:
        representations: numpy array of shape (n_cities, flat_dim)
        metadata: dict with city_info and other metadata
    """
    # Find the checkpoint directory
    if checkpoint == "final" or checkpoint == "last":
        # Find the highest numbered checkpoint
        ckpt_dirs = [d for d in repr_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not ckpt_dirs:
            raise ValueError(f"No checkpoint directories found in {repr_dir}")
        # Sort by step number and get the last one
        ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.name.split("-")[1]))
        ckpt_dir = ckpt_dirs[-1]
    else:
        ckpt_dir = repr_dir / f"checkpoint-{checkpoint}"
        if not ckpt_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    # Load representations
    repr_path = ckpt_dir / "representations.pt"
    if not repr_path.exists():
        raise ValueError(f"Representations file not found: {repr_path}")

    repr_data = torch.load(repr_path, map_location="cpu")
    representations = repr_data["representations_flat"].numpy()

    # Load metadata
    metadata_path = ckpt_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return representations, metadata


def filter_cities_by_pattern(city_info: list, pattern: str) -> list:
    """Filter city indices based on region pattern.

    Args:
        city_info: List of city dicts with 'region' and 'row_id' keys
        pattern: Filter pattern (e.g., "region:^(?!Atlantis).*" for non-Atlantis)

    Returns:
        List of indices into city_info that match the pattern
    """
    import re

    indices = []

    # Parse pattern - format is "field:regex && field:regex && ..."
    conditions = [c.strip() for c in pattern.split("&&")]

    for i, city in enumerate(city_info):
        match = True
        for condition in conditions:
            if ":" not in condition:
                continue
            field, regex = condition.split(":", 1)
            field = field.strip()
            regex = regex.strip()

            value = str(city.get(field, ""))
            if not re.match(regex, value):
                match = False
                break

        if match:
            indices.append(i)

    return indices


def main():
    parser = argparse.ArgumentParser(description="Evaluate probe generalization")
    parser.add_argument("config_path", type=str, help="Path to config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()

    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["output_dir", "repr_dir", "probe_train", "probe_test"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

    # Extract config values
    output_dir = Path(config["output_dir"])
    repr_dir = Path(config["repr_dir"])
    probe_train_pattern = config["probe_train"]
    probe_test_pattern = config["probe_test"]
    checkpoint = config.get("checkpoint", "final")
    method_config = config.get("method", {"name": "linear"})
    seed = config.get("seed", 42)

    # Initialize output directory
    output_dir = init_directory(output_dir, overwrite=args.overwrite)

    # Copy config to output
    import shutil
    shutil.copy(args.config_path, output_dir / "config.yaml")

    print("=" * 60)
    print("Probe Generalization Evaluation")
    print("=" * 60)
    print(f"Representations: {repr_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Train pattern: {probe_train_pattern}")
    print(f"Test pattern: {probe_test_pattern}")
    print(f"Method: {method_config.get('name', 'linear')}")

    # Load representations
    print("\nLoading representations...")
    representations, metadata = load_representations(repr_dir, checkpoint)
    city_info = metadata["city_info"]

    print(f"Loaded {len(city_info)} cities, representation dim: {representations.shape[1]}")

    # Filter cities for train and test
    train_candidates = filter_cities_by_pattern(city_info, probe_train_pattern)
    test_indices = filter_cities_by_pattern(city_info, probe_test_pattern)

    # Remove any test indices that are in train candidates
    test_indices = [i for i in test_indices if i not in train_candidates]

    # Subsample training cities if n_train specified
    n_train = config.get("n_train", len(train_candidates))
    n_baseline = config.get("n_baseline", 100)  # baseline test on held-out train-pattern cities

    np.random.seed(seed)
    if n_train + n_baseline > len(train_candidates):
        raise ValueError(f"n_train ({n_train}) + n_baseline ({n_baseline}) > available train candidates ({len(train_candidates)})")

    # Shuffle and split
    shuffled_train_candidates = np.random.permutation(train_candidates).tolist()
    train_indices = shuffled_train_candidates[:n_train]
    baseline_indices = shuffled_train_candidates[n_train:n_train + n_baseline]

    print(f"Training cities: {len(train_indices)}")
    print(f"Test cities (target): {len(test_indices)}")
    print(f"Baseline cities (held-out from train pattern): {len(baseline_indices)}")

    if len(train_indices) == 0:
        raise ValueError("No training cities found matching pattern!")
    if len(test_indices) == 0:
        raise ValueError("No test cities found matching pattern!")

    # Extract coordinates and representations
    X_train = representations[train_indices]
    X_test = representations[test_indices]
    X_baseline = representations[baseline_indices]

    x_train = np.array([city_info[i]["x"] for i in train_indices])
    y_train = np.array([city_info[i]["y"] for i in train_indices])
    x_test = np.array([city_info[i]["x"] for i in test_indices])
    y_test = np.array([city_info[i]["y"] for i in test_indices])
    x_baseline = np.array([city_info[i]["x"] for i in baseline_indices])
    y_baseline = np.array([city_info[i]["y"] for i in baseline_indices])

    # Center targets (predict deviations from training mean)
    x_train_mean = x_train.mean()
    y_train_mean = y_train.mean()

    x_train_centered = x_train - x_train_mean
    y_train_centered = y_train - y_train_mean
    x_test_centered = x_test - x_train_mean
    y_test_centered = y_test - y_train_mean

    # Create probes
    method_name = method_config.get("name", "linear")
    if method_name == "linear":
        x_probe = LinearRegression()
        y_probe = LinearRegression()
    elif method_name == "ridge":
        alpha = method_config.get("alpha", 10.0)
        x_probe = Ridge(alpha=alpha)
        y_probe = Ridge(alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    print(f"\nTraining probes...")

    # Train probes
    x_probe.fit(X_train, x_train_centered)
    y_probe.fit(X_train, y_train_centered)

    # Predict on train set
    x_train_pred_centered = x_probe.predict(X_train)
    y_train_pred_centered = y_probe.predict(X_train)
    x_train_pred = x_train_pred_centered + x_train_mean
    y_train_pred = y_train_pred_centered + y_train_mean

    # Predict on test set
    x_test_pred_centered = x_probe.predict(X_test)
    y_test_pred_centered = y_probe.predict(X_test)
    x_test_pred = x_test_pred_centered + x_train_mean
    y_test_pred = y_test_pred_centered + y_train_mean

    # Predict on baseline set (held-out non-Atlantis cities)
    x_baseline_pred_centered = x_probe.predict(X_baseline)
    y_baseline_pred_centered = y_probe.predict(X_baseline)
    x_baseline_pred = x_baseline_pred_centered + x_train_mean
    y_baseline_pred = y_baseline_pred_centered + y_train_mean

    # Calculate metrics
    # Training metrics
    train_x_r2 = r2_score(x_train, x_train_pred)
    train_y_r2 = r2_score(y_train, y_train_pred)
    train_x_mae = mean_absolute_error(x_train, x_train_pred)
    train_y_mae = mean_absolute_error(y_train, y_train_pred)
    train_dist_errors = np.sqrt((x_train - x_train_pred)**2 + (y_train - y_train_pred)**2)
    train_mean_dist_error = np.mean(train_dist_errors)
    train_median_dist_error = np.median(train_dist_errors)

    # Test metrics
    test_x_r2 = r2_score(x_test, x_test_pred)
    test_y_r2 = r2_score(y_test, y_test_pred)
    test_x_mae = mean_absolute_error(x_test, x_test_pred)
    test_y_mae = mean_absolute_error(y_test, y_test_pred)
    test_dist_errors = np.sqrt((x_test - x_test_pred)**2 + (y_test - y_test_pred)**2)
    test_mean_dist_error = np.mean(test_dist_errors)
    test_median_dist_error = np.median(test_dist_errors)
    test_std_dist_error = np.std(test_dist_errors)
    test_max_dist_error = np.max(test_dist_errors)
    test_min_dist_error = np.min(test_dist_errors)

    # Baseline metrics (held-out non-Atlantis cities)
    baseline_dist_errors = np.sqrt((x_baseline - x_baseline_pred)**2 + (y_baseline - y_baseline_pred)**2)
    baseline_mean_dist_error = np.mean(baseline_dist_errors)
    baseline_median_dist_error = np.median(baseline_dist_errors)

    # Location-aware metrics: does the probe know WHERE the test region is?
    test_x_mean = x_test.mean()
    test_y_mean = y_test.mean()
    pred_x_mean = x_test_pred.mean()
    pred_y_mean = y_test_pred.mean()
    # Bias: how far is the mean prediction from the true mean?
    x_bias = pred_x_mean - test_x_mean
    y_bias = pred_y_mean - test_y_mean
    location_bias = np.sqrt(x_bias**2 + y_bias**2)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print("\nTraining Set:")
    print(f"  X R²: {train_x_r2:.4f}")
    print(f"  Y R²: {train_y_r2:.4f}")
    print(f"  Mean R²: {(train_x_r2 + train_y_r2) / 2:.4f}")
    print(f"  X MAE: {train_x_mae:.2f}")
    print(f"  Y MAE: {train_y_mae:.2f}")
    print(f"  Mean Distance Error: {train_mean_dist_error:.2f}")
    print(f"  Median Distance Error: {train_median_dist_error:.2f}")

    print("\nTest Set:")
    print(f"  X R²: {test_x_r2:.4f}")
    print(f"  Y R²: {test_y_r2:.4f}")
    print(f"  Mean R²: {(test_x_r2 + test_y_r2) / 2:.4f}")
    print(f"  X MAE: {test_x_mae:.2f}")
    print(f"  Y MAE: {test_y_mae:.2f}")
    print(f"  Mean Distance Error: {test_mean_dist_error:.2f}")
    print(f"  Median Distance Error: {test_median_dist_error:.2f}")
    print(f"  Std Distance Error: {test_std_dist_error:.2f}")
    print(f"  Min Distance Error: {test_min_dist_error:.2f}")
    print(f"  Max Distance Error: {test_max_dist_error:.2f}")

    print(f"\nBaseline (held-out non-Atlantis, {len(baseline_indices)} cities):")
    print(f"  Mean Distance Error: {baseline_mean_dist_error:.2f}")
    print(f"  Median Distance Error: {baseline_median_dist_error:.2f}")

    print("\nLocation Awareness (does probe know WHERE test region is?):")
    print(f"  True center: ({test_x_mean:.1f}, {test_y_mean:.1f})")
    print(f"  Predicted center: ({pred_x_mean:.1f}, {pred_y_mean:.1f})")
    print(f"  Location Bias (center error): {location_bias:.1f}")

    print("\n" + "=" * 60)
    print("SUMMARY: Test vs Baseline Distance Error")
    print("=" * 60)
    print(f"  Test (Atlantis):     {test_mean_dist_error:.1f}")
    print(f"  Baseline (non-Atl):  {baseline_mean_dist_error:.1f}")

    # Save results
    results = {
        "config": {
            "repr_dir": str(repr_dir),
            "checkpoint": checkpoint,
            "probe_train": probe_train_pattern,
            "probe_test": probe_test_pattern,
            "method": method_config,
            "seed": seed,
        },
        "train": {
            "n_cities": len(train_indices),
            "x_r2": float(train_x_r2),
            "y_r2": float(train_y_r2),
            "mean_r2": float((train_x_r2 + train_y_r2) / 2),
            "x_mae": float(train_x_mae),
            "y_mae": float(train_y_mae),
            "mean_dist_error": float(train_mean_dist_error),
            "median_dist_error": float(train_median_dist_error),
        },
        "test": {
            "n_cities": len(test_indices),
            "x_r2": float(test_x_r2),
            "y_r2": float(test_y_r2),
            "mean_r2": float((test_x_r2 + test_y_r2) / 2),
            "x_mae": float(test_x_mae),
            "y_mae": float(test_y_mae),
            "mean_dist_error": float(test_mean_dist_error),
            "median_dist_error": float(test_median_dist_error),
            "std_dist_error": float(test_std_dist_error),
            "min_dist_error": float(test_min_dist_error),
            "max_dist_error": float(test_max_dist_error),
        },
        "baseline": {
            "n_cities": len(baseline_indices),
            "mean_dist_error": float(baseline_mean_dist_error),
            "median_dist_error": float(baseline_median_dist_error),
        },
        "location_awareness": {
            "true_center_x": float(test_x_mean),
            "true_center_y": float(test_y_mean),
            "pred_center_x": float(pred_x_mean),
            "pred_center_y": float(pred_y_mean),
            "x_bias": float(x_bias),
            "y_bias": float(y_bias),
            "location_bias": float(location_bias),
        },
    }

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save per-city predictions for test set
    test_predictions = []
    for i, idx in enumerate(test_indices):
        city = city_info[idx]
        test_predictions.append({
            "city_id": city["row_id"],
            "name": city["name"],
            "region": city.get("region", ""),
            "x_true": float(x_test[i]),
            "y_true": float(y_test[i]),
            "x_pred": float(x_test_pred[i]),
            "y_pred": float(y_test_pred[i]),
            "dist_error": float(test_dist_errors[i]),
        })

    predictions_df = pd.DataFrame(test_predictions)
    predictions_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Save per-city predictions for baseline set
    baseline_predictions = []
    for i, idx in enumerate(baseline_indices):
        city = city_info[idx]
        baseline_predictions.append({
            "city_id": city["row_id"],
            "name": city["name"],
            "region": city.get("region", ""),
            "x_true": float(x_baseline[i]),
            "y_true": float(y_baseline[i]),
            "x_pred": float(x_baseline_pred[i]),
            "y_pred": float(y_baseline_pred[i]),
            "dist_error": float(baseline_dist_errors[i]),
        })

    baseline_predictions_df = pd.DataFrame(baseline_predictions)
    baseline_predictions_path = output_dir / "baseline_predictions.csv"
    baseline_predictions_df.to_csv(baseline_predictions_path, index=False)
    print(f"Baseline predictions saved to {baseline_predictions_path}")

    # Create visualization figure
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: World map with train cities (subsampled) and test cities
    ax = axes[0, 0]
    # Subsample train cities for visibility
    n_train_show = min(500, len(x_train))
    train_show_idx = np.random.choice(len(x_train), n_train_show, replace=False)
    ax.scatter(x_train[train_show_idx], y_train[train_show_idx],
               c='blue', alpha=0.3, s=5, label=f'Train ({len(x_train)} cities)')
    ax.scatter(x_test, y_test, c='red', alpha=0.8, s=30, label=f'Test ({len(x_test)} cities)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('City Locations (World Map)')
    ax.legend()
    ax.set_xlim(-1900, 1900)
    ax.set_ylim(-1000, 1000)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Test set - True vs Predicted locations
    ax = axes[0, 1]
    ax.scatter(x_test, y_test, c='green', alpha=0.7, s=50, label='True', marker='o')
    ax.scatter(x_test_pred, y_test_pred, c='red', alpha=0.7, s=50, label='Predicted', marker='x')
    # Draw arrows from predicted to true
    for i in range(len(x_test)):
        ax.annotate('', xy=(x_test[i], y_test[i]), xytext=(x_test_pred[i], y_test_pred[i]),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Test Set: True vs Predicted\nMean Error: {test_mean_dist_error:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 3: X coordinate - True vs Predicted
    ax = axes[0, 2]
    ax.scatter(x_test, x_test_pred, c='blue', alpha=0.7, s=30)
    # Perfect prediction line
    x_range = [min(x_test.min(), x_test_pred.min()), max(x_test.max(), x_test_pred.max())]
    ax.plot(x_range, x_range, 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True X')
    ax.set_ylabel('Predicted X')
    ax.set_title(f'X Coordinate\nR²={test_x_r2:.3f}, MAE={test_x_mae:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 4: Y coordinate - True vs Predicted
    ax = axes[1, 0]
    ax.scatter(y_test, y_test_pred, c='blue', alpha=0.7, s=30)
    # Perfect prediction line
    y_range = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    ax.plot(y_range, y_range, 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True Y')
    ax.set_ylabel('Predicted Y')
    ax.set_title(f'Y Coordinate\nR²={test_y_r2:.3f}, MAE={test_y_mae:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 5: Error distribution histogram
    ax = axes[1, 1]
    ax.hist(test_dist_errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(test_mean_dist_error, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {test_mean_dist_error:.1f}')
    ax.axvline(test_median_dist_error, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {test_median_dist_error:.1f}')
    ax.set_xlabel('Distance Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary text
    ax = axes[1, 2]
    ax.axis('off')

    # Calculate additional context
    x_test_range = x_test.max() - x_test.min()
    y_test_range = y_test.max() - y_test.min()
    x_test_var = np.var(x_test)
    y_test_var = np.var(y_test)

    summary_text = f"""
    TRAIN SET ({len(train_indices)} cities)
    ─────────────────────────
    X Range: [{x_train.min():.0f}, {x_train.max():.0f}]
    Y Range: [{y_train.min():.0f}, {y_train.max():.0f}]
    X R²: {train_x_r2:.4f}
    Y R²: {train_y_r2:.4f}
    Mean Dist Error: {train_mean_dist_error:.1f}

    TEST SET ({len(test_indices)} cities)
    ─────────────────────────
    X Range: [{x_test.min():.0f}, {x_test.max():.0f}] (span={x_test_range:.0f})
    Y Range: [{y_test.min():.0f}, {y_test.max():.0f}] (span={y_test_range:.0f})
    X Variance: {x_test_var:.1f}
    Y Variance: {y_test_var:.1f}

    X R²: {test_x_r2:.4f}
    Y R²: {test_y_r2:.4f}
    Mean Dist Error: {test_mean_dist_error:.1f}

    NOTE: R² can be negative when test set
    has small variance (narrow region) and
    predictions have larger spread than truth.
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Probe Generalization: {probe_train_pattern} → {probe_test_pattern}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {fig_path}")

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
