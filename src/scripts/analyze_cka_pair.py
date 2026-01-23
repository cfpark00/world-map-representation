"""
Analyze CKA between two experiments across all checkpoints.

This script computes CKA similarity for a single experiment pair at a specific layer.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')
sys.path.insert(0, str(project_root))

import argparse
import yaml
import json
import shutil
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.utils import init_directory
from src.analysis.cka_v2.load_representations import load_all_checkpoints, align_representations
from src.analysis.cka_v2.compute_cka import compute_cka


def plot_cka_timeline(df, output_path, summary):
    """Plot CKA timeline across checkpoints."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['step'], df['cka'], linewidth=2, color='steelblue')
    ax.axhline(y=summary['mean_cka'], color='gray', linestyle='--', alpha=0.5, label=f"Mean: {summary['mean_cka']:.4f}")

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('CKA', fontsize=12)
    ax.set_title(f"CKA Timeline: {summary['exp1']} vs {summary['exp2']}\nLayer {summary['layer']}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(config_path, overwrite=False, debug=False):
    """Analyze CKA between two experiments."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Initialize output directory
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Create subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    # Copy config to output
    shutil.copy(config_path, output_dir / 'config.yaml')

    if debug:
        print(f"DEBUG MODE: Output will be written to {output_dir}")

    # Load representations for both experiments
    print(f"Loading representations for {config['exp1']['name']}...")
    repr1, meta1 = load_all_checkpoints(Path(config['exp1']['repr_dir']))

    print(f"Loading representations for {config['exp2']['name']}...")
    repr2, meta2 = load_all_checkpoints(Path(config['exp2']['repr_dir']))

    # Find common checkpoints
    common_steps = sorted(set(repr1.keys()) & set(repr2.keys()))

    if len(common_steps) == 0:
        raise ValueError(f"No common checkpoints found between {config['exp1']['name']} and {config['exp2']['name']}")

    print(f"Found {len(common_steps)} common checkpoints")

    # Filter checkpoints if specified
    if config.get('checkpoint_steps') is not None:
        checkpoint_steps = config['checkpoint_steps']
        common_steps = [s for s in common_steps if s in checkpoint_steps]
        print(f"Filtered to {len(common_steps)} specified checkpoints")
    elif config.get('use_final_only', True):
        # Use only the final (largest) checkpoint
        common_steps = [max(common_steps)]
        print(f"Using final checkpoint only: {common_steps[0]}")

    if len(common_steps) == 0:
        raise ValueError(f"No checkpoints remaining after filtering. Available: {sorted(set(repr1.keys()) & set(repr2.keys()))[:10]}")

    # Compute CKA for each checkpoint
    results = []
    city_filter = config.get('city_filter', None)
    kernel_type = config.get('kernel_type', 'linear')
    center_kernels = config.get('center_kernels', True)
    use_gpu = config.get('use_gpu', True)

    for step in tqdm(common_steps, desc="Computing CKA"):
        # Align representations
        R1, R2, common_cities = align_representations(
            repr1[step], meta1[step],
            repr2[step], meta2[step],
            city_filter=city_filter
        )

        if debug and step == common_steps[0]:
            print(f"DEBUG: {len(common_cities)} common cities after filtering")
            print(f"DEBUG: R1 shape: {R1.shape}, R2 shape: {R2.shape}")

        # Compute CKA
        cka_val = compute_cka(
            R1, R2,
            kernel_type=kernel_type,
            centered=center_kernels,
            use_gpu=use_gpu
        )

        results.append({'step': step, 'cka': cka_val})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'cka_timeline.csv', index=False)

    # Compute summary statistics
    summary = {
        'exp1': config['exp1']['name'],
        'exp2': config['exp2']['name'],
        'layer': config.get('layer', None),
        'n_checkpoints': len(results),
        'n_cities': len(common_cities),
        'final_cka': float(results[-1]['cka']),
        'mean_cka': float(df['cka'].mean()),
        'std_cka': float(df['cka'].std()),
        'min_cka': float(df['cka'].min()),
        'max_cka': float(df['cka'].max()),
    }

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot timeline
    if config.get('save_timeline_plot', True):
        plot_cka_timeline(df, output_dir / 'cka_timeline.png', summary)

    print(f"\nResults saved to {output_dir}")
    print(f"Final CKA: {summary['final_cka']:.4f}")
    print(f"Mean CKA: {summary['mean_cka']:.4f} ± {summary['std_cka']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze CKA between two experiments')
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
