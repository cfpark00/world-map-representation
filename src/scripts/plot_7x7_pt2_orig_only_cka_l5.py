#!/usr/bin/env python3
"""
Generate 7×7 CKA matrix for PT2 ORIGINAL seed only, layer 5.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import FuncNorm

def three_slope_mapping(x, breakpoint1=0.4, breakpoint2=0.6):
    """Piecewise linear mapping with three segments."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask1 = x <= breakpoint1
    result[mask1] = (x[mask1] / breakpoint1) * 0.2
    mask2 = (x > breakpoint1) & (x <= breakpoint2)
    result[mask2] = 0.2 + ((x[mask2] - breakpoint1) / (breakpoint2 - breakpoint1)) * 0.3
    mask3 = x > breakpoint2
    result[mask3] = 0.5 + ((x[mask3] - breakpoint2) / (1.0 - breakpoint2)) * 0.5
    return result

def three_slope_inverse(x, breakpoint1=0.4, breakpoint2=0.6):
    """Inverse mapping for three-slope normalization."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask1 = x <= 0.2
    result[mask1] = (x[mask1] / 0.2) * breakpoint1
    mask2 = (x > 0.2) & (x <= 0.5)
    result[mask2] = breakpoint1 + ((x[mask2] - 0.2) / 0.3) * (breakpoint2 - breakpoint1)
    mask3 = x > 0.5
    result[mask3] = breakpoint2 + ((x[mask3] - 0.5) / 0.5) * (1.0 - breakpoint2)
    return result

def load_cka_matrix(base_path):
    """Load CKA values for original PT2 models only."""
    cka_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all'

    variants = list(range(1, 8))  # pt2-1 through pt2-7
    n = len(variants)
    matrix = np.full((n, n), np.nan)

    # Fill diagonal with 1.0
    np.fill_diagonal(matrix, 1.0)

    # Load CKA values for orig only
    for i, var1 in enumerate(variants):
        for j, var2 in enumerate(variants):
            if i >= j:
                continue

            exp1 = f'pt2-{var1}'
            exp2 = f'pt2-{var2}'

            pair_name = f'{exp1}_vs_{exp2}'
            pair_dir = cka_dir / pair_name / 'layer5'

            summary_file = pair_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                cka_value = data['final_cka']
                matrix[i, j] = cka_value
                matrix[j, i] = cka_value

    return matrix, variants

def plot_7x7_matrix(matrix, variants, output_dir):
    """Plot the 7×7 CKA matrix."""

    labels = [str(i) for i in variants]

    fig, ax = plt.subplots(figsize=(10, 9))

    norm = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)
    im = ax.imshow(matrix, cmap='magma', norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=14)

    # Add annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center",
                          color="white" if matrix[i, j] > 0.5 else "black",
                          fontsize=24)

    # Set ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_yticklabels(labels, fontsize=24)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('PT2 CKA Matrix (7×7, Original Seed Only, Layer 5)',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    output_file = output_dir / 'pt2_cka_7x7_orig_only_l5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Save matrix as CSV
    csv_file = output_dir / 'pt2_cka_7x7_orig_only_l5.csv'
    np.savetxt(csv_file, matrix, delimiter=',', fmt='%.6f')
    print(f"Saved: {csv_file}")

def main():
    base_path = Path(__file__).resolve().parents[2]
    output_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PT2 original seed CKA matrix...")
    matrix, variants = load_cka_matrix(base_path)

    # Check for missing values
    n_missing = np.sum(np.isnan(matrix)) - 7  # Exclude diagonal
    if n_missing > 0:
        print(f"Warning: {n_missing} CKA values are missing!")

    print("\nGenerating plot...")
    plot_7x7_matrix(matrix, variants, output_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()
