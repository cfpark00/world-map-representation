#!/usr/bin/env python3
"""
Generate 8×8 (actually 7×7, pt2-8 missing) CKA matrix for PT2 original models, layer 5.
Loads from /data/experiments/cka_analysis_pt2/
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
    """Load CKA values for original PT2 models from revision/exp2/cka_analysis_all."""
    cka_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all'

    # Only use pt2-1 through pt2-7
    variants = list(range(1, 8))  # pt2-1 through pt2-7

    n = len(variants)
    matrix = np.full((n, n), np.nan)

    # Fill diagonal with 1.0
    np.fill_diagonal(matrix, 1.0)

    # Load CKA values
    for i, var1 in enumerate(variants):
        for j, var2 in enumerate(variants):
            if i >= j:
                continue

            pair_name = f'pt2-{var1}_vs_pt2-{var2}'
            pair_dir = cka_dir / pair_name / 'layer5'
            summary_file = pair_dir / 'summary.json'

            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                cka_value = data['final_cka']
                matrix[i, j] = cka_value
                matrix[j, i] = cka_value
            else:
                print(f"Missing: {pair_name}")

    return matrix, variants

def plot_matrix(matrix, variants, output_dir):
    """Plot the CKA matrix."""

    labels = [str(i) for i in variants]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, 9))

    norm = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)
    im = ax.imshow(matrix, cmap='magma', norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=14)

    # Add annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                              ha="center", va="center",
                              color="white" if matrix[i, j] > 0.5 else "black",
                              fontsize=24)

    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_yticklabels(labels, fontsize=24)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'PT2 CKA Matrix ({n}×{n}, Original Models, Layer 5)',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    output_file = output_dir / f'pt2_cka_{n}x{n}_orig_l5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Save matrix as CSV
    csv_file = output_dir / f'pt2_cka_{n}x{n}_orig_l5.csv'
    # Create header with variant numbers
    header = ','.join([str(v) for v in variants])
    np.savetxt(csv_file, matrix, delimiter=',', fmt='%.6f', header=header, comments='')
    print(f"Saved: {csv_file}")

def main():
    base_path = Path(__file__).resolve().parents[2]
    output_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PT2 original CKA matrix...")
    matrix, variants = load_cka_matrix(base_path)

    # Check for missing values (excluding diagonal)
    n_missing = np.sum(np.isnan(matrix))
    if n_missing > 0:
        print(f"Warning: {n_missing} CKA values are missing (including diagonal)!")

    print("\nGenerating plot...")
    plot_matrix(matrix, variants, output_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()
