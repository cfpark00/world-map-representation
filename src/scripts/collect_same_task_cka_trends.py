#!/usr/bin/env python3
"""
Collect CKA values for same-task, different-seed comparisons.
This isolates the effect of multi-task training on representation alignment.

For each task variant (1-7), we compare different seeds of the SAME task:
- PT1-X: Single task training (exp4)
- PT2-X: Two-task training (exp2)
- PT3-X: Three-task training (exp2)

We extract the off-diagonal entries of each 3×3 seed block:
- (orig, seed1), (orig, seed2), (seed1, seed2)
This gives 6 comparisons per variant (3 unique pairs × 2 orderings)
But we only count unique pairs once, giving 3 values × 7 variants = 21 values per (PT level, layer)

IMPORTANT: PT1-5 has special seed mapping:
- seed2 failed, so seed3 → seed2, seed4 → seed3
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def get_pt1_variant_seeds(variant):
    """Get available seeds for PT1 variant."""
    if variant == 5:
        # PT1-5: seed2 failed, seed3→seed2, seed4→seed3
        return {
            42: 'orig',
            2: 'seed3',  # Maps to seed2
            3: 'seed4',  # Maps to seed3
        }
    else:
        # Standard variants have orig, seed1, seed2
        # Note: PT1-7 (crossing) exists but may have limited training
        return {
            42: 'orig',
            1: 'seed1',
            2: 'seed2',
        }


def get_pt23_variant_seeds(variant):
    """Get available seeds for PT2/PT3 variant."""
    # Standard variants have orig, seed1, seed2
    # Note: variants 7,8 contain crossing task which may have limited training
    return {
        42: 'orig',
        1: 'seed1',
        2: 'seed2',
    }


def collect_pt1_same_task_data(base_path):
    """Collect PT1-X same-task, different-seed CKA values."""
    cka_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp4' / 'cka_analysis'

    data = []

    if not cka_dir.exists():
        print(f"Warning: PT1-X CKA directory not found: {cka_dir}")
        return data

    # For each variant (1-7, including crossing)
    for variant in range(1, 8):
        seeds = get_pt1_variant_seeds(variant)
        if not seeds:
            continue

        # Get all unique seed pairs
        seed_list = sorted(seeds.keys())

        for i, seed1 in enumerate(seed_list):
            for seed2 in seed_list[i+1:]:  # Only upper triangle (unique pairs)
                # Construct experiment names
                if seed1 == 42:
                    exp1 = f'pt1-{variant}'
                else:
                    exp1 = f'pt1-{variant}_seed{seed1}'

                if seed2 == 42:
                    exp2 = f'pt1-{variant}'
                else:
                    exp2 = f'pt1-{variant}_seed{seed2}'

                # Look for pair directory (try both orderings)
                pair_names = [
                    f'{exp1}_vs_{exp2}',
                    f'{exp2}_vs_{exp1}',
                ]

                for pair_name in pair_names:
                    pair_dir = cka_dir / pair_name
                    if pair_dir.exists():
                        # Read all layers
                        for layer_dir in pair_dir.glob('layer*'):
                            layer = int(layer_dir.name.replace('layer', ''))
                            summary_file = layer_dir / 'summary.json'

                            if summary_file.exists():
                                with open(summary_file) as f:
                                    summary = json.load(f)

                                # Map seeds for output (handle PT1-5 remapping)
                                output_seed1 = seed1 if variant != 5 else (2 if seed1 == 2 else (3 if seed1 == 3 else seed1))
                                output_seed2 = seed2 if variant != 5 else (2 if seed2 == 2 else (3 if seed2 == 3 else seed2))

                                data.append({
                                    'prefix': 'pt1',
                                    'variant': variant,
                                    'seed1': min(output_seed1, output_seed2),  # Canonical ordering
                                    'seed2': max(output_seed1, output_seed2),
                                    'layer': layer,
                                    'final_cka': summary['final_cka'],
                                    'is_crossing': variant == 7,
                                })
                        break  # Found the pair

    return data


def collect_pt23_same_task_data(base_path, prefix):
    """Collect PT2/PT3 same-task, different-seed CKA values."""
    # PT2 and PT3 data is in exp2 directory
    cka_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all'

    data = []

    if not cka_dir.exists():
        print(f"Warning: {prefix.upper()} CKA directory not found: {cka_dir}")
        return data

    # For each variant (including crossing tasks)
    max_variant = 8 if prefix == 'pt2' else 9
    for variant in range(1, max_variant):
        seeds = get_pt23_variant_seeds(variant)
        if not seeds:
            continue

        # Get all unique seed pairs
        seed_list = sorted(seeds.keys())

        for i, seed1 in enumerate(seed_list):
            for seed2 in seed_list[i+1:]:  # Only upper triangle (unique pairs)
                # Construct experiment names
                if seed1 == 42:
                    exp1 = f'{prefix}-{variant}'
                else:
                    exp1 = f'{prefix}-{variant}_seed{seed1}'

                if seed2 == 42:
                    exp2 = f'{prefix}-{variant}'
                else:
                    exp2 = f'{prefix}-{variant}_seed{seed2}'

                # Look for pair directory (try both orderings)
                pair_names = [
                    f'{exp1}_vs_{exp2}',
                    f'{exp2}_vs_{exp1}',
                ]

                for pair_name in pair_names:
                    pair_dir = cka_dir / pair_name
                    if pair_dir.exists():
                        # Read all layers
                        for layer_dir in pair_dir.glob('layer*'):
                            layer = int(layer_dir.name.replace('layer', ''))
                            summary_file = layer_dir / 'summary.json'

                            if summary_file.exists():
                                with open(summary_file) as f:
                                    summary = json.load(f)

                                # Check if this variant contains crossing task
                                is_crossing = (prefix == 'pt2' and variant in [4, 7]) or \
                                              (prefix == 'pt3' and variant in [3, 5, 7])

                                data.append({
                                    'prefix': prefix,
                                    'variant': variant,
                                    'seed1': min(seed1, seed2),  # Canonical ordering
                                    'seed2': max(seed1, seed2),
                                    'layer': layer,
                                    'final_cka': summary['final_cka'],
                                    'is_crossing': is_crossing,
                                })
                        break  # Found the pair

    return data


def main():
    base_path = Path(__file__).resolve().parents[2]
    output_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_trends'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting same-task, different-seed CKA data...")

    # Collect data
    pt1_data = collect_pt1_same_task_data(base_path)
    pt2_data = collect_pt23_same_task_data(base_path, 'pt2')
    pt3_data = collect_pt23_same_task_data(base_path, 'pt3')

    print(f"Found {len(pt1_data)} PT1 same-task comparisons")
    print(f"Found {len(pt2_data)} PT2 same-task comparisons")
    print(f"Found {len(pt3_data)} PT3 same-task comparisons")

    # Combine into dataframe
    all_data = pt1_data + pt2_data + pt3_data
    df = pd.DataFrame(all_data)

    # Save to CSV
    output_file = output_dir / 'same_task_cka_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Print statistics by prefix and layer
    print("\nStatistics (mean ± SEM):")
    print("=" * 60)
    for prefix in ['pt1', 'pt2', 'pt3']:
        prefix_data = df[df['prefix'] == prefix]
        if len(prefix_data) == 0:
            continue

        print(f"\n{prefix.upper()}:")
        for layer in [3, 4, 5, 6]:
            layer_data = prefix_data[prefix_data['layer'] == layer]
            if len(layer_data) > 0:
                mean_cka = layer_data['final_cka'].mean()
                sem_cka = layer_data['final_cka'].std() / np.sqrt(len(layer_data))
                print(f"  Layer {layer}: {mean_cka:.4f} ± {sem_cka:.4f} (n={len(layer_data)})")

    # Print breakdown by variant
    print("\n\nBreakdown by variant:")
    print("=" * 60)
    for prefix in ['pt1', 'pt2', 'pt3']:
        print(f"\n{prefix.upper()}:")
        prefix_data = df[df['prefix'] == prefix]
        if len(prefix_data) == 0:
            continue

        for variant in sorted(prefix_data['variant'].unique()):
            variant_data = prefix_data[prefix_data['variant'] == variant]
            n_seeds = len(variant_data[variant_data['layer'] == 5])  # Count at one layer
            print(f"  {prefix}-{variant}: {n_seeds} seed pairs")


if __name__ == '__main__':
    main()
