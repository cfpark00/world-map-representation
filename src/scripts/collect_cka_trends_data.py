#!/usr/bin/env python3
"""
Collect CKA data from exp4 (PT1-X) and exp2 (PT2/PT3) for trends analysis.
"""

import json
import pandas as pd
from pathlib import Path

def get_pt1x_task_sets():
    """Define task sets for each PT1-X experiment."""
    return {
        'pt1-1': {'distance'},
        'pt1-2': {'trianglearea'},
        'pt1-3': {'angle'},
        'pt1-4': {'compass'},
        'pt1-5': {'inside'},
        'pt1-6': {'perimeter'},
        'pt1-7': {'crossing'},
    }

def get_pt2_task_sets():
    """Define task sets for each PT2 experiment."""
    return {
        'pt2-1': {'distance', 'trianglearea'},
        'pt2-2': {'angle', 'compass'},
        'pt2-3': {'inside', 'perimeter'},
        'pt2-4': {'crossing', 'distance'},
        'pt2-5': {'trianglearea', 'angle'},
        'pt2-6': {'compass', 'inside'},
        'pt2-7': {'perimeter', 'crossing'},
    }

def get_pt3_task_sets():
    """Define task sets for each PT3 experiment."""
    return {
        'pt3-1': {'distance', 'trianglearea', 'angle'},
        'pt3-2': {'compass', 'inside', 'perimeter'},
        'pt3-3': {'crossing', 'distance', 'trianglearea'},
        'pt3-4': {'angle', 'compass', 'inside'},
        'pt3-5': {'perimeter', 'crossing', 'distance'},
        'pt3-6': {'trianglearea', 'angle', 'compass'},
        'pt3-7': {'inside', 'perimeter', 'crossing'},
    }

def has_task_overlap(tasks1, tasks2):
    """Check if two task sets have any overlap."""
    return len(tasks1 & tasks2) > 0

def parse_exp_name(exp_name):
    """Parse experiment name to get prefix, variant, and seed.

    Special handling for pt1-5:
    - seed2 failed, so seed3 -> seed2, seed4 -> seed3
    """
    # Examples: pt1-1, pt1-1_seed1, pt2-3_seed2
    if '_seed' in exp_name:
        base, seed_part = exp_name.split('_seed')
        seed = int(seed_part)
    else:
        base = exp_name
        seed = 42  # original

    # Parse base (e.g., pt1-1, pt2-3)
    prefix, variant = base.rsplit('-', 1)

    # Remap pt1-5 seeds: seed3->seed2, seed4->seed3
    if base == 'pt1-5':
        if seed == 3:
            seed = 2
        elif seed == 4:
            seed = 3

    return prefix, int(variant), seed

def collect_pt1x_data(base_path):
    """Collect CKA data from PT1-X experiments (exp4).

    NOTE: Excludes pt1-7 (crossing) as it failed to train.
    """
    cka_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp4' / 'cka_analysis'
    task_sets = get_pt1x_task_sets()

    data = []

    if not cka_dir.exists():
        print(f"Warning: PT1-X CKA directory not found: {cka_dir}")
        return data

    # Iterate through pair directories
    for pair_dir in cka_dir.glob('pt1-*_vs_pt1-*'):
        # Parse pair name
        pair_name = pair_dir.name
        parts = pair_name.split('_vs_')
        exp1_name = parts[0]
        exp2_name = parts[1]

        # Parse experiment names
        prefix1, var1, seed1 = parse_exp_name(exp1_name)
        prefix2, var2, seed2 = parse_exp_name(exp2_name)

        # Skip pt1-7 (crossing) - training failed
        if var1 == 7 or var2 == 7:
            continue

        # Get task sets
        exp1_key = f'{prefix1}-{var1}'
        exp2_key = f'{prefix2}-{var2}'
        tasks1 = task_sets.get(exp1_key, set())
        tasks2 = task_sets.get(exp2_key, set())

        # Check for overlap
        overlapping = has_task_overlap(tasks1, tasks2)

        # Check each layer
        for layer_dir in pair_dir.glob('layer*'):
            layer = int(layer_dir.name.replace('layer', ''))
            summary_file = layer_dir / 'summary.json'

            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)

                data.append({
                    'prefix': 'pt1',
                    'exp1': exp1_name,
                    'exp2': exp2_name,
                    'var1': var1,
                    'var2': var2,
                    'seed1': seed1,
                    'seed2': seed2,
                    'layer': layer,
                    'final_cka': summary['final_cka'],
                    'mean_cka': summary['mean_cka'],
                    'std_cka': summary.get('std_cka', 0),
                    'n_checkpoints': summary['n_checkpoints'],
                    'training_overlap': overlapping,
                })

    return data

def collect_pt2_pt3_data(base_path):
    """Collect CKA data from PT2 and PT3 experiments (exp2).

    NOTE: Excludes pt2-7 and pt3-7 as they contain crossing task which failed to train.
    """
    # Check all three directories:
    # - cka_analysis: original cross-seed comparisons
    # - cka_analysis_same_seed: same-seed non-overlapping (layers 3,4,6)
    # - cka_analysis_all: all pairs including overlapping (layer 5 only)
    cka_dirs = [
        base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis',
        base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_same_seed',
        base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_analysis_all',
    ]
    pt2_task_sets = get_pt2_task_sets()
    pt3_task_sets = get_pt3_task_sets()

    data = []

    for cka_dir in cka_dirs:
        if not cka_dir.exists():
            print(f"Warning: PT2/PT3 CKA directory not found: {cka_dir}")
            continue

        # Iterate through pair directories
        for pair_dir in cka_dir.glob('pt*-*_vs_pt*-*'):
            # Parse pair name
            pair_name = pair_dir.name

            # Check if pair has seed suffix (e.g., pt2-1_vs_pt2-2_seed1)
            # In this case, BOTH experiments share the same seed
            pair_seed = None
            if '_seed' in pair_name.split('_vs_')[1]:
                # Extract seed from end of pair name
                last_part = pair_name.split('_vs_')[1]
                if '_seed' in last_part:
                    seed_suffix = last_part.split('_seed')[-1]
                    pair_seed = int(seed_suffix)
                    # Remove seed suffix to get base experiment names
                    pair_name_base = pair_name.rsplit('_seed', 1)[0]
                    parts = pair_name_base.split('_vs_')
                else:
                    parts = pair_name.split('_vs_')
            else:
                parts = pair_name.split('_vs_')

            exp1_name = parts[0]
            exp2_name = parts[1]

            # Parse experiment names
            prefix1, var1, seed1 = parse_exp_name(exp1_name)
            prefix2, var2, seed2 = parse_exp_name(exp2_name)

            # Skip pt2-7 and pt3-7 (contain crossing task which failed)
            if var1 == 7 or var2 == 7:
                continue

            # If pair has seed suffix, override parsed seeds
            if pair_seed is not None:
                seed1 = pair_seed
                seed2 = pair_seed

            # Get task sets
            exp1_key = f'{prefix1}-{var1}'
            exp2_key = f'{prefix2}-{var2}'

            if prefix1 == 'pt2':
                task_sets = pt2_task_sets
            else:  # pt3
                task_sets = pt3_task_sets

            tasks1 = task_sets.get(exp1_key, set())
            tasks2 = task_sets.get(exp2_key, set())

            # Check for overlap
            overlapping = has_task_overlap(tasks1, tasks2)

            # Check each layer
            for layer_dir in pair_dir.glob('layer*'):
                layer = int(layer_dir.name.replace('layer', ''))
                summary_file = layer_dir / 'summary.json'

                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)

                    data.append({
                        'prefix': prefix1,
                        'exp1': exp1_name,
                        'exp2': exp2_name,
                        'var1': var1,
                        'var2': var2,
                        'seed1': seed1,
                        'seed2': seed2,
                        'layer': layer,
                        'final_cka': summary['final_cka'],
                        'mean_cka': summary['mean_cka'],
                        'std_cka': summary.get('std_cka', 0),
                        'n_checkpoints': summary['n_checkpoints'],
                        'training_overlap': overlapping,
                    })

    return data

def main():
    base_path = Path(__file__).resolve().parents[2]
    output_dir = base_path / 'data' / 'experiments' / 'revision' / 'exp2' / 'cka_trends'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting CKA data...")

    # Collect data from all sources
    pt1x_data = collect_pt1x_data(base_path)
    pt2_pt3_data = collect_pt2_pt3_data(base_path)

    print(f"Found {len(pt1x_data)} PT1-X CKA comparisons")
    print(f"Found {len(pt2_pt3_data)} PT2/PT3 CKA comparisons")

    # Combine into dataframe
    all_data = pt1x_data + pt2_pt3_data
    df = pd.DataFrame(all_data)

    # Save to CSV
    output_file = output_dir / 'cka_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")

    # Print statistics
    stats_file = output_dir / 'cka_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("CKA Statistics Summary\n")
        f.write("=" * 80 + "\n\n")

        for prefix in ['pt1', 'pt2', 'pt3']:
            prefix_data = df[df['prefix'] == prefix]
            if len(prefix_data) == 0:
                continue

            f.write(f"{prefix.upper()} (n={len(prefix_data)} comparisons)\n")
            f.write("-" * 40 + "\n")

            for layer in [3, 4, 5, 6]:
                layer_data = prefix_data[prefix_data['layer'] == layer]
                if len(layer_data) > 0:
                    # Non-overlapping only
                    non_overlap = layer_data[~layer_data['training_overlap']]
                    if len(non_overlap) > 0:
                        f.write(f"  Layer {layer} (non-overlapping, n={len(non_overlap)}): "
                               f"{non_overlap['final_cka'].mean():.4f} ± {non_overlap['final_cka'].std():.4f}\n")
            f.write("\n")

    print(f"Saved statistics to: {stats_file}")

    # Print breakdown by seed
    print("\nBreakdown by prefix and seed:")
    for prefix in ['pt1', 'pt2', 'pt3']:
        prefix_data = df[df['prefix'] == prefix]
        if len(prefix_data) == 0:
            continue
        print(f"\n{prefix.upper()}:")
        for seed in sorted(df['seed1'].unique()):
            seed_data = prefix_data[prefix_data['seed1'] == seed]
            if len(seed_data) > 0:
                seed_label = 'orig' if seed == 42 else f'seed{seed}'
                print(f"  {seed_label}: {len(seed_data)} comparisons")

if __name__ == '__main__':
    main()
