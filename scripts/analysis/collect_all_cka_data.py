#!/usr/bin/env python3
"""
Collect all CKA data across layers and experiments into a clean CSV format.
Saves both checkpoint-level and summary data.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Task mappings for each experiment
TASK_MAPPINGS = {
    'pt1': {
        1: 'distance', 2: 'trianglearea', 3: 'angle', 4: 'compass',
        5: 'inside', 6: 'perimeter', 7: 'crossing'
    },
    'pt2': {
        1: 'distance', 2: 'angle', 3: 'inside', 4: 'crossing',
        5: 'trianglearea', 6: 'compass', 7: 'perimeter', 8: 'distance'
    },
    'pt3': {
        1: 'distance', 2: 'compass', 3: 'crossing', 4: 'angle',
        5: 'perimeter', 6: 'trianglearea', 7: 'inside', 8: 'distance'
    }
}

# Training data composition for each experiment
TRAINING_DATA = {
    'pt1': {
        1: ['distance'],
        2: ['trianglearea'],
        3: ['angle'],
        4: ['compass'],
        5: ['inside'],
        6: ['perimeter'],
        7: ['crossing']
    },
    'pt2': {
        1: ['distance', 'trianglearea'],
        2: ['angle', 'compass'],
        3: ['inside', 'perimeter'],
        4: ['crossing', 'distance'],
        5: ['trianglearea', 'angle'],
        6: ['compass', 'inside'],
        7: ['perimeter', 'crossing'],
        8: ['distance', 'trianglearea']
    },
    'pt3': {
        1: ['distance', 'angle', 'inside'],
        2: ['compass', 'perimeter', 'crossing'],
        3: ['crossing', 'trianglearea', 'distance'],
        4: ['angle', 'compass', 'inside'],
        5: ['perimeter', 'crossing', 'trianglearea'],
        6: ['trianglearea', 'distance', 'angle'],
        7: ['inside', 'perimeter', 'compass'],
        8: ['distance', 'angle', 'inside']
    }
}

def find_cka_results(prefix, exp1_num, exp2_num, layer):
    """Find CKA results file for a given pair and layer."""

    # Possible directory locations
    possible_dirs = [
        f'/n/home12/cfpark00/WM_1/data/experiments/cka_analysis/{prefix}-{exp1_num}_vs_{prefix}-{exp2_num}_l{layer}',
        f'/n/home12/cfpark00/WM_1/data/experiments/cka_analysis_{prefix}/{prefix}-{exp1_num}_vs_{prefix}-{exp2_num}_l{layer}',
        # Handle old naming format for pt1
        f'/n/home12/cfpark00/WM_1/data/experiments/cka_analysis/{prefix}-{exp1_num}_{TASK_MAPPINGS[prefix][exp1_num]}_vs_{prefix}-{exp2_num}_{TASK_MAPPINGS[prefix][exp2_num]}_l{layer}'
    ]

    for dir_path in possible_dirs:
        results_file = Path(dir_path) / 'cka_results.json'
        if results_file.exists():
            return results_file

    return None

def main():
    output_dir = Path('/n/home12/cfpark00/WM_1/scratch/cka_analysis_clean')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all data
    all_summary_data = []
    all_checkpoint_data = []

    layers = [3, 4, 5, 6]

    for prefix in ['pt1', 'pt2', 'pt3']:
        n_models = 7 if prefix == 'pt1' else 8

        for layer in layers:
            print(f"\nProcessing {prefix} layer {layer}...")

            for i in range(1, n_models + 1):
                for j in range(i + 1, n_models + 1):
                    # Find CKA results file
                    results_file = find_cka_results(prefix, i, j, layer)

                    if not results_file:
                        print(f"  Missing: {prefix}-{i} vs {prefix}-{j} layer {layer}")
                        continue

                    # Load CKA results
                    with open(results_file, 'r') as f:
                        data = json.load(f)

                    # Get experiment info
                    exp1 = f"{prefix}-{i}"
                    exp2 = f"{prefix}-{j}"
                    task1 = TASK_MAPPINGS[prefix][i]
                    task2 = TASK_MAPPINGS[prefix][j]
                    train_data1 = ','.join(TRAINING_DATA[prefix][i])
                    train_data2 = ','.join(TRAINING_DATA[prefix][j])

                    # Check if training data overlaps
                    overlap = len(set(TRAINING_DATA[prefix][i]) & set(TRAINING_DATA[prefix][j])) > 0

                    # Summary data (one row per pair-layer)
                    summary_row = {
                        'prefix': prefix,
                        'exp1': exp1,
                        'exp2': exp2,
                        'exp1_num': i,
                        'exp2_num': j,
                        'layer': layer,
                        'task1': task1,
                        'task2': task2,
                        'train_data1': train_data1,
                        'train_data2': train_data2,
                        'training_overlap': overlap,
                        'final_cka': data.get('final_cka', np.nan),
                        'mean_cka': data.get('mean_cka', np.nan),
                        'std_cka': data.get('std_cka', np.nan),
                        'min_cka': data.get('min_cka', np.nan),
                        'max_cka': data.get('max_cka', np.nan),
                        'n_checkpoints': len(data.get('cka_values', {}))
                    }
                    all_summary_data.append(summary_row)

                    # Checkpoint-level data (one row per checkpoint)
                    cka_values = data.get('cka_values', {})
                    for checkpoint, cka_value in cka_values.items():
                        checkpoint_row = {
                            'prefix': prefix,
                            'exp1': exp1,
                            'exp2': exp2,
                            'layer': layer,
                            'checkpoint': int(checkpoint),
                            'cka': cka_value,
                            'task1': task1,
                            'task2': task2,
                            'training_overlap': overlap
                        }
                        all_checkpoint_data.append(checkpoint_row)

    # Create DataFrames
    summary_df = pd.DataFrame(all_summary_data)
    checkpoint_df = pd.DataFrame(all_checkpoint_data)

    # Sort for clean output
    summary_df = summary_df.sort_values(['prefix', 'layer', 'exp1_num', 'exp2_num'])
    checkpoint_df = checkpoint_df.sort_values(['prefix', 'layer', 'exp1', 'exp2', 'checkpoint'])

    # Save summary CSV
    summary_path = output_dir / 'cka_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print(f"  {len(summary_df)} pair-layer combinations")
    print(f"  File size: {summary_path.stat().st_size / 1024:.1f} KB")

    # Save checkpoint-level CSV
    checkpoint_path = output_dir / 'cka_checkpoints.csv'
    checkpoint_df.to_csv(checkpoint_path, index=False)
    print(f"\nSaved checkpoint data to {checkpoint_path}")
    print(f"  {len(checkpoint_df)} total checkpoint measurements")
    print(f"  File size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save a compact JSON version organized by prefix/layer
    organized_data = {}
    for prefix in ['pt1', 'pt2', 'pt3']:
        organized_data[prefix] = {}
        for layer in layers:
            layer_data = summary_df[(summary_df['prefix'] == prefix) & (summary_df['layer'] == layer)]
            organized_data[prefix][f'layer_{layer}'] = {
                'pairs': layer_data[['exp1', 'exp2', 'final_cka', 'training_overlap']].to_dict('records'),
                'mean_cka_all': layer_data['final_cka'].mean(),
                'mean_cka_no_overlap': layer_data[~layer_data['training_overlap']]['final_cka'].mean() if any(~layer_data['training_overlap']) else None,
                'mean_cka_with_overlap': layer_data[layer_data['training_overlap']]['final_cka'].mean() if any(layer_data['training_overlap']) else None
            }

    json_path = output_dir / 'cka_organized.json'
    with open(json_path, 'w') as f:
        json.dump(organized_data, f, indent=2)
    print(f"\nSaved organized JSON to {json_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for prefix in ['pt1', 'pt2', 'pt3']:
        prefix_data = summary_df[summary_df['prefix'] == prefix]
        print(f"\n{prefix.upper()}:")
        print(f"  Total pairs: {len(prefix_data) // 4}")
        print(f"  Missing pairs: {((7*6//2 if prefix == 'pt1' else 8*7//2) * 4) - len(prefix_data)}")

        for layer in layers:
            layer_data = prefix_data[prefix_data['layer'] == layer]
            if len(layer_data) > 0:
                print(f"  Layer {layer}: {len(layer_data)} pairs, mean CKA = {layer_data['final_cka'].mean():.4f}")

                # Special statistics for non-overlapping pairs
                no_overlap = layer_data[~layer_data['training_overlap']]
                if len(no_overlap) > 0:
                    print(f"    Non-overlapping: {len(no_overlap)} pairs, mean CKA = {no_overlap['final_cka'].mean():.4f}")

if __name__ == '__main__':
    main()