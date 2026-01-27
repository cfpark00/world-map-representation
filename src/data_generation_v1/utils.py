#!/usr/bin/env python3
"""
Shared utility functions for data processing and dataset creation.
Contains reusable pair generation strategies for various tasks.
"""
import numpy as np
import pandas as pd


def generate_pairs(df, config, n_pairs):
    """Generate city pairs based on configuration strategy."""
    np.random.seed(config['seed'])

    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        # Generate pairs from all cities (no restrictions)
        return generate_all_pairs(df, n_pairs)

    elif strategy == 'within_groups':
        # Generate pairs only within specified groups
        groups = config['pair_generation'].get('groups', df['group'].unique())
        return generate_within_group_pairs(df, groups, n_pairs)

    elif strategy == 'between_groups':
        # Generate pairs between specified groups
        group_pairs = config['pair_generation']['group_pairs']
        return generate_between_group_pairs(df, group_pairs, n_pairs)

    elif strategy == 'mixed':
        # Mix different pair types with specified ratios
        mix_config = config['pair_generation']['mix']
        return generate_mixed_pairs(df, mix_config, n_pairs)

    elif strategy == 'must_include':
        # All pairs must include at least one city from specified groups
        must_include_groups = config['pair_generation']['must_include_groups']
        return generate_must_include_pairs(df, must_include_groups, n_pairs)

    else:
        raise ValueError(f"Unknown pair generation strategy: {strategy}")


def generate_all_pairs(df, n_pairs):
    """Generate random pairs from all cities."""
    n_cities = len(df)
    n_unique_pairs = n_cities * (n_cities - 1) // 2

    if n_pairs > n_unique_pairs:
        print(f"Warning: Requested {n_pairs} pairs but only {n_unique_pairs} unique pairs exist")
        n_pairs = n_unique_pairs

    # Generate upper triangle indices
    triu_indices = np.triu_indices(n_cities, k=1)

    # Sample pairs
    selected_indices = np.random.choice(len(triu_indices[0]), size=n_pairs, replace=False)
    indices_i = triu_indices[0][selected_indices]
    indices_j = triu_indices[1][selected_indices]

    return indices_i, indices_j


def generate_within_group_pairs(df, groups, n_pairs):
    """Generate pairs only within specified groups."""
    all_pairs_i = []
    all_pairs_j = []

    # Calculate pairs per group (proportional to group size squared)
    group_sizes = {g: len(df[df['group'] == g]) for g in groups}
    total_weight = sum(s * (s - 1) / 2 for s in group_sizes.values())

    if total_weight == 0:
        raise ValueError("No valid pairs can be generated within specified groups")

    for group in groups:
        group_df = df[df['group'] == group]
        group_indices = group_df.index.values
        n_group = len(group_indices)

        if n_group < 2:
            continue

        # Calculate this group's share of pairs
        group_weight = n_group * (n_group - 1) / 2
        n_group_pairs = int(n_pairs * group_weight / total_weight)

        if n_group_pairs == 0:
            continue

        # Generate pairs within this group
        triu_indices = np.triu_indices(n_group, k=1)
        n_available = len(triu_indices[0])
        n_sample = min(n_group_pairs, n_available)

        selected = np.random.choice(n_available, size=n_sample, replace=False)
        pairs_i = group_indices[triu_indices[0][selected]]
        pairs_j = group_indices[triu_indices[1][selected]]

        all_pairs_i.extend(pairs_i)
        all_pairs_j.extend(pairs_j)

    return np.array(all_pairs_i), np.array(all_pairs_j)


def generate_between_group_pairs(df, group_pairs, n_pairs):
    """Generate pairs between specified group pairs."""
    all_pairs_i = []
    all_pairs_j = []

    # Calculate total possible pairs for weighting
    total_possible = 0
    for gp in group_pairs:
        g1_size = len(df[df['group'] == gp[0]])
        g2_size = len(df[df['group'] == gp[1]])
        total_possible += g1_size * g2_size

    for group_pair in group_pairs:
        g1, g2 = group_pair
        g1_indices = df[df['group'] == g1].index.values
        g2_indices = df[df['group'] == g2].index.values

        # Calculate this pair's share
        n_possible = len(g1_indices) * len(g2_indices)
        n_pair_samples = int(n_pairs * n_possible / total_possible)

        if n_pair_samples == 0:
            continue

        # Generate all possible pairs and sample
        g1_repeated = np.repeat(g1_indices, len(g2_indices))
        g2_tiled = np.tile(g2_indices, len(g1_indices))

        selected = np.random.choice(len(g1_repeated), size=min(n_pair_samples, len(g1_repeated)), replace=False)

        all_pairs_i.extend(g1_repeated[selected])
        all_pairs_j.extend(g2_tiled[selected])

    return np.array(all_pairs_i), np.array(all_pairs_j)


def generate_mixed_pairs(df, mix_config, n_pairs):
    """Generate mixed pairs with specified ratios."""
    all_pairs_i = []
    all_pairs_j = []

    for mix_item in mix_config:
        ratio = mix_item['ratio']
        n_mix_pairs = int(n_pairs * ratio)

        if mix_item['type'] == 'within_group':
            groups = mix_item['groups']
            pairs_i, pairs_j = generate_within_group_pairs(df, groups, n_mix_pairs)

        elif mix_item['type'] == 'between_groups':
            group_pairs = mix_item['group_pairs']
            pairs_i, pairs_j = generate_between_group_pairs(df, group_pairs, n_mix_pairs)

        elif mix_item['type'] == 'all':
            pairs_i, pairs_j = generate_all_pairs(df, n_mix_pairs)

        all_pairs_i.extend(pairs_i)
        all_pairs_j.extend(pairs_j)

    # Shuffle combined pairs
    shuffle_idx = np.random.permutation(len(all_pairs_i))
    return np.array(all_pairs_i)[shuffle_idx], np.array(all_pairs_j)[shuffle_idx]


def generate_must_include_pairs(df, must_include_groups, n_pairs):
    """Generate pairs where at least one city must be from specified groups."""
    must_include_mask = df['group'].isin(must_include_groups)
    must_include_indices = df[must_include_mask].index.values
    all_indices = df.index.values

    if len(must_include_indices) == 0:
        raise ValueError("No cities found in must_include groups")

    all_pairs_i = []
    all_pairs_j = []

    for _ in range(n_pairs):
        # Simple: pick one from must_include, one from all cities
        must_idx = np.random.choice(must_include_indices)
        other_idx = np.random.choice(all_indices)

        # Ensure they're different
        while other_idx == must_idx:
            other_idx = np.random.choice(all_indices)

        # Randomly assign which position gets the must_include city
        if np.random.random() < 0.5:
            all_pairs_i.append(must_idx)
            all_pairs_j.append(other_idx)
        else:
            all_pairs_i.append(other_idx)
            all_pairs_j.append(must_idx)

    return np.array(all_pairs_i), np.array(all_pairs_j)