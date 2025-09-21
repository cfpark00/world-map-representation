#!/usr/bin/env python3
"""
Angle dataset creation script.
Generates triples of cities and calculates the angle at the center city.
"""
import pandas as pd
import numpy as np
import sys
import yaml
import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast

sys.path.append('.')  # Add root to path
from src.utils import init_directory


def load_config(config_path):
    """Load and validate YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault('seed', 42)
    config.setdefault('n_train', 100000)
    config.setdefault('n_val', 128)
    config.setdefault('n_test', 10000)

    if 'pair_generation' not in config:
        config['pair_generation'] = {'strategy': 'all_pairs'}

    return config


def apply_group_definitions(df, group_definitions):
    """Apply group definitions to create group labels."""
    # Initialize all cities with 'unassigned' group
    df['group'] = 'unassigned'

    for group_name, group_def in group_definitions.items():
        mask = create_group_mask(df, group_def)
        df.loc[mask, 'group'] = group_name
        print(f"Group '{group_name}': {mask.sum():,} cities")

    return df


def create_group_mask(df, group_def):
    """Create a boolean mask for cities matching group definition."""
    mask = pd.Series([True] * len(df), index=df.index)

    if 'city_ids' in group_def:
        city_ids = set(group_def['city_ids'])
        city_mask = df['city_id'].isin(city_ids)
        mask = mask & city_mask

    if 'city_names' in group_def:
        names = group_def['city_names']
        if isinstance(names, list):
            name_mask = df['asciiname'].isin(names)
        else:
            name_mask = df['asciiname'].str.contains(names, na=False)
        mask = mask & name_mask

    if 'country_codes' in group_def:
        cc_def = group_def['country_codes']
        if isinstance(cc_def, list):
            pattern = '^(' + '|'.join(cc_def) + ')$'
            cc_mask = df['country_code'].str.contains(pattern, na=False)
        else:
            cc_mask = df['country_code'].str.contains(cc_def, na=False)
        mask = mask & cc_mask

    if 'regions' in group_def:
        region_def = group_def['regions']
        if isinstance(region_def, list):
            pattern = '^(' + '|'.join(region_def) + ')$'
            region_mask = df['region'].str.contains(pattern, na=False)
        else:
            region_mask = df['region'].str.contains(region_def, na=False)
        mask = mask & region_mask

    if 'bounds' in group_def:
        bounds = group_def['bounds']
        if 'y' in bounds:
            y_min, y_max = bounds['y']
            mask = mask & (df['y'] >= y_min) & (df['y'] <= y_max)
        if 'x' in bounds:
            x_min, x_max = bounds['x']
            mask = mask & (df['x'] >= x_min) & (df['x'] <= x_max)

    return mask


def load_cities(csv_path, config):
    """Load cities from CSV and apply group definitions."""
    print(f"Loading cities from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'x' not in df.columns:
        raise ValueError("CSV must have 'x' and 'y' columns")

    if 'city_id' not in df.columns:
        if 'row_id' in df.columns:
            df['city_id'] = df['row_id']
        else:
            df['city_id'] = df.index

    if 'groups' in config:
        print("\nApplying group definitions...")
        df = apply_group_definitions(df, config['groups'])
    else:
        df['group'] = 'all'
        print(f"No groups defined, using single group 'all': {len(df):,} cities")

    required_cols = ['city_id', 'x', 'y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def calculate_angle(x1, y1, x2, y2, x3, y3):
    """
    Calculate the angle at city 2 (center) formed by cities 1-2-3.
    Returns angle in degrees (0-180).
    """
    # Vectors from center (city2) to city1 and city3
    v1_x = x1 - x2
    v1_y = y1 - y2
    v2_x = x3 - x2
    v2_y = y3 - y2

    # Calculate dot product and magnitudes
    dot_product = v1_x * v2_x + v1_y * v2_y
    mag1 = np.sqrt(v1_x**2 + v1_y**2)
    mag2 = np.sqrt(v2_x**2 + v2_y**2)

    # Handle zero magnitude cases (same cities)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_angle = dot_product / (mag1 * mag2)
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # Calculate angle in radians then convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

    # Handle NaN cases (when cities overlap)
    angle_deg = np.where(np.isnan(angle_deg), 0, angle_deg)

    return angle_deg


def generate_triples(df, config, n_triples):
    """Generate city triples based on configuration strategy."""
    np.random.seed(config['seed'])

    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        # Generate triples from all cities
        return generate_all_triples(df, n_triples)

    elif strategy == 'within_groups':
        # Generate triples only within specified groups
        groups = config['pair_generation'].get('groups', df['group'].unique())
        return generate_within_group_triples(df, groups, n_triples)

    elif strategy == 'between_groups':
        # For triples, interpret as mixed groups
        print("Note: 'between_groups' strategy interpreted as mixed groups for triples")
        return generate_all_triples(df, n_triples)

    elif strategy == 'mixed':
        # Mix different triple types
        mix_config = config['pair_generation']['mix']
        return generate_mixed_triples(df, mix_config, n_triples)

    elif strategy == 'must_include':
        # All triples must include at least one city from specified groups
        must_include_groups = config['pair_generation']['must_include_groups']
        return generate_must_include_triples(df, must_include_groups, n_triples)

    else:
        # Default to all triples
        return generate_all_triples(df, n_triples)


def generate_all_triples(df, n_triples):
    """Generate random triples from all cities."""
    n_cities = len(df)

    # Generate random indices for three cities
    indices_i = np.random.randint(0, n_cities, size=n_triples)
    indices_j = np.random.randint(0, n_cities, size=n_triples)
    indices_k = np.random.randint(0, n_cities, size=n_triples)

    # Filter out degenerate cases (where any two cities are the same)
    valid_mask = (indices_i != indices_j) & (indices_j != indices_k) & (indices_i != indices_k)

    indices_i = indices_i[valid_mask]
    indices_j = indices_j[valid_mask]
    indices_k = indices_k[valid_mask]

    # If we don't have enough valid triples, generate more
    while len(indices_i) < n_triples:
        n_needed = n_triples - len(indices_i)
        new_i = np.random.randint(0, n_cities, size=n_needed * 2)
        new_j = np.random.randint(0, n_cities, size=n_needed * 2)
        new_k = np.random.randint(0, n_cities, size=n_needed * 2)

        valid_mask = (new_i != new_j) & (new_j != new_k) & (new_i != new_k)
        new_i = new_i[valid_mask][:n_needed]
        new_j = new_j[valid_mask][:n_needed]
        new_k = new_k[valid_mask][:n_needed]

        indices_i = np.concatenate([indices_i, new_i])
        indices_j = np.concatenate([indices_j, new_j])
        indices_k = np.concatenate([indices_k, new_k])

    return indices_i[:n_triples], indices_j[:n_triples], indices_k[:n_triples]


def generate_within_group_triples(df, groups, n_triples):
    """Generate triples only within specified groups."""
    all_triples_i = []
    all_triples_j = []
    all_triples_k = []

    # Calculate triples per group proportional to group size cubed
    group_sizes = {g: len(df[df['group'] == g]) for g in groups}
    total_weight = sum(s**3 for s in group_sizes.values() if s >= 3)

    if total_weight == 0:
        raise ValueError("No valid triples can be generated within specified groups")

    for group in groups:
        group_df = df[df['group'] == group]
        group_indices = group_df.index.values
        n_group = len(group_indices)

        if n_group < 3:
            continue

        # Calculate this group's share of triples
        group_weight = n_group**3
        n_group_triples = int(n_triples * group_weight / total_weight)

        if n_group_triples == 0:
            continue

        # Generate triples within this group
        triples_i = np.random.choice(group_indices, size=n_group_triples, replace=True)
        triples_j = np.random.choice(group_indices, size=n_group_triples, replace=True)
        triples_k = np.random.choice(group_indices, size=n_group_triples, replace=True)

        # Filter out degenerate cases
        valid_mask = (triples_i != triples_j) & (triples_j != triples_k) & (triples_i != triples_k)

        all_triples_i.extend(triples_i[valid_mask])
        all_triples_j.extend(triples_j[valid_mask])
        all_triples_k.extend(triples_k[valid_mask])

    return np.array(all_triples_i), np.array(all_triples_j), np.array(all_triples_k)


def generate_mixed_triples(df, mix_config, n_triples):
    """Generate mixed triples with specified ratios."""
    all_triples_i = []
    all_triples_j = []
    all_triples_k = []

    for mix_item in mix_config:
        ratio = mix_item['ratio']
        n_mix_triples = int(n_triples * ratio)

        if mix_item['type'] == 'within_group':
            groups = mix_item['groups']
            triples_i, triples_j, triples_k = generate_within_group_triples(df, groups, n_mix_triples)

        elif mix_item['type'] == 'all':
            triples_i, triples_j, triples_k = generate_all_triples(df, n_mix_triples)

        all_triples_i.extend(triples_i)
        all_triples_j.extend(triples_j)
        all_triples_k.extend(triples_k)

    # Shuffle combined triples
    shuffle_idx = np.random.permutation(len(all_triples_i))
    return (np.array(all_triples_i)[shuffle_idx],
            np.array(all_triples_j)[shuffle_idx],
            np.array(all_triples_k)[shuffle_idx])


def generate_must_include_triples(df, must_include_groups, n_triples):
    """Generate triples where at least one city must be from specified groups."""
    must_include_mask = df['group'].isin(must_include_groups)
    must_include_indices = df[must_include_mask].index.values
    all_indices = df.index.values

    if len(must_include_indices) == 0:
        raise ValueError("No cities found in must_include groups")

    triples_i = []
    triples_j = []
    triples_k = []

    attempts = 0
    max_attempts = n_triples * 10

    while len(triples_i) < n_triples and attempts < max_attempts:
        attempts += 1

        # Simple: pick one from must_include, two from all cities
        must_idx = np.random.choice(must_include_indices)
        other_idxs = np.random.choice(all_indices, size=2, replace=False)

        # Randomly assign positions (shuffling which one is from Atlantis)
        positions = np.random.permutation([must_idx, other_idxs[0], other_idxs[1]])
        i, j, k = positions

        # Ensure no duplicates
        if i != j and j != k and i != k:
            triples_i.append(i)
            triples_j.append(j)
            triples_k.append(k)

    if len(triples_i) < n_triples:
        print(f"Warning: Could only generate {len(triples_i)} unique triples out of {n_triples} requested")

    return np.array(triples_i), np.array(triples_j), np.array(triples_k)


def create_dataset_dict(indices_i, indices_j, indices_k, df, tokenizer, config=None):
    """Create a dictionary suitable for HuggingFace Dataset."""
    # Get coordinates
    x1 = df.iloc[indices_i]['x'].values
    y1 = df.iloc[indices_i]['y'].values
    x2 = df.iloc[indices_j]['x'].values  # Center city
    y2 = df.iloc[indices_j]['y'].values
    x3 = df.iloc[indices_k]['x'].values
    y3 = df.iloc[indices_k]['y'].values

    # Calculate angles at center city
    angles = calculate_angle(x1, y1, x2, y2, x3, y3)
    angles = np.round(angles).astype(int)

    # Get city IDs
    city1_ids = df.iloc[indices_i]['city_id'].values.astype(int)
    city2_ids = df.iloc[indices_j]['city_id'].values.astype(int)  # Center
    city3_ids = df.iloc[indices_k]['city_id'].values.astype(int)

    # Check for padding configuration
    use_padding = config.get('leading_zeros', False) if config else False
    if use_padding:
        if 'n_id_digits' not in config:
            raise ValueError("When leading_zeros=true, n_id_digits must be specified in config")
        n_digits = config['n_id_digits']
        max_expressible = 10**n_digits - 1

        # Validate city IDs
        max_id = max(city1_ids.max(), city2_ids.max(), city3_ids.max())
        if max_id > max_expressible:
            raise ValueError(f"City ID {max_id} exceeds maximum expressible with {n_digits} digits")

    # Create text format
    text_list = []
    task_type_list = []
    token_lengths = []
    loss_mask_list = []

    for c1, c2, c3, angle in tqdm(zip(city1_ids, city2_ids, city3_ids, angles),
                                   total=len(city1_ids),
                                   desc="Formatting samples",
                                   leave=False):
        # Format with padding if configured
        # Note: c2 is the center city where angle is measured
        if use_padding:
            c1_str = str(c1).zfill(n_digits)
            c2_str = str(c2).zfill(n_digits)
            c3_str = str(c3).zfill(n_digits)
            angle_str = f"angle(c_{c1_str},c_{c2_str},c_{c3_str})={angle}"
        else:
            angle_str = f"angle(c_{c1},c_{c2},c_{c3})={angle}"

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(angle_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("angle")

        # Tokenize for length and mask
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_lengths.append(len(tokens))

        # Create loss mask
        equals_token_id = tokenizer.encode('=', add_special_tokens=False)[0]

        mask = []
        found_equals = False
        for token_id in tokens:
            if not found_equals:
                mask.append('0')
                if token_id == equals_token_id:
                    found_equals = True
            else:
                mask.append('1')

        loss_mask_list.append(''.join(mask))

    return {
        'text': text_list,
        'task_type': task_type_list,
        'token_lengths': token_lengths,
        'loss_mask': loss_mask_list
    }


def main():
    parser = argparse.ArgumentParser(description='Create angle dataset')
    parser.add_argument('config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory if it exists')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config_path}")
    config = load_config(args.config_path)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if 'cities_csv' not in config:
        raise ValueError("FATAL: 'cities_csv' is required in config")

    # Load tokenizer
    tokenizer_path = config.get('tokenizer_path', 'data/tokenizers/default_tokenizer')
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load cities
    df = load_cities(config['cities_csv'], config)
    print(f"Loaded {len(df):,} cities total")

    # Display group statistics
    print("\nCity groups:")
    for group in df['group'].unique():
        count = len(df[df['group'] == group])
        print(f"  {group}: {count:,} cities")

    # Generate triples for each split
    n_train = config['n_train']
    n_val = config['n_val']
    n_test = config['n_test']

    print(f"\nGenerating triples (seed={config['seed']})...")
    print(f"  Train: {n_train:,}")
    print(f"  Val: {n_val:,}")
    print(f"  Test: {n_test:,}")

    # Generate all triples at once and split
    n_total = n_train + n_val + n_test
    all_i, all_j, all_k = generate_triples(df, config, n_total)

    # Apply random shuffling to each triple for symmetry
    print("Applying random triple shuffling for symmetry...")
    for idx in range(len(all_i)):
        # Randomly shuffle the three cities in each triple
        triple = [all_i[idx], all_j[idx], all_k[idx]]
        np.random.shuffle(triple)
        all_i[idx], all_j[idx], all_k[idx] = triple

    # Split into train/val/test
    train_i = all_i[:n_train]
    train_j = all_j[:n_train]
    train_k = all_k[:n_train]

    val_i = all_i[n_train:n_train+n_val]
    val_j = all_j[n_train:n_train+n_val]
    val_k = all_k[n_train:n_train+n_val]

    test_i = all_i[n_train+n_val:]
    test_j = all_j[n_train+n_val:]
    test_k = all_k[n_train+n_val:]

    # Create datasets
    print("\nCreating train dataset...")
    train_data = create_dataset_dict(train_i, train_j, train_k, df, tokenizer, config)
    train_dataset = Dataset.from_dict(train_data)

    print("Creating validation dataset...")
    val_data = create_dataset_dict(val_i, val_j, val_k, df, tokenizer, config)
    val_dataset = Dataset.from_dict(val_data)

    print("Creating test dataset...")
    test_data = create_dataset_dict(test_i, test_j, test_k, df, tokenizer, config)
    test_dataset = Dataset.from_dict(test_data)

    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # Initialize output directory
    output_path = init_directory(config['output_dir'], overwrite=args.overwrite)

    print(f"\nSaving dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    # Calculate statistics
    all_token_lengths = (train_data['token_lengths'] +
                        val_data['token_lengths'] +
                        test_data['token_lengths'])
    max_len = max(all_token_lengths)
    min_len = min(all_token_lengths)
    avg_len = sum(all_token_lengths) / len(all_token_lengths)

    # Save metadata
    metadata = {
        'csv_file': config['cities_csv'],
        'config_file': args.config_path,
        'config': config,
        'created': pd.Timestamp.now(tz='UTC').isoformat(),
        'total_cities': len(df),
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'seed': config['seed'],
        'max_len': max_len,
        'min_len': min_len,
        'avg_len': round(avg_len, 2)
    }

    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy config file
    config_copy_path = output_path / 'config.yaml'
    shutil.copy(args.config_path, config_copy_path)

    print(f"\nSaved files:")
    print(f"  - HuggingFace dataset files")
    print(f"  - metadata.json: Dataset metadata")
    print(f"  - config.yaml: Configuration used")

    # Display samples
    print("\nSample train rows:")
    for i in range(min(5, len(train_dataset))):
        print(f"  {train_dataset[i]['text']}")

    print("\nDataset created successfully!")
    print(f"Output: {output_path}")
    print(f"Train size: {len(train_dataset):,}")
    print(f"Val size: {len(val_dataset):,}")
    print(f"Test size: {len(test_dataset):,}")


if __name__ == "__main__":
    main()