#!/usr/bin/env python3
"""
Perimeter dataset creation script.
Calculates the perimeter of a polygon formed by connecting cities in order.
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
from src.utils import euclidean_distance, init_directory


def load_config(config_path):
    """Load and validate YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault('seed', 42)
    config.setdefault('n_train', 100000)
    config.setdefault('n_val', 128)
    config.setdefault('n_test', 10000)

    # Perimeter specific parameters
    config.setdefault('min_n', 2)
    config.setdefault('max_n', 5)

    if config['min_n'] < 2:
        raise ValueError("min_n must be >= 2 for perimeter calculation")

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


def get_eligible_cities(df, config):
    """Get cities eligible for perimeter calculation based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        return df
    elif strategy == 'within_groups':
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]
    elif strategy == 'must_include':
        # For must_include, return all cities - we'll handle the constraint in generate_perimeter_samples
        return df
    else:
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for perimeter calculation")
        return df


def calculate_perimeter(cities):
    """
    Calculate perimeter by summing distances between consecutive cities.
    If n=2, returns distance between the two cities.
    If n>2, returns sum of distances connecting cities in order (including last to first).
    """
    n_cities = len(cities)

    if n_cities < 2:
        return 0

    total_distance = 0

    # Calculate distances between consecutive cities
    for i in range(n_cities):
        current = cities.iloc[i]
        # Connect to next city (wrapping around to first for last city)
        next_idx = (i + 1) % n_cities
        next_city = cities.iloc[next_idx]

        # Calculate distance
        distance = euclidean_distance(
            np.array([current['x']]),
            np.array([current['y']]),
            np.array([next_city['x']]),
            np.array([next_city['y']])
        ).item()

        total_distance += distance

        # For n=2, we only want the single distance (not doubled)
        if n_cities == 2 and i == 0:
            break

    # Round to integer
    return int(round(total_distance))


def generate_perimeter_samples(df, config, n_samples):
    """Generate perimeter calculation samples."""
    np.random.seed(config['seed'])

    eligible_df = get_eligible_cities(df, config)
    n_cities = len(eligible_df)

    min_n = config['min_n']
    max_n = config['max_n']

    if n_cities < max_n:
        raise ValueError(f"Need at least {max_n} cities, but only have {n_cities}")

    samples = []

    # Track perimeter statistics
    perimeter_stats = []

    for _ in tqdm(range(n_samples), desc="Generating perimeter samples"):
        # Random number of cities for this polygon
        n_polygon = np.random.randint(min_n, max_n + 1)

        # Select random cities (no duplicates)
        if config['pair_generation'].get('strategy') == 'must_include':
            # Ensure at least one city is from must_include groups
            must_include_groups = config['pair_generation']['must_include_groups']
            must_include_mask = eligible_df['group'].isin(must_include_groups)
            must_include_indices = eligible_df[must_include_mask].index.values
            other_indices = eligible_df[~must_include_mask].index.values

            # Pick one from must_include
            must_idx = np.random.choice(must_include_indices, size=1)

            # Pick rest from anywhere (can include more from must_include)
            all_indices = np.concatenate([must_include_indices, other_indices])
            # Remove the already selected must_idx
            available = all_indices[all_indices != must_idx[0]]

            # Need n_polygon - 1 more cities
            if n_polygon > 1:
                other_idx = np.random.choice(available, size=n_polygon-1, replace=False)
                selected_indices = np.concatenate([must_idx, other_idx])
            else:
                selected_indices = must_idx

            np.random.shuffle(selected_indices)  # Shuffle so must_include isn't always first
            selected_cities = eligible_df.loc[selected_indices]
        else:
            indices = np.random.choice(n_cities, size=n_polygon, replace=False)
            selected_cities = eligible_df.iloc[indices]

        # Calculate perimeter
        perimeter = calculate_perimeter(selected_cities)
        perimeter_stats.append(perimeter)

        sample = {
            'cities': selected_cities,
            'n_cities': n_polygon,
            'perimeter': perimeter
        }

        samples.append(sample)

    # Print statistics
    print(f"\nPerimeter statistics:")
    print(f"  Min: {min(perimeter_stats)}")
    print(f"  Max: {max(perimeter_stats)}")
    print(f"  Mean: {np.mean(perimeter_stats):.1f}")
    print(f"  Median: {np.median(perimeter_stats):.1f}")

    return samples


def create_dataset_dict(samples, tokenizer, config):
    """Create a dictionary suitable for HuggingFace Dataset."""
    text_list = []
    task_type_list = []
    token_lengths = []
    loss_mask_list = []

    use_padding = config.get('leading_zeros', False)

    if use_padding:
        if 'n_id_digits' not in config:
            raise ValueError("When leading_zeros=true, n_id_digits must be specified in config")
        n_digits = config['n_id_digits']

    for sample in tqdm(samples, desc="Formatting samples", leave=False):
        city_ids = sample['cities']['city_id'].values.astype(int)
        perimeter = sample['perimeter']

        # Format with padding for city IDs if configured
        if use_padding:
            city_strs = [str(cid).zfill(n_digits) for cid in city_ids]
            cities_part = ','.join(f"c_{s}" for s in city_strs)
        else:
            cities_part = ','.join(f"c_{cid}" for cid in city_ids)

        # Perimeter value is NOT padded
        perimeter_str = f"perimeter({cities_part})={perimeter}"

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(perimeter_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("perimeter")

        # Tokenize for length and mask
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_lengths.append(len(tokens))

        # Create loss mask - mask everything up to and including '='
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
    parser = argparse.ArgumentParser(description='Create perimeter dataset')
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

    # Assert min_n >= 2
    if config['min_n'] < 2:
        raise ValueError(f"min_n must be >= 2 for perimeter, got {config['min_n']}")

    # Load tokenizer
    tokenizer_path = config.get('tokenizer_path', 'data/tokenizers/default_tokenizer')
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load cities
    df = load_cities(config['cities_csv'], config)
    print(f"Loaded {len(df):,} cities total")

    # Display parameters
    print(f"\nPerimeter parameters:")
    print(f"  Number of cities range: [{config['min_n']}, {config['max_n']}]")

    # Generate samples for each split
    n_train = config['n_train']
    n_val = config['n_val']
    n_test = config['n_test']

    print(f"\nGenerating samples (seed={config['seed']})...")
    print(f"  Train: {n_train:,}")
    print(f"  Val: {n_val:,}")
    print(f"  Test: {n_test:,}")

    # Generate samples for each split
    print("\nGenerating train samples...")
    train_samples = generate_perimeter_samples(df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("\nGenerating validation samples...")
    val_samples = generate_perimeter_samples(df, config, n_val)

    config['seed'] += 1
    print("\nGenerating test samples...")
    test_samples = generate_perimeter_samples(df, config, n_test)

    # Reset seed
    config['seed'] -= 2

    # Calculate perimeter statistics
    train_perimeters = [s['perimeter'] for s in train_samples]
    val_perimeters = [s['perimeter'] for s in val_samples]
    test_perimeters = [s['perimeter'] for s in test_samples]

    print(f"\nPerimeter statistics by split:")
    print(f"  Train: min={min(train_perimeters)}, max={max(train_perimeters)}, mean={np.mean(train_perimeters):.1f}")
    print(f"  Val: min={min(val_perimeters)}, max={max(val_perimeters)}, mean={np.mean(val_perimeters):.1f}")
    print(f"  Test: min={min(test_perimeters)}, max={max(test_perimeters)}, mean={np.mean(test_perimeters):.1f}")

    # Create datasets
    print("\nCreating train dataset...")
    train_data = create_dataset_dict(train_samples, tokenizer, config)
    train_dataset = Dataset.from_dict(train_data)

    print("Creating validation dataset...")
    val_data = create_dataset_dict(val_samples, tokenizer, config)
    val_dataset = Dataset.from_dict(val_data)

    print("Creating test dataset...")
    test_data = create_dataset_dict(test_samples, tokenizer, config)
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
        'avg_len': round(avg_len, 2),
        'min_n': config['min_n'],
        'max_n': config['max_n'],
        'perimeter_stats': {
            'train': {'min': min(train_perimeters), 'max': max(train_perimeters), 'mean': np.mean(train_perimeters)},
            'val': {'min': min(val_perimeters), 'max': max(val_perimeters), 'mean': np.mean(val_perimeters)},
            'test': {'min': min(test_perimeters), 'max': max(test_perimeters), 'mean': np.mean(test_perimeters)}
        }
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