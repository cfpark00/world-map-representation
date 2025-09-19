#!/usr/bin/env python3
"""
Center of mass dataset creation script.
Finds the city closest to the center of mass of a set of cities.
Can search within the given cities (in=TRUE) or all cities (in=FALSE).
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

    # Center-specific parameters
    config.setdefault('min_n_cities', 1)
    config.setdefault('max_n_cities', 5)
    config.setdefault('in_true_ratio', 0.5)  # 50% in=TRUE, 50% in=FALSE

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
    print(f"Loaded {len(df):,} cities total")

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
    """Get cities eligible for center calculation based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        return df
    elif strategy == 'within_groups':
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]
    elif strategy == 'must_include':
        # For must_include, return all cities - we'll handle the constraint in generate_center_samples
        return df
    else:
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for center calculation")
        return df


def find_closest_city(center_x, center_y, candidate_cities):
    """Find the city closest to the given center point.
    In case of ties, returns the first city in the list."""
    # Calculate distances to all candidate cities
    distances = euclidean_distance(
        np.array([center_x]),
        np.array([center_y]),
        candidate_cities['x'].values,
        candidate_cities['y'].values
    ).squeeze()

    # Ensure distances is at least 1D (handles case when only 1 candidate city)
    distances = np.atleast_1d(distances)

    # Find the minimum distance
    min_distance = np.min(distances)

    # Find ALL cities with the minimum distance (to handle ties)
    min_indices = np.where(np.abs(distances - min_distance) < 1e-10)[0]

    # Return the FIRST city among those with minimum distance
    # This ensures deterministic behavior and breaks symmetry
    return candidate_cities.iloc[min_indices[0]]


def generate_center_samples(df, config, n_samples):
    """Generate center of mass samples with controlled in=TRUE/FALSE ratio."""
    np.random.seed(config['seed'])

    eligible_df = get_eligible_cities(df, config)
    n_cities = len(eligible_df)

    min_n = config['min_n_cities']
    max_n = config['max_n_cities']
    in_true_ratio = config['in_true_ratio']

    if n_cities < max_n:
        raise ValueError(f"Need at least {max_n} cities, but only have {n_cities}")

    n_in_true = int(n_samples * in_true_ratio)
    n_in_false = n_samples - n_in_true

    samples = []

    pbar = tqdm(total=n_samples, desc="Generating center samples")

    for i in range(n_samples):
        # Random number of cities
        n_selected = np.random.randint(min_n, max_n + 1)

        # Select random cities
        indices = np.random.choice(n_cities, size=n_selected, replace=False)
        selected_cities = eligible_df.iloc[indices]

        # Calculate center of mass
        center_x = selected_cities['x'].mean()
        center_y = selected_cities['y'].mean()

        # Decide if in=TRUE or in=FALSE
        if i < n_in_true:
            in_constraint = True
            # Find closest city within the selected cities
            closest_city = find_closest_city(center_x, center_y, selected_cities)
        else:
            in_constraint = False
            # Find closest city from ALL eligible cities
            closest_city = find_closest_city(center_x, center_y, eligible_df)

        sample = {
            'selected_cities': selected_cities,
            'in_constraint': in_constraint,
            'closest_city': closest_city,
            'center_x': center_x,
            'center_y': center_y
        }

        samples.append(sample)
        pbar.update(1)

    pbar.close()

    # Shuffle to mix in=TRUE and in=FALSE samples
    np.random.shuffle(samples)

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
        selected_city_ids = sample['selected_cities']['city_id'].values.astype(int)
        in_constraint = sample['in_constraint']
        closest_city_id = int(sample['closest_city']['city_id'])

        # Format with padding if configured
        if use_padding:
            selected_strs = [str(cid).zfill(n_digits) for cid in selected_city_ids]
            closest_str = str(closest_city_id).zfill(n_digits)
            cities_part = ','.join(f"c_{s}" for s in selected_strs)
        else:
            cities_part = ','.join(f"c_{cid}" for cid in selected_city_ids)
            closest_str = str(closest_city_id)

        # Build the string with in=TRUE or in=FALSE
        if in_constraint:
            center_str = f"center({cities_part};in=TRUE)=c_{closest_str}"
        else:
            center_str = f"center({cities_part};in=FALSE)=c_{closest_str}"

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(center_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("center")

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
    parser = argparse.ArgumentParser(description='Create center of mass dataset')
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

    # Display parameters
    print(f"\nCenter of mass parameters:")
    print(f"  Number of cities range: [{config['min_n_cities']}, {config['max_n_cities']}]")
    print(f"  in=TRUE/FALSE ratio: {config['in_true_ratio']:.1%} in=TRUE")

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
    train_samples = generate_center_samples(df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("Generating validation samples...")
    val_samples = generate_center_samples(df, config, n_val)

    config['seed'] += 1
    print("Generating test samples...")
    test_samples = generate_center_samples(df, config, n_test)

    # Reset seed
    config['seed'] -= 2

    # Count in=TRUE/FALSE distribution
    train_in_true = sum(1 for s in train_samples if s['in_constraint'])
    val_in_true = sum(1 for s in val_samples if s['in_constraint'])
    test_in_true = sum(1 for s in test_samples if s['in_constraint'])

    print(f"\nActual in=TRUE ratios:")
    print(f"  Train: {train_in_true}/{len(train_samples)} ({train_in_true/len(train_samples):.1%})")
    print(f"  Val: {val_in_true}/{len(val_samples)} ({val_in_true/len(val_samples):.1%})")
    print(f"  Test: {test_in_true}/{len(test_samples)} ({test_in_true/len(test_samples):.1%})")

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
        'min_n_cities': config['min_n_cities'],
        'max_n_cities': config['max_n_cities'],
        'in_true_ratio_target': config['in_true_ratio'],
        'in_true_ratio_actual': {
            'train': train_in_true / len(train_samples),
            'val': val_in_true / len(val_samples),
            'test': test_in_true / len(test_samples)
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