#!/usr/bin/env python3
"""
Random ring dataset creation script.
Randomly samples n cities from an annulus (ring) defined by inner radius r and outer radius R.
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

    # Randring-specific parameters
    config.setdefault('min_r', 10)    # Min inner radius
    config.setdefault('max_r', 450)   # Max inner radius
    config.setdefault('min_R', 50)    # Min outer radius
    config.setdefault('max_R', 500)   # Max outer radius
    config.setdefault('min_n', 1)     # Min number of cities to sample
    config.setdefault('max_n', 10)    # Max number of cities to sample

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
            raise ValueError(f"Cities DataFrame must have '{col}' column")

    return df


def get_eligible_cities(df, config):
    """Get cities eligible for randring based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        return df
    elif strategy == 'within_groups':
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]
    else:
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for randring")
        return df


def sample_cities_from_ring(center_city, all_cities, inner_radius, outer_radius, n_samples):
    """
    Sample n cities randomly from the ring (annulus) around center_city.
    If fewer than n cities exist in the ring, return all of them.
    """
    # Calculate distances to all cities
    distances = euclidean_distance(
        np.array([center_city['x']]),
        np.array([center_city['y']]),
        all_cities['x'].values,
        all_cities['y'].values
    ).squeeze()

    # Ensure distances is at least 1D
    distances = np.atleast_1d(distances)

    # Find cities in the ring (excluding center city itself)
    center_mask = all_cities['city_id'] == center_city['city_id']
    in_ring = (distances >= inner_radius) & (distances <= outer_radius) & ~center_mask

    ring_cities = all_cities[in_ring]
    n_available = len(ring_cities)

    if n_available == 0:
        return []

    # Sample min(n_samples, n_available) cities
    n_to_sample = min(n_samples, n_available)

    if n_to_sample == n_available:
        # Return all cities in ring
        sampled = ring_cities
    else:
        # Randomly sample from ring
        sampled = ring_cities.sample(n=n_to_sample, replace=False)

    return sampled['city_id'].tolist()


def generate_randring_samples(df, config, n_samples):
    """Generate randring samples with random parameters."""
    np.random.seed(config['seed'])

    eligible_df = get_eligible_cities(df, config)
    n_cities = len(eligible_df)

    min_r = config['min_r']
    max_r = config['max_r']
    min_R = config['min_R']
    max_R = config['max_R']
    min_n = config['min_n']
    max_n = config['max_n']

    if n_cities < 2:  # Need at least center + 1 other city
        raise ValueError(f"Need at least 2 cities, but only have {n_cities}")

    samples = []

    pbar = tqdm(total=n_samples, desc="Generating randring samples")

    for i in range(n_samples):
        # Select random center city
        center_idx = np.random.choice(n_cities)
        center_city = eligible_df.iloc[center_idx]

        # Sample inner radius r
        r = np.random.randint(min_r, max_r + 1)

        # Sample outer radius R such that R > r
        # Ensure min_R_actual >= r + 1 to have R > r
        min_R_actual = max(min_R, r + 1)
        if min_R_actual > max_R:
            # Can't satisfy R > r with current constraints, use R = r + 1
            R = r + 1
        else:
            R = np.random.randint(min_R_actual, max_R + 1)

        # Sample number of cities to return
        n = np.random.randint(min_n, max_n + 1)

        # Sample cities from the ring
        sampled_cities = sample_cities_from_ring(center_city, eligible_df, r, R, n)

        samples.append({
            'center_city_id': center_city['city_id'],
            'inner_radius': r,
            'outer_radius': R,
            'n_requested': n,
            'sampled_cities': sampled_cities,
            'n_actual': len(sampled_cities)
        })

        pbar.update(1)

    pbar.close()

    return samples


def create_dataset_dict(samples, tokenizer, config):
    """Convert samples to dataset dictionary with text."""
    texts = []

    # Check if padding is enabled in config (default to no padding)
    use_padding = config.get('leading_zeros', False)
    n_digits = config.get('n_id_digits', 4)

    for sample in tqdm(samples, desc="Creating text samples"):
        center_city_id = sample['center_city_id']
        r = sample['inner_radius']
        R = sample['outer_radius']
        n = sample['n_requested']
        sampled_cities = sample['sampled_cities']

        # Format city IDs - pad if use_padding is True
        if use_padding:
            center_str = str(center_city_id).zfill(n_digits)
            result_str = ','.join(['c_' + str(cid).zfill(n_digits) for cid in sampled_cities])
        else:
            center_str = str(center_city_id)
            result_str = ','.join(['c_' + str(cid) for cid in sampled_cities])

        # Build the formatted string: randring(c_ID,r=MIN,R=MAX,n=NUM)=c_ID1,c_ID2,...
        randring_str = f"randring(c_{center_str},r={r},R={R},n={n})={result_str}"

        # Character-level tokenization with spaces
        spaced_str = ' '.join(randring_str)

        # Add <bos> and <eos> tokens
        final_text = f"<bos> {spaced_str} <eos>"
        texts.append(final_text)

    # Tokenize all texts - don't add special tokens since they're already in the text
    encodings = tokenizer(texts, truncation=False, padding=False, add_special_tokens=False)

    # Store task type for multi-task training
    task_types = ['randring'] * len(texts)

    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'text': texts,
        'task_type': task_types
    }


def main():
    parser = argparse.ArgumentParser(description='Create randring dataset')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')

    args = parser.parse_args()

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
    print(f"\nRandring parameters:")
    print(f"  Inner radius range: [{config['min_r']}, {config['max_r']}]")
    print(f"  Outer radius range: [{config['min_R']}, {config['max_R']}]")
    print(f"  Sample count range: [{config['min_n']}, {config['max_n']}]")

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
    train_samples = generate_randring_samples(df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("Generating validation samples...")
    val_samples = generate_randring_samples(df, config, n_val)

    config['seed'] += 1
    print("Generating test samples...")
    test_samples = generate_randring_samples(df, config, n_test)

    # Reset seed
    config['seed'] -= 2

    # Calculate statistics
    train_actual = [s['n_actual'] for s in train_samples]
    val_actual = [s['n_actual'] for s in val_samples]
    test_actual = [s['n_actual'] for s in test_samples]

    print(f"\nActual sample counts:")
    print(f"  Train: mean={np.mean(train_actual):.2f}, min={min(train_actual)}, max={max(train_actual)}")
    print(f"  Val: mean={np.mean(val_actual):.2f}, min={min(val_actual)}, max={max(val_actual)}")
    print(f"  Test: mean={np.mean(test_actual):.2f}, min={min(test_actual)}, max={max(test_actual)}")

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

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # Calculate token statistics
    all_lens = [len(ids) for ids in train_data['input_ids']] + \
               [len(ids) for ids in val_data['input_ids']] + \
               [len(ids) for ids in test_data['input_ids']]
    max_len = max(all_lens)
    min_len = min(all_lens)
    avg_len = np.mean(all_lens)

    # Initialize output directory
    output_path = init_directory(config['output_dir'], overwrite=args.overwrite)

    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    # Save metadata
    metadata = {
        'dataset_type': 'randring',
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'config': config,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'max_tokens': max_len,
        'min_tokens': min_len,
        'avg_tokens': round(avg_len, 2),
        'min_r': config['min_r'],
        'max_r': config['max_r'],
        'min_R': config['min_R'],
        'max_R': config['max_R'],
        'min_n': config['min_n'],
        'max_n': config['max_n'],
        'sample_stats': {
            'train': {'mean': float(np.mean(train_actual)), 'min': int(min(train_actual)), 'max': int(max(train_actual))},
            'val': {'mean': float(np.mean(val_actual)), 'min': int(min(val_actual)), 'max': int(max(val_actual))},
            'test': {'mean': float(np.mean(test_actual)), 'min': int(min(test_actual)), 'max': int(max(test_actual))}
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