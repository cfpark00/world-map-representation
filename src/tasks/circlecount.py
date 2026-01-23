#!/usr/bin/env python3
"""
Circle count dataset creation script.
Counts the number of cities within a given radius from a center city.
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

    # Circle count specific parameters
    config.setdefault('min_radius', 5)
    config.setdefault('max_radius', 200)

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
    """Get cities eligible for circle count based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        return df
    elif strategy == 'within_groups':
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]
    else:
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for circle count")
        return df


def count_cities_in_circle(center_city, all_cities, radius):
    """Count cities within given radius from center city."""
    # Calculate distances to all cities
    distances = euclidean_distance(
        np.array([center_city['x']]),
        np.array([center_city['y']]),
        all_cities['x'].values,
        all_cities['y'].values
    ).squeeze()

    # Count cities within radius (excluding the center city itself)
    # Find center city in all_cities to exclude it
    center_mask = (all_cities['city_id'] == center_city['city_id']).values
    within_radius = np.sum((distances <= radius) & ~center_mask)

    return within_radius


def generate_circlecount_samples(df, config, n_samples):
    """Generate circle count samples with random cities and radii."""
    np.random.seed(config['seed'])

    eligible_df = get_eligible_cities(df, config)
    n_cities = len(eligible_df)

    min_radius = config['min_radius']
    max_radius = config['max_radius']

    if n_cities < 1:
        raise ValueError("Need at least 1 city")

    samples = []

    for _ in tqdm(range(n_samples), desc="Generating circle count samples"):
        # Random city
        city_idx = np.random.choice(n_cities)
        center_city = eligible_df.iloc[city_idx]

        # Random radius
        radius = np.random.randint(min_radius, max_radius + 1)

        # Count cities within radius
        # Use all eligible cities for counting (not just world cities)
        count = count_cities_in_circle(center_city, eligible_df, radius)

        sample = {
            'center_city': center_city,
            'radius': radius,
            'count': count
        }

        samples.append(sample)

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
        center_city_id = int(sample['center_city']['city_id'])
        radius = sample['radius']
        count = sample['count']

        # Format with padding for city ID if configured
        # But count and radius are NEVER padded
        if use_padding:
            city_str = str(center_city_id).zfill(n_digits)
            circlecount_str = f"circlecount(c_{city_str},r={radius})={count}"
        else:
            circlecount_str = f"circlecount(c_{center_city_id},r={radius})={count}"

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(circlecount_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("circlecount")

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
    parser = argparse.ArgumentParser(description='Create circle count dataset')
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

    # Display parameters
    print(f"\nCircle count parameters:")
    print(f"  Radius range: [{config['min_radius']}, {config['max_radius']}]")

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
    train_samples = generate_circlecount_samples(df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("Generating validation samples...")
    val_samples = generate_circlecount_samples(df, config, n_val)

    config['seed'] += 1
    print("Generating test samples...")
    test_samples = generate_circlecount_samples(df, config, n_test)

    # Reset seed
    config['seed'] -= 2

    # Calculate count statistics
    train_counts = [s['count'] for s in train_samples]
    val_counts = [s['count'] for s in val_samples]
    test_counts = [s['count'] for s in test_samples]

    print(f"\nCount statistics:")
    print(f"  Train: min={min(train_counts)}, max={max(train_counts)}, mean={np.mean(train_counts):.1f}")
    print(f"  Val: min={min(val_counts)}, max={max(val_counts)}, mean={np.mean(val_counts):.1f}")
    print(f"  Test: min={min(test_counts)}, max={max(test_counts)}, mean={np.mean(test_counts):.1f}")

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
        'min_radius': config['min_radius'],
        'max_radius': config['max_radius'],
        'count_stats': {
            'train': {'min': int(min(train_counts)), 'max': int(max(train_counts)), 'mean': float(np.mean(train_counts))},
            'val': {'min': int(min(val_counts)), 'max': int(max(val_counts)), 'mean': float(np.mean(val_counts))},
            'test': {'min': int(min(test_counts)), 'max': int(max(test_counts)), 'mean': float(np.mean(test_counts))}
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


# ============================================================================
# Evaluation Metric
# ============================================================================

import re


class CircleCountMetric:
    """Metric for circle count tasks: absolute error in count."""

    def __init__(self):
        self.failure_value = 1000.0
        self.display_name = "Count Error"

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        # Match )=COUNT pattern to get the actual count, not r=radius
        true_text = (prompt + true_completion).replace(' ', '')
        gen_text = generated.replace(' ', '')

        true_match = re.search(r'\)=(\d+)', true_text)
        gen_match = re.search(r'\)=(\d+)', gen_text)

        if true_match and gen_match:
            true_count = int(true_match.group(1))
            gen_count = int(gen_match.group(1))
            return abs(gen_count - true_count)
        else:
            return self.failure_value

    def format_for_print(self, value: float) -> str:
        return f"{value:.2f}"


# Singleton instance for import
METRIC = CircleCountMetric()


if __name__ == "__main__":
    main()