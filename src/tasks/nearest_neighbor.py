#!/usr/bin/env python3
"""
Nearest neighbor dataset creation script.
Generates queries for k nearest neighbors to a given city.
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

    # Nearest neighbor specific defaults
    config.setdefault('min_k', 1)
    config.setdefault('max_k', 5)

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
    """Get cities eligible for nearest neighbor queries based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        # Use all cities
        return df

    elif strategy == 'within_groups':
        # Use only specified groups
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]

    elif strategy == 'must_include':
        # For nearest neighbor, we interpret this as: query cities must be from specified groups
        must_include_groups = config['pair_generation']['must_include_groups']
        mask = df['group'].isin(must_include_groups)
        return df[mask], df  # Return query cities and pool cities separately

    else:
        # Default to all cities
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for nearest neighbors")
        return df


def generate_nearest_neighbor_queries(df, config, n_queries):
    """Generate nearest neighbor queries with varying k values."""
    np.random.seed(config['seed'])

    min_k = config['min_k']
    max_k = config['max_k']

    # Handle different strategies
    strategy = config['pair_generation'].get('strategy', 'all_pairs')

    if strategy == 'must_include':
        # Query cities must be from specified groups, but can find neighbors from all cities
        must_include_groups = config['pair_generation']['must_include_groups']
        query_mask = df['group'].isin(must_include_groups)
        query_df = df[query_mask]
        pool_df = df  # All cities can be neighbors
    else:
        # Normal case - both queries and neighbors from same pool
        query_df = get_eligible_cities(df, config)
        pool_df = query_df

    if len(query_df) < 1:
        raise ValueError("No eligible cities for queries")

    # We need at least k+1 cities in the pool (excluding the query city itself)
    if len(pool_df) <= max_k:
        raise ValueError(f"Need at least {max_k + 1} cities in pool, but only have {len(pool_df)}")

    queries = []
    for _ in tqdm(range(n_queries), desc="Generating queries"):
        # Random query city from eligible cities
        query_idx = np.random.choice(len(query_df))
        query_city = query_df.iloc[query_idx]

        # Random k value
        k = np.random.randint(min_k, max_k + 1)

        # Calculate distances to all cities in the pool
        distances = euclidean_distance(
            np.array([query_city['x']]),
            np.array([query_city['y']]),
            pool_df['x'].values,
            pool_df['y'].values
        ).squeeze()

        # Create indices array for pool cities
        pool_indices = np.arange(len(pool_df))

        # Exclude the query city itself if it's in the pool
        if query_city['city_id'] in pool_df['city_id'].values:
            query_pool_idx = pool_df[pool_df['city_id'] == query_city['city_id']].index[0]
            valid_mask = pool_indices != query_pool_idx
            valid_distances = distances[valid_mask]
            valid_indices = pool_indices[valid_mask]
        else:
            valid_distances = distances
            valid_indices = pool_indices

        # Find k nearest neighbors
        nearest_indices = valid_indices[np.argsort(valid_distances)[:k]]
        nearest_cities = pool_df.iloc[nearest_indices]

        queries.append({
            'query_city': query_city,
            'k': k,
            'neighbors': nearest_cities
        })

    return queries


def create_dataset_dict(queries, tokenizer, config):
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

    for query_data in tqdm(queries, desc="Formatting samples", leave=False):
        query_city = query_data['query_city']
        k = query_data['k']
        neighbors = query_data['neighbors']

        query_id = int(query_city['city_id'])
        neighbor_ids = neighbors['city_id'].values.astype(int)

        # Format with padding if configured
        if use_padding:
            query_str = str(query_id).zfill(n_digits)
            neighbor_strs = [str(nid).zfill(n_digits) for nid in neighbor_ids]
            nn_str = f"nearest(c_{query_str},{k})=" + ','.join(f"c_{n}" for n in neighbor_strs)
        else:
            nn_str = f"nearest(c_{query_id},{k})=" + ','.join(f"c_{nid}" for nid in neighbor_ids)

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(nn_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("nearest_neighbor")

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
    parser = argparse.ArgumentParser(description='Create nearest neighbor dataset')
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

    # Display nearest neighbor parameters
    print(f"\nNearest neighbor parameters:")
    print(f"  k range: [{config['min_k']}, {config['max_k']}]")

    # Generate queries for each split
    n_train = config['n_train']
    n_val = config['n_val']
    n_test = config['n_test']

    print(f"\nGenerating queries (seed={config['seed']})...")
    print(f"  Train: {n_train:,}")
    print(f"  Val: {n_val:,}")
    print(f"  Test: {n_test:,}")

    # Generate queries for each split
    # Special handling for must_include strategy to avoid train/test overlap
    if config['pair_generation'].get('strategy') == 'must_include':
        print("\nUsing deterministic generation for must_include strategy...")

        # Generate ALL possible queries first
        must_include_groups = config['pair_generation']['must_include_groups']
        query_mask = df['group'].isin(must_include_groups)
        query_df = df[query_mask]

        # Create all possible (city, k) combinations
        all_queries_list = []
        for _, query_city in query_df.iterrows():
            for k in range(config['min_k'], config['max_k'] + 1):
                all_queries_list.append((query_city, k))

        print(f"Total possible unique queries: {len(all_queries_list)}")

        # Shuffle with seed for reproducibility
        np.random.seed(config['seed'])
        np.random.shuffle(all_queries_list)

        # Split into train/val/test
        # Val can overlap with train, but test must be completely separate
        if n_train + n_test > len(all_queries_list):
            raise ValueError(f"Requested {n_train} train + {n_test} test = {n_train + n_test} samples, "
                           f"but only {len(all_queries_list)} unique queries possible")

        # Take first n_train for train
        train_queries_list = all_queries_list[:n_train]

        # Take last n_test for test (guaranteed no overlap with train)
        test_queries_list = all_queries_list[-n_test:]

        # For val, sample from train portion (allowing overlap)
        val_indices = np.random.choice(n_train, size=min(n_val, n_train), replace=False)
        val_queries_list = [all_queries_list[i] for i in val_indices]

        # Now generate the actual nearest neighbor results for each split
        print("\nGenerating train queries...")
        train_queries = []
        for query_city, k in tqdm(train_queries_list, desc="Processing train queries", leave=False):
            # Calculate distances to all cities
            distances = euclidean_distance(
                np.array([query_city['x']]),
                np.array([query_city['y']]),
                df['x'].values,
                df['y'].values
            ).squeeze()

            # Exclude the query city itself
            valid_mask = df['city_id'] != query_city['city_id']
            valid_distances = distances[valid_mask]
            valid_df = df[valid_mask]

            # Find k nearest neighbors
            nearest_indices = np.argsort(valid_distances)[:k]
            nearest_cities = valid_df.iloc[nearest_indices]

            train_queries.append({
                'query_city': query_city,
                'k': k,
                'neighbors': nearest_cities
            })

        print("Generating validation queries...")
        val_queries = []
        for query_city, k in tqdm(val_queries_list, desc="Processing val queries", leave=False):
            distances = euclidean_distance(
                np.array([query_city['x']]),
                np.array([query_city['y']]),
                df['x'].values,
                df['y'].values
            ).squeeze()

            valid_mask = df['city_id'] != query_city['city_id']
            valid_distances = distances[valid_mask]
            valid_df = df[valid_mask]

            nearest_indices = np.argsort(valid_distances)[:k]
            nearest_cities = valid_df.iloc[nearest_indices]

            val_queries.append({
                'query_city': query_city,
                'k': k,
                'neighbors': nearest_cities
            })

        print("Generating test queries...")
        test_queries = []
        for query_city, k in tqdm(test_queries_list, desc="Processing test queries", leave=False):
            distances = euclidean_distance(
                np.array([query_city['x']]),
                np.array([query_city['y']]),
                df['x'].values,
                df['y'].values
            ).squeeze()

            valid_mask = df['city_id'] != query_city['city_id']
            valid_distances = distances[valid_mask]
            valid_df = df[valid_mask]

            nearest_indices = np.argsort(valid_distances)[:k]
            nearest_cities = valid_df.iloc[nearest_indices]

            test_queries.append({
                'query_city': query_city,
                'k': k,
                'neighbors': nearest_cities
            })
    else:
        # Original stochastic generation for other strategies
        print("\nGenerating train queries...")
        train_queries = generate_nearest_neighbor_queries(df, config, n_train)

        # Increment seed for different splits
        config['seed'] += 1
        print("Generating validation queries...")
        val_queries = generate_nearest_neighbor_queries(df, config, n_val)

        config['seed'] += 1
        print("Generating test queries...")
        test_queries = generate_nearest_neighbor_queries(df, config, n_test)

        # Reset seed
        config['seed'] -= 2

    # Create datasets
    print("\nCreating train dataset...")
    train_data = create_dataset_dict(train_queries, tokenizer, config)
    train_dataset = Dataset.from_dict(train_data)

    print("Creating validation dataset...")
    val_data = create_dataset_dict(val_queries, tokenizer, config)
    val_dataset = Dataset.from_dict(val_data)

    print("Creating test dataset...")
    test_data = create_dataset_dict(test_queries, tokenizer, config)
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
        'min_k': config['min_k'],
        'max_k': config['max_k']
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
from typing import Set


class NearestNeighborMetric:
    """Metric for nearest neighbor tasks: Jaccard similarity."""

    def __init__(self):
        self.failure_value = 0.0
        self.display_name = "Jaccard Similarity"

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        # Extract city IDs from completions
        true_cities = set(re.findall(r'c_\d+', true_completion.replace(' ', '')))

        # Extract answer after )= pattern ONLY
        gen_no_space = generated.replace(' ', '')
        pattern_match = re.search(r'\)=(.+)$', gen_no_space)
        if pattern_match:
            gen_completion = pattern_match.group(1)
            gen_cities = set(re.findall(r'c_\d+', gen_completion))
        else:
            gen_cities = set()

        expected_k = len(true_cities)

        if len(gen_cities) == expected_k and expected_k > 0:
            intersection = true_cities.intersection(gen_cities)
            union = true_cities.union(gen_cities)
            return len(intersection) / len(union) if union else 0.0
        else:
            return 0.0

    def format_for_print(self, value: float) -> str:
        return f"{value:.3f}"


# Singleton instance for import
METRIC = NearestNeighborMetric()


if __name__ == "__main__":
    main()