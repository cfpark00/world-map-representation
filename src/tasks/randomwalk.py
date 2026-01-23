#!/usr/bin/env python3
"""
Random walk dataset creation script.
Generates sequences of cities where each next city is within a maximum distance.
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

    # Random walk specific defaults - now using ranges
    config.setdefault('min_max_distance', 50)   # Min value for max_distance sampling
    config.setdefault('max_max_distance', 200)  # Max value for max_distance sampling
    config.setdefault('min_chain_length', 3)    # Min chain length
    config.setdefault('max_chain_length', 8)    # Max chain length

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
    """Get cities eligible for random walk generation based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        # Use all cities
        return df

    elif strategy == 'within_groups':
        # Use only specified groups
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]

    else:
        # For random walks, we interpret other strategies as "all cities"
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for random walks")
        return df


def generate_random_walk(df, config, max_distance, chain_length):
    """Generate a single random walk with specified parameters."""
    # Always allow revisiting cities for more flexibility
    # Start from a random city
    current_idx = np.random.choice(len(df))
    walk_indices = [current_idx]

    for _ in range(chain_length - 1):
        current_city = df.iloc[current_idx]

        # Calculate distances to all other cities
        # euclidean_distance expects arrays, so convert scalars to arrays
        distances = euclidean_distance(
            np.array([current_city['x']]),
            np.array([current_city['y']]),
            df['x'].values,
            df['y'].values
        )
        # Squeeze to get 1D array since we only have one source point
        distances = distances.squeeze()

        # Find eligible next cities (within max_distance)
        eligible_mask = distances <= max_distance
        eligible_mask[current_idx] = False  # Can't stay at same city

        eligible_indices = np.where(eligible_mask)[0]

        if len(eligible_indices) == 0:
            # No valid next city - terminate walk early
            break

        # Choose next city uniformly from eligible ones
        next_idx = np.random.choice(eligible_indices)
        walk_indices.append(next_idx)
        current_idx = next_idx

    return walk_indices


def generate_walks(df, config, n_walks):
    """Generate multiple random walks with varying parameters."""
    np.random.seed(config['seed'])

    # Sample parameters for each walk
    min_max_dist = config['min_max_distance']
    max_max_dist = config['max_max_distance']
    min_chain = config['min_chain_length']
    max_chain = config['max_chain_length']

    # Sample parameters for all walks upfront
    max_distances = np.random.randint(min_max_dist, max_max_dist + 1, size=n_walks * 2)  # Extra for retries
    chain_lengths = np.random.randint(min_chain, max_chain + 1, size=n_walks * 2)

    walks = []
    walk_params = []  # Store (max_distance, chain_length) for each successful walk
    max_attempts = n_walks * 10  # Allow retries for failed walks
    param_idx = 0

    pbar = tqdm(total=n_walks, desc="Generating random walks")

    while len(walks) < n_walks and param_idx < len(max_distances) and param_idx < max_attempts:
        max_dist = max_distances[param_idx]
        chain_len = chain_lengths[param_idx]

        walk = generate_random_walk(df, config, max_dist, chain_len)

        # Only keep walks that reached the target length
        if len(walk) == chain_len:
            walks.append(walk)
            walk_params.append((max_dist, chain_len))
            pbar.update(1)

        param_idx += 1

    pbar.close()

    if len(walks) < n_walks:
        print(f"Warning: Could only generate {len(walks)} valid walks out of {n_walks} requested")

    return walks, walk_params


def create_dataset_dict(walks, walk_params, df, tokenizer, config):
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
        max_expressible = 10**n_digits - 1

        # Validate city IDs
        all_city_ids = df['city_id'].values
        max_id = all_city_ids.max()
        if max_id > max_expressible:
            raise ValueError(f"City ID {max_id} exceeds maximum expressible with {n_digits} digits")

    for walk_indices, (max_distance, chain_length) in tqdm(zip(walks, walk_params),
                                                            total=len(walks),
                                                            desc="Formatting samples",
                                                            leave=False):
        city_ids = df.iloc[walk_indices]['city_id'].values.astype(int)

        # Format city chain
        if use_padding:
            city_strs = [f"c_{str(cid).zfill(n_digits)}" for cid in city_ids]
        else:
            city_strs = [f"c_{cid}" for cid in city_ids]

        # Build the formatted string: rw(max_dist, chain_len)=city1,city2,...
        chain_str = ','.join(city_strs)
        rw_str = f"rw({max_distance},{chain_length})={chain_str}"

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(rw_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("randomwalk")

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
    parser = argparse.ArgumentParser(description='Create random walk dataset')
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
    eligible_df = get_eligible_cities(df, config)
    print(f"Using {len(eligible_df):,} eligible cities for random walks")

    # Display walk parameters
    print(f"\nRandom walk parameters:")
    print(f"  Max distance range: [{config['min_max_distance']}, {config['max_max_distance']}]")
    print(f"  Chain length range: [{config['min_chain_length']}, {config['max_chain_length']}]")
    print(f"  Allow revisit: True (cities can be revisited)")

    # Generate walks for each split
    n_train = config['n_train']
    n_val = config['n_val']
    n_test = config['n_test']

    print(f"\nGenerating walks (seed={config['seed']})...")
    print(f"  Train: {n_train:,}")
    print(f"  Val: {n_val:,}")
    print(f"  Test: {n_test:,}")

    # Generate walks for each split
    print("\nGenerating train walks...")
    train_walks, train_params = generate_walks(eligible_df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("Generating validation walks...")
    val_walks, val_params = generate_walks(eligible_df, config, n_val)

    config['seed'] += 1
    print("Generating test walks...")
    test_walks, test_params = generate_walks(eligible_df, config, n_test)

    # Reset seed
    config['seed'] -= 2

    # Create datasets
    print("\nCreating train dataset...")
    train_data = create_dataset_dict(train_walks, train_params, eligible_df, tokenizer, config)
    train_dataset = Dataset.from_dict(train_data)

    print("Creating validation dataset...")
    val_data = create_dataset_dict(val_walks, val_params, eligible_df, tokenizer, config)
    val_dataset = Dataset.from_dict(val_data)

    print("Creating test dataset...")
    test_data = create_dataset_dict(test_walks, test_params, eligible_df, tokenizer, config)
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
        'eligible_cities': len(eligible_df),
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'seed': config['seed'],
        'max_len': max_len,
        'min_len': min_len,
        'avg_len': round(avg_len, 2),
        'min_max_distance': config['min_max_distance'],
        'max_max_distance': config['max_max_distance'],
        'min_chain_length': config['min_chain_length'],
        'max_chain_length': config['max_chain_length']
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
from typing import Tuple, List


class RandomWalkMetric:
    """Metric for random walk tasks: validity ratio × length penalty."""

    def __init__(self):
        self.failure_value = 0.0
        self.display_name = "Walk Validity Score"

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        cities_df = kwargs.get('cities_df')
        if cities_df is None:
            return self.failure_value

        # Parse expected parameters from prompt
        rw_match = re.search(r'rw\((\d+),(\d+)\)', prompt.replace(' ', ''))
        if not rw_match:
            return self.failure_value

        expected_max_dist = int(rw_match.group(1))
        expected_chain_len = int(rw_match.group(2))

        # Parse generated walk and count all attempted transitions
        transitions, total_attempted = self._parse_walk_transitions(generated)

        if total_attempted == 0:
            return self.failure_value

        # Validate transitions
        valid_trans = self._validate_transitions(
            transitions, cities_df, expected_max_dist
        )

        # Calculate validity ratio (valid transitions / total attempted transitions)
        validity_ratio = valid_trans / total_attempted

        # Calculate length penalty based on actual parsed cities (not attempted)
        actual_chain_len = len(transitions) + 1 if transitions else 0
        chain_len_diff = abs(actual_chain_len - expected_chain_len)
        length_penalty = np.exp(-chain_len_diff / expected_chain_len) if expected_chain_len > 0 else 0.0

        # Combined score
        return validity_ratio * length_penalty

    def _parse_walk_transitions(self, text: str) -> Tuple[List[Tuple[int, int]], int]:
        """Parse transitions from walk text. Returns: (list of valid transitions, total attempted transitions)"""
        text = text.replace(' ', '')

        match = re.search(r'=(.+)', text)
        if not match:
            return [], 0

        sequence = match.group(1)

        # Find ALL city-like tokens (valid or invalid)
        all_city_tokens = re.findall(r'c_\w+', sequence)

        if len(all_city_tokens) < 2:
            return [], len(all_city_tokens)

        # Total attempted transitions is number of consecutive city pairs
        total_attempted = len(all_city_tokens) - 1

        # Now extract only the valid transitions (with numeric IDs)
        city_matches = list(re.finditer(r'c_(\d+)', sequence))

        transitions = []
        for i in range(len(city_matches) - 1):
            city1_id = int(city_matches[i].group(1))
            city2_id = int(city_matches[i + 1].group(1))
            transitions.append((city1_id, city2_id))

        return transitions, total_attempted

    def _validate_transitions(self, transitions, cities_df, distance_threshold_km):
        """Validate transitions. Returns: number of valid transitions (both cities exist AND distance is valid)"""
        if not transitions:
            return 0

        valid_transitions = 0

        for city1_id, city2_id in transitions:
            # Check if both cities exist
            city1_rows = cities_df[cities_df['city_id'] == city1_id]
            city2_rows = cities_df[cities_df['city_id'] == city2_id]

            # If either city doesn't exist, this transition fails
            if len(city1_rows) == 0 or len(city2_rows) == 0:
                continue

            city1 = city1_rows.iloc[0]
            city2 = city2_rows.iloc[0]

            distance = np.sqrt((city2['x'] - city1['x'])**2 + (city2['y'] - city1['y'])**2)

            if distance <= distance_threshold_km:
                valid_transitions += 1

        return valid_transitions

    def format_for_print(self, value: float) -> str:
        return f"{value:.3f}"


# Singleton instance for import
METRIC = RandomWalkMetric()


if __name__ == "__main__":
    main()