#!/usr/bin/env python3
"""
Unified distance dataset creation script.
Takes a single CSV file and defines groups through YAML configuration.
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
from src.data_processing.data_utils import generate_pairs


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
    mask = pd.Series([True] * len(df), index=df.index)  # Start with all True
    
    # Apply city_ids filter if specified
    if 'city_ids' in group_def:
        city_ids = set(group_def['city_ids'])
        city_mask = df['city_id'].isin(city_ids)
        mask = mask & city_mask
    
    # Apply city_names filter if specified
    if 'city_names' in group_def:
        # Support both exact names and patterns
        names = group_def['city_names']
        if isinstance(names, list):
            name_mask = df['asciiname'].isin(names)
        else:
            # Single name or pattern
            name_mask = df['asciiname'].str.contains(names, na=False)
        mask = mask & name_mask
    
    # Apply country_codes filter
    if 'country_codes' in group_def:
        cc_def = group_def['country_codes']
        if isinstance(cc_def, list):
            # Simple list of country codes - convert to regex pattern
            pattern = '^(' + '|'.join(cc_def) + ')$'
            cc_mask = df['country_code'].str.contains(pattern, na=False)
        else:
            # Single regex pattern
            cc_mask = df['country_code'].str.contains(cc_def, na=False)
        mask = mask & cc_mask
    
    # Apply regions filter
    if 'regions' in group_def:
        region_def = group_def['regions']
        if isinstance(region_def, list):
            # Simple list of regions - convert to regex pattern
            pattern = '^(' + '|'.join(region_def) + ')$'
            region_mask = df['region'].str.contains(pattern, na=False)
        else:
            # Single regex pattern
            region_mask = df['region'].str.contains(region_def, na=False)
        mask = mask & region_mask
    
    # Apply coordinate bounds if specified
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
    
    # Ensure we have x,y columns
    if 'x' not in df.columns:
        raise ValueError("CSV must have 'x' and 'y' columns")
    
    # Add city_id if not present
    if 'city_id' not in df.columns:
        if 'row_id' in df.columns:
            df['city_id'] = df['row_id']
        else:
            df['city_id'] = df.index
    
    # Apply group definitions if specified
    if 'groups' in config:
        print("\nApplying group definitions...")
        df = apply_group_definitions(df, config['groups'])
    else:
        # Default: all cities in one group
        df['group'] = 'all'
        print(f"No groups defined, using single group 'all': {len(df):,} cities")
    
    # Ensure we have required columns
    required_cols = ['city_id', 'x', 'y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


# Pair generation functions are now imported from data_utils


def create_dataset_dict(indices_i, indices_j, df, tokenizer, config=None):
    """Create a dictionary suitable for HuggingFace Dataset."""
    # Get coordinates in 2D space
    x1 = df.iloc[indices_i]['x'].values
    y1 = df.iloc[indices_i]['y'].values
    x2 = df.iloc[indices_j]['x'].values
    y2 = df.iloc[indices_j]['y'].values

    # Calculate Euclidean distances in 2D space
    distances = euclidean_distance(x1, y1, x2, y2)
    distances = np.round(distances).astype(int)

    # Get city IDs
    city1_ids = df.iloc[indices_i]['city_id'].values.astype(int)
    city2_ids = df.iloc[indices_j]['city_id'].values.astype(int)

    # Check for leading zeros configuration
    use_padding = config.get('leading_zeros', False) if config else False
    if use_padding:
        if 'n_id_digits' not in config:
            raise ValueError("When leading_zeros=true, n_id_digits must be specified in config")
        n_digits = config['n_id_digits']
        max_expressible = 10**n_digits - 1

        # Validate that all city IDs can be expressed with n_digits
        max_id = max(city1_ids.max(), city2_ids.max())
        if max_id > max_expressible:
            raise ValueError(f"City ID {max_id} exceeds maximum expressible with {n_digits} digits ({max_expressible}). "
                           f"Please increase n_id_digits or ensure city IDs are within range.")
    
    # Create text format and measure token lengths
    text_list = []
    task_type_list = []
    token_lengths = []
    loss_mask_list = []  # ALWAYS generate loss mask
    
    for c1, c2, d in tqdm(zip(city1_ids, city2_ids, distances),
                          total=len(city1_ids),
                          desc="Formatting samples",
                          leave=False):
        # Format for new tokenizer: space-delimited characters
        # Apply padding if configured
        if use_padding:
            c1_str = str(c1).zfill(n_digits)
            c2_str = str(c2).zfill(n_digits)
            dist_str = f"dist(c_{c1_str},c_{c2_str})={d}"
        else:
            # Original format without padding
            dist_str = f"dist(c_{c1},c_{c2})={d}"
        # Add spaces between each character for the tokenizer
        spaced_str = ' '.join(dist_str)
        # Add special tokens with spaces
        text = f"<bos> {spaced_str} <eos>"
        text_list.append(text)
        task_type_list.append("distance")
        
        # Tokenize once for both length and mask
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_lengths.append(len(tokens))
        
        # ALWAYS create loss mask (training decides whether to use it)
        # Find where the equals sign appears in the tokenized version
        # We need to mask everything up to and including the '=' token
        equals_token_id = tokenizer.encode('=', add_special_tokens=False)[0]
        
        # Create mask: 0 for prompt, 1 for answer
        mask = []
        found_equals = False
        for token_id in tokens:
            if not found_equals:
                mask.append('0')
                if token_id == equals_token_id:
                    found_equals = True
            else:
                mask.append('1')  # Everything after = gets loss (including <eos>)
        
        loss_mask = ''.join(mask)
        loss_mask_list.append(loss_mask)
    
    return {
        'text': text_list,
        'task_type': task_type_list,
        'token_lengths': token_lengths,
        'loss_mask': loss_mask_list  # Always included
    }




def main():
    parser = argparse.ArgumentParser(description='Create distance dataset with configurable pair generation')
    parser.add_argument('config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory if it exists')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config_path}")
    config = load_config(args.config_path)
    
    # Validate config has required fields
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if 'cities_csv' not in config:
        raise ValueError("FATAL: 'cities_csv' is required in config")
    
    # Load tokenizer (default path, can be overridden in config)
    tokenizer_path = config.get('tokenizer_path', 'data/tokenizers/default_tokenizer')
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Load cities from config-specified path
    df = load_cities(config['cities_csv'], config)
    print(f"Loaded {len(df):,} cities total")
    
    # Display group statistics
    print("\nCity groups:")
    for group in df['group'].unique():
        count = len(df[df['group'] == group])
        print(f"  {group}: {count:,} cities")
    
    # Generate pairs for each split
    n_train = config['n_train']
    n_val = config['n_val']
    n_test = config['n_test']
    
    print(f"\nGenerating pairs (seed={config['seed']})...")
    print(f"  Train: {n_train:,}")
    print(f"  Val: {n_val:,}")
    print(f"  Test: {n_test:,}")
    
    # Generate all pairs at once and split
    n_total = n_train + n_val + n_test
    all_i, all_j = generate_pairs(df, config, n_total)
    
    # Apply random swapping to make pairs symmetric
    print("Applying random pair swapping for symmetry...")
    swap_mask = np.random.random(len(all_i)) < 0.5
    all_i_swapped = np.where(swap_mask, all_j, all_i)
    all_j_swapped = np.where(swap_mask, all_i, all_j)
    all_i, all_j = all_i_swapped, all_j_swapped
    
    # Split into train/val/test
    train_i = all_i[:n_train]
    train_j = all_j[:n_train]
    val_i = all_i[n_train:n_train+n_val]
    val_j = all_j[n_train:n_train+n_val]
    test_i = all_i[n_train+n_val:]
    test_j = all_j[n_train+n_val:]
    
    # Create datasets (loss_mask is always generated)
    print("\nCreating train dataset...")
    train_data = create_dataset_dict(train_i, train_j, df, tokenizer, config)
    train_dataset = Dataset.from_dict(train_data)

    print("Creating validation dataset...")
    val_data = create_dataset_dict(val_i, val_j, df, tokenizer, config)
    val_dataset = Dataset.from_dict(val_data)

    print("Creating test dataset...")
    test_data = create_dataset_dict(test_i, test_j, df, tokenizer, config)
    test_dataset = Dataset.from_dict(test_data)
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Initialize output directory with safety checks
    output_path = init_directory(config['output_dir'], overwrite=args.overwrite)
    
    print(f"\nSaving dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))
    
    # Calculate token length statistics across all splits
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
    
    # Copy config file to output directory
    config_copy_path = output_path / 'config.yaml'
    shutil.copy(args.config_path, config_copy_path)
    
    print(f"\nSaved files:")
    print(f"  - HuggingFace dataset files")
    print(f"  - metadata.json: Dataset metadata")
    print(f"  - config.yaml: Configuration used")
    
    
    # Display sample rows
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