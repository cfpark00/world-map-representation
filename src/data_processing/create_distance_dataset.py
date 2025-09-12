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
    other_indices = df[~must_include_mask].index.values
    
    all_pairs_i = []
    all_pairs_j = []
    
    # Generate inter-must-include pairs (both cities from must_include groups)
    n_must = len(must_include_indices)
    if n_must >= 2:
        n_inter_pairs = min(n_pairs // 10, n_must * (n_must - 1) // 2)  # 10% inter pairs
        triu_indices = np.triu_indices(n_must, k=1)
        selected = np.random.choice(len(triu_indices[0]), size=min(n_inter_pairs, len(triu_indices[0])), replace=False)
        inter_i = must_include_indices[triu_indices[0][selected]]
        inter_j = must_include_indices[triu_indices[1][selected]]
        all_pairs_i.extend(inter_i)
        all_pairs_j.extend(inter_j)
    
    # Generate cross pairs (one from must_include, one from other)
    n_cross_pairs = n_pairs - len(all_pairs_i)
    if n_cross_pairs > 0 and len(other_indices) > 0:
        must_repeated = np.repeat(must_include_indices, len(other_indices))
        other_tiled = np.tile(other_indices, len(must_include_indices))
        
        n_available = len(must_repeated)
        selected = np.random.choice(n_available, size=min(n_cross_pairs, n_available), replace=False)
        
        cross_must = must_repeated[selected]
        cross_other = other_tiled[selected]
        
        all_pairs_i.extend(cross_must)
        all_pairs_j.extend(cross_other)
    
    # Shuffle all pairs
    shuffle_idx = np.random.permutation(len(all_pairs_i))
    return np.array(all_pairs_i)[shuffle_idx], np.array(all_pairs_j)[shuffle_idx]


def create_dataset_dict(indices_i, indices_j, df, tokenizer):
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
        # Convert "dist(c_1234,c_5678)=90" to space-delimited format
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
    train_data = create_dataset_dict(train_i, train_j, df, tokenizer)
    train_dataset = Dataset.from_dict(train_data)
    
    print("Creating validation dataset...")
    val_data = create_dataset_dict(val_i, val_j, df, tokenizer)
    val_dataset = Dataset.from_dict(val_data)
    
    print("Creating test dataset...")
    test_data = create_dataset_dict(test_i, test_j, df, tokenizer)
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