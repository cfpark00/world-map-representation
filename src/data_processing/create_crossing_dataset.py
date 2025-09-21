#!/usr/bin/env python3
"""
Line crossing dataset creation script.
Determines if two line segments intersect within their bounds.
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

    # Control ratio of TRUE/FALSE examples
    config.setdefault('true_ratio', 0.5)  # 50% TRUE, 50% FALSE

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


def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Check if line segment (x1,y1)-(x2,y2) intersects with segment (x3,y3)-(x4,y4).
    Uses the cross product method for robust intersection detection.
    """
    def ccw(ax, ay, bx, by, cx, cy):
        """Check if three points are in counter-clockwise order."""
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    # Check if segments actually intersect (not just the infinite lines)
    # Two segments intersect if each segment straddles the line containing the other
    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4)) and \
           (ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))


def get_eligible_cities(df, config):
    """Get cities eligible for segment generation based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        return df
    elif strategy == 'within_groups':
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]
    elif strategy == 'must_include':
        # For must_include, return all cities - we'll handle the constraint in generate_crossing_pairs
        return df
    else:
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for crossing detection")
        return df


def generate_crossing_pairs(df, config, n_samples):
    """Generate pairs of line segments with controlled TRUE/FALSE ratio."""
    np.random.seed(config['seed'])

    eligible_df = get_eligible_cities(df, config)
    n_cities = len(eligible_df)

    if n_cities < 4:
        raise ValueError(f"Need at least 4 cities, but only have {n_cities}")

    true_ratio = config['true_ratio']
    n_true = int(n_samples * true_ratio)
    n_false = n_samples - n_true

    samples = []

    # Strategy: Generate random segments and check if they intersect
    # Keep generating until we have enough TRUE and FALSE examples

    true_samples = []
    false_samples = []

    max_attempts = n_samples * 100  # Prevent infinite loop
    attempts = 0

    pbar = tqdm(total=n_samples, desc="Generating crossing pairs")

    while (len(true_samples) < n_true or len(false_samples) < n_false) and attempts < max_attempts:
        attempts += 1

        # Generate 4 random cities
        if config['pair_generation'].get('strategy') == 'must_include':
            # Ensure at least one city is from must_include groups
            must_include_groups = config['pair_generation']['must_include_groups']
            must_include_mask = eligible_df['group'].isin(must_include_groups)
            must_include_indices = eligible_df[must_include_mask].index.values
            other_indices = eligible_df[~must_include_mask].index.values

            # Pick one from must_include, rest can be from anywhere
            must_idx = np.random.choice(must_include_indices, size=1)
            all_other_indices = np.concatenate([must_include_indices, other_indices])
            # Remove the already selected must_idx from possible choices
            available = all_other_indices[all_other_indices != must_idx[0]]
            other_idx = np.random.choice(available, size=3, replace=False)
            selected_indices = np.concatenate([must_idx, other_idx])
            np.random.shuffle(selected_indices)  # Shuffle so must_include isn't always first

            cities = eligible_df.loc[selected_indices]
        else:
            indices = np.random.choice(n_cities, size=4, replace=False)
            cities = eligible_df.iloc[indices]

        x1, y1 = cities.iloc[0]['x'], cities.iloc[0]['y']
        x2, y2 = cities.iloc[1]['x'], cities.iloc[1]['y']
        x3, y3 = cities.iloc[2]['x'], cities.iloc[2]['y']
        x4, y4 = cities.iloc[3]['x'], cities.iloc[3]['y']

        # Check if segments intersect
        intersects = segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4)

        sample = {
            'city1': cities.iloc[0],
            'city2': cities.iloc[1],
            'city3': cities.iloc[2],
            'city4': cities.iloc[3],
            'intersects': intersects
        }

        if intersects and len(true_samples) < n_true:
            true_samples.append(sample)
            pbar.update(1)
        elif not intersects and len(false_samples) < n_false:
            false_samples.append(sample)
            pbar.update(1)

    pbar.close()

    # Combine and shuffle
    samples = true_samples + false_samples
    np.random.shuffle(samples)

    # Apply random shuffling of all 4 cities for symmetry
    print("Applying random city shuffling for symmetry...")
    for sample in samples:
        # Randomly shuffle all 4 cities
        cities = [sample['city1'], sample['city2'], sample['city3'], sample['city4']]
        np.random.shuffle(cities)
        sample['city1'], sample['city2'], sample['city3'], sample['city4'] = cities

        # Recalculate intersection with the new arrangement
        x1, y1 = sample['city1']['x'], sample['city1']['y']
        x2, y2 = sample['city2']['x'], sample['city2']['y']
        x3, y3 = sample['city3']['x'], sample['city3']['y']
        x4, y4 = sample['city4']['x'], sample['city4']['y']
        sample['intersects'] = segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4)

    if len(samples) < n_samples:
        print(f"Warning: Could only generate {len(samples)} samples out of {n_samples} requested")
        print(f"  TRUE samples: {len(true_samples)}")
        print(f"  FALSE samples: {len(false_samples)}")

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
        city1_id = int(sample['city1']['city_id'])
        city2_id = int(sample['city2']['city_id'])
        city3_id = int(sample['city3']['city_id'])
        city4_id = int(sample['city4']['city_id'])
        intersects = sample['intersects']

        # Format with padding if configured
        if use_padding:
            c1_str = str(city1_id).zfill(n_digits)
            c2_str = str(city2_id).zfill(n_digits)
            c3_str = str(city3_id).zfill(n_digits)
            c4_str = str(city4_id).zfill(n_digits)
            cross_str = f"cross(c_{c1_str},c_{c2_str};c_{c3_str},c_{c4_str})="
        else:
            cross_str = f"cross(c_{city1_id},c_{city2_id};c_{city3_id},c_{city4_id})="

        # Add TRUE or FALSE
        result_str = "TRUE" if intersects else "FALSE"
        full_str = cross_str + result_str

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(full_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("crossing")

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
    parser = argparse.ArgumentParser(description='Create line crossing dataset')
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
    print(f"\nLine crossing parameters:")
    print(f"  TRUE/FALSE ratio: {config['true_ratio']:.1%} TRUE")

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
    train_samples = generate_crossing_pairs(df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("Generating validation samples...")
    val_samples = generate_crossing_pairs(df, config, n_val)

    config['seed'] += 1
    print("Generating test samples...")
    test_samples = generate_crossing_pairs(df, config, n_test)

    # Reset seed
    config['seed'] -= 2

    # Count TRUE/FALSE distribution
    train_true = sum(1 for s in train_samples if s['intersects'])
    val_true = sum(1 for s in val_samples if s['intersects'])
    test_true = sum(1 for s in test_samples if s['intersects'])

    print(f"\nActual TRUE ratios:")
    print(f"  Train: {train_true}/{len(train_samples)} ({train_true/len(train_samples):.1%})")
    print(f"  Val: {val_true}/{len(val_samples)} ({val_true/len(val_samples):.1%})")
    print(f"  Test: {test_true}/{len(test_samples)} ({test_true/len(test_samples):.1%})")

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
        'true_ratio_target': config['true_ratio'],
        'true_ratio_actual': {
            'train': train_true / len(train_samples),
            'val': val_true / len(val_samples),
            'test': test_true / len(test_samples)
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