#!/usr/bin/env python3
"""
Compass direction dataset creation script.
Determines the compass direction from one city to another.
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


# get_eligible_cities removed - now using generate_pairs from data_utils


def calculate_compass_direction(x1, y1, x2, y2):
    """
    Calculate compass direction from city1 to city2.
    Returns one of: N, NE, E, SE, S, SW, W, NW
    """
    # Calculate angle from city1 to city2
    dx = x2 - x1
    dy = y2 - y1

    # Special case: same location
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        # Could return a special value, but let's default to N
        return 'N'

    # Calculate angle in radians (-π to π)
    angle_rad = np.arctan2(dy, dx)

    # Convert to degrees (0 to 360)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    # Quantize to 8 directions
    # N: 337.5-22.5, NE: 22.5-67.5, E: 67.5-112.5, SE: 112.5-157.5
    # S: 157.5-202.5, SW: 202.5-247.5, W: 247.5-292.5, NW: 292.5-337.5

    # Rotate by 22.5 degrees to align boundaries
    rotated = (angle_deg + 22.5) % 360

    # Divide into 8 sectors
    sector = int(rotated / 45)

    # Map to compass directions (starting from E and going counter-clockwise)
    directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']

    return directions[sector]


def generate_compass_samples(df, config, n_samples):
    """Generate compass direction samples."""
    np.random.seed(config['seed'])

    # Use the generate_pairs function from data_utils
    indices_i, indices_j = generate_pairs(df, config, n_samples)

    # Apply random swapping to make pairs symmetric
    print("Applying random pair swapping for symmetry...")
    swap_mask = np.random.random(len(indices_i)) < 0.5
    indices_i_swapped = np.where(swap_mask, indices_j, indices_i)
    indices_j_swapped = np.where(swap_mask, indices_i, indices_j)
    indices_i, indices_j = indices_i_swapped, indices_j_swapped

    samples = []

    # Count directions for balance check
    direction_counts = {d: 0 for d in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}

    for idx_i, idx_j in tqdm(zip(indices_i, indices_j), total=n_samples, desc="Generating compass samples"):
        # Get cities from indices
        city1 = df.iloc[idx_i]
        city2 = df.iloc[idx_j]

        # Calculate direction
        direction = calculate_compass_direction(
            city1['x'], city1['y'],
            city2['x'], city2['y']
        )

        direction_counts[direction] += 1

        sample = {
            'city1': city1,
            'city2': city2,
            'direction': direction
        }

        samples.append(sample)

    # Print direction distribution
    print("\nDirection distribution:")
    for direction, count in sorted(direction_counts.items()):
        percentage = (count / n_samples) * 100
        print(f"  {direction}: {count} ({percentage:.1f}%)")

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
        direction = sample['direction']

        # Format with padding for city IDs if configured
        if use_padding:
            c1_str = str(city1_id).zfill(n_digits)
            c2_str = str(city2_id).zfill(n_digits)
            compass_str = f"compass(c_{c1_str},c_{c2_str})={direction}"
        else:
            compass_str = f"compass(c_{city1_id},c_{city2_id})={direction}"

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(compass_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("compass")

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
    parser = argparse.ArgumentParser(description='Create compass direction dataset')
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
    train_samples = generate_compass_samples(df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("\nGenerating validation samples...")
    val_samples = generate_compass_samples(df, config, n_val)

    config['seed'] += 1
    print("\nGenerating test samples...")
    test_samples = generate_compass_samples(df, config, n_test)

    # Reset seed
    config['seed'] -= 2

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


# ============================================================================
# Evaluation Metric
# ============================================================================


class CompassMetric:
    """Metric for compass direction tasks: binary accuracy."""

    def __init__(self):
        self.failure_value = 0.0
        self.display_name = "Direction Accuracy"

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        true_direction = true_completion.replace(' ', '').strip().upper()
        if '<EOS>' in true_direction:
            true_direction = true_direction.replace('<EOS>', '').strip()

        # Extract answer after the last '=' in generated text
        gen_no_space = generated.replace(' ', '')
        eq_pos = gen_no_space.rfind('=')
        if eq_pos != -1:
            gen_direction = gen_no_space[eq_pos+1:].strip().upper()
        else:
            gen_direction = gen_no_space.strip().upper()

        return 1.0 if true_direction == gen_direction else 0.0

    def format_for_print(self, value: float) -> str:
        return f"{value:.2f}"


# Singleton instance for import
METRIC = CompassMetric()


if __name__ == "__main__":
    main()