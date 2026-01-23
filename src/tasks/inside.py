#!/usr/bin/env python3
"""
Inside convex hull dataset creation script.
Determines if a point is inside the convex hull of a set of points.
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
from scipy.spatial import ConvexHull, Delaunay, QhullError

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

    # Inside-specific parameters
    config.setdefault('min_n', 3)
    config.setdefault('max_n', 6)
    config.setdefault('true_ratio', 0.5)  # 50% TRUE, 50% FALSE

    # Validate min_n
    if config['min_n'] < 3:
        raise ValueError("min_n must be >= 3 for convex hull in 2D")

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
    """Get cities eligible for hull generation based on config."""
    strategy = config['pair_generation']['strategy']

    if strategy == 'all_pairs':
        return df
    elif strategy == 'within_groups':
        groups = config['pair_generation'].get('groups', df['group'].unique())
        mask = df['group'].isin(groups)
        return df[mask]
    elif strategy == 'must_include':
        # For must_include, return all cities - we'll handle the constraint in generate_inside_queries
        return df
    else:
        print(f"Note: Strategy '{strategy}' interpreted as 'all_pairs' for inside detection")
        return df


def point_in_convex_hull(point, hull_points):
    """
    Check if a point is inside the convex hull of given points.
    Returns True if inside, False otherwise.
    """
    try:
        # Need at least 3 points for a 2D hull
        if len(hull_points) < 3:
            return False

        # Check if points are collinear
        if len(hull_points) == 3:
            # Use cross product to check collinearity
            p1, p2, p3 = hull_points
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if abs(cross) < 1e-10:  # Points are collinear
                return False

        # Compute convex hull
        hull = ConvexHull(hull_points)

        # Get only the hull vertices (boundary points)
        hull_vertices = hull_points[hull.vertices]

        # Use Delaunay triangulation for point-in-hull test
        delaunay = Delaunay(hull_vertices)

        # Check if point is inside
        return delaunay.find_simplex(point) >= 0

    except QhullError:
        # Handle degenerate cases (collinear points, etc.)
        return False


def generate_inside_samples(df, config, n_samples):
    """Generate samples with controlled TRUE/FALSE ratio."""
    np.random.seed(config['seed'])

    eligible_df = get_eligible_cities(df, config)
    n_cities = len(eligible_df)

    min_n = config['min_n']
    max_n = config['max_n']
    true_ratio = config['true_ratio']

    if n_cities < max_n + 1:
        raise ValueError(f"Need at least {max_n + 1} cities, but only have {n_cities}")

    n_true = int(n_samples * true_ratio)
    n_false = n_samples - n_true

    true_samples = []
    false_samples = []

    max_attempts = n_samples * 100
    attempts = 0

    pbar = tqdm(total=n_samples, desc="Generating inside samples")

    while (len(true_samples) < n_true or len(false_samples) < n_false) and attempts < max_attempts:
        attempts += 1

        # Random number of hull points
        n_hull = np.random.randint(min_n, max_n + 1)

        # Select n_hull + 1 cities (one for test point, rest for hull)
        if config['pair_generation'].get('strategy') == 'must_include':
            # Simple: ensure at least one city (anywhere) is from must_include groups
            must_include_groups = config['pair_generation']['must_include_groups']
            must_include_mask = eligible_df['group'].isin(must_include_groups)
            must_include_indices = eligible_df[must_include_mask].index.values
            all_indices = eligible_df.index.values

            # Pick one from must_include
            must_idx = np.random.choice(must_include_indices)

            # Pick n_hull more cities from all cities (for test point + hull vertices)
            other_indices = np.random.choice(all_indices, size=n_hull, replace=False)

            # Ensure must_idx is not duplicated
            while must_idx in other_indices:
                other_indices = np.random.choice(all_indices, size=n_hull, replace=False)

            # Combine all cities and shuffle completely
            all_selected = np.concatenate([[must_idx], other_indices])
            np.random.shuffle(all_selected)

            cities = eligible_df.loc[all_selected]
        else:
            indices = np.random.choice(n_cities, size=n_hull + 1, replace=False)
            cities = eligible_df.iloc[indices]

        # After shuffling, first city is the test point
        test_city = cities.iloc[0]
        test_point = np.array([test_city['x'], test_city['y']])

        # Rest form the hull
        hull_cities = cities.iloc[1:]
        hull_points = np.array([[c['x'], c['y']] for _, c in hull_cities.iterrows()])

        # Check if test point is inside hull
        inside = point_in_convex_hull(test_point, hull_points)

        sample = {
            'test_city': test_city,
            'hull_cities': hull_cities,
            'inside': inside
        }

        if inside and len(true_samples) < n_true:
            true_samples.append(sample)
            pbar.update(1)
        elif not inside and len(false_samples) < n_false:
            false_samples.append(sample)
            pbar.update(1)

    pbar.close()

    # Combine and shuffle
    samples = true_samples + false_samples
    np.random.shuffle(samples)

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
        test_city_id = int(sample['test_city']['city_id'])
        hull_city_ids = sample['hull_cities']['city_id'].values.astype(int)
        inside = sample['inside']

        # Format with padding if configured
        if use_padding:
            test_str = str(test_city_id).zfill(n_digits)
            hull_strs = [str(cid).zfill(n_digits) for cid in hull_city_ids]
            inside_str = f"inside(c_{test_str};" + ','.join(f"c_{h}" for h in hull_strs) + ")="
        else:
            inside_str = f"inside(c_{test_city_id};" + ','.join(f"c_{cid}" for cid in hull_city_ids) + ")="

        # Add TRUE or FALSE
        result_str = "TRUE" if inside else "FALSE"
        full_str = inside_str + result_str

        # Add spaces between each character for tokenizer
        spaced_str = ' '.join(full_str)
        text = f"<bos> {spaced_str} <eos>"

        text_list.append(text)
        task_type_list.append("inside")

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
    parser = argparse.ArgumentParser(description='Create inside convex hull dataset')
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

    # Assert min_n >= 3
    if config['min_n'] < 3:
        raise ValueError(f"min_n must be >= 3 for convex hull, got {config['min_n']}")

    # Load tokenizer
    tokenizer_path = config.get('tokenizer_path', 'data/tokenizers/default_tokenizer')
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load cities
    df = load_cities(config['cities_csv'], config)
    print(f"Loaded {len(df):,} cities total")

    # Display parameters
    print(f"\nInside convex hull parameters:")
    print(f"  Hull size range: [{config['min_n']}, {config['max_n']}]")
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
    train_samples = generate_inside_samples(df, config, n_train)

    # Increment seed for different splits
    config['seed'] += 1
    print("Generating validation samples...")
    val_samples = generate_inside_samples(df, config, n_val)

    config['seed'] += 1
    print("Generating test samples...")
    test_samples = generate_inside_samples(df, config, n_test)

    # Reset seed
    config['seed'] -= 2

    # Count TRUE/FALSE distribution
    train_true = sum(1 for s in train_samples if s['inside'])
    val_true = sum(1 for s in val_samples if s['inside'])
    test_true = sum(1 for s in test_samples if s['inside'])

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
        'min_n': config['min_n'],
        'max_n': config['max_n'],
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


# ============================================================================
# Evaluation Metric
# ============================================================================


class BooleanMetric:
    """Metric for TRUE/FALSE tasks: binary accuracy."""

    def __init__(self):
        self.failure_value = 0.0
        self.display_name = "Binary Accuracy"

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        true_value = true_completion.replace(' ', '').strip().upper()
        if '<EOS>' in true_value:
            true_value = true_value.replace('<EOS>', '').strip()

        # Extract answer after the last '=' in generated text
        gen_no_space = generated.replace(' ', '')
        eq_pos = gen_no_space.rfind('=')
        if eq_pos != -1:
            gen_value = gen_no_space[eq_pos+1:].strip().upper()
        else:
            gen_value = gen_no_space.strip().upper()

        return 1.0 if true_value == gen_value else 0.0

    def format_for_print(self, value: float) -> str:
        return f"{value:.2f}"


# Singleton instance for import
METRIC = BooleanMetric()


if __name__ == "__main__":
    main()