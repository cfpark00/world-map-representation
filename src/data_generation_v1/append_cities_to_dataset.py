#!/usr/bin/env python3
"""
Append new synthetic cities to an existing city dataset.
Reads existing CSV, adds new cities, assigns unused city_ids randomly.
"""
import pandas as pd
import numpy as np
import sys
import argparse
import yaml
import json
import shutil
from pathlib import Path
sys.path.append('.')  # Add root to path
from src.utils import init_directory


def main():
    parser = argparse.ArgumentParser(description='Append cities to existing dataset')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output if it exists')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if 'input_csv' not in config:
        raise ValueError("FATAL: 'input_csv' is required in config")
    if 'pacificus_region' not in config:
        raise ValueError("FATAL: 'pacificus_region' is required in config")

    # Get config values
    input_csv = Path(config['input_csv'])
    output_dir = Path(config['output_dir'])
    seed = config.get('seed', 42)
    max_id = config.get('max_id', 10000)
    region_mapping = config.get('region_mapping', {})
    pacificus = config['pacificus_region']

    # Load existing dataset (READ ONLY)
    print(f"Loading existing cities from {input_csv}...")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    existing_df = pd.read_csv(input_csv)
    print(f"Loaded {len(existing_df)} existing cities")

    # Get used city_ids
    used_ids = set(existing_df['city_id'].values)
    print(f"Found {len(used_ids)} used city IDs")

    # Find available IDs
    all_possible_ids = set(range(max_id))
    available_ids = all_possible_ids - used_ids
    print(f"Available city IDs: {len(available_ids)} out of {max_id}")

    # Check if we have enough IDs
    n_new_cities = pacificus['n_cities']
    if len(available_ids) < n_new_cities:
        raise ValueError(f"Not enough available IDs! Need {n_new_cities}, but only {len(available_ids)} available")

    # Generate Pacificus cities
    print(f"\nGenerating {n_new_cities} Pacificus cities...")
    print(f"  Center: ({pacificus['center_x']}, {pacificus['center_y']})")
    print(f"  Std dev: {pacificus['std_dev']}")
    print(f"  Country code: {pacificus['country_code']}")

    # Generate random points using Gaussian distribution
    np.random.seed(seed)
    x_pacificus = np.random.normal(pacificus['center_x'], pacificus['std_dev'], n_new_cities)
    y_pacificus = np.random.normal(pacificus['center_y'], pacificus['std_dev'], n_new_cities)

    # Clip to valid coordinate ranges (original scale)
    x_pacificus = np.clip(x_pacificus, -180, 180)
    y_pacificus = np.clip(y_pacificus, -90, 90)

    # Scale by 10 to match existing dataset format
    x_pacificus_scaled = (x_pacificus * 10).round().astype(int)
    y_pacificus_scaled = (y_pacificus * 10).round().astype(int)

    # Create city names
    city_names = [f"{pacificus['city_prefix']}_{i:03d}" for i in range(1, n_new_cities + 1)]

    # Randomly select city IDs from available ones
    np.random.seed(seed)
    selected_ids = np.random.choice(list(available_ids), size=n_new_cities, replace=False)

    # Get region name from mapping
    region_name = region_mapping.get(pacificus['country_code'], 'Unknown')

    # Create DataFrame for new cities
    new_cities_df = pd.DataFrame({
        'id': [-2000000 - i for i in range(n_new_cities)],  # Negative IDs for synthetic cities
        'asciiname': city_names,
        'x': x_pacificus_scaled,
        'y': y_pacificus_scaled,
        'country_code': pacificus['country_code'],
        'region': region_name,
        'city_id': selected_ids
    })

    print(f"Generated {len(new_cities_df)} new cities with IDs from available pool")

    # Combine with existing dataset
    combined_df = pd.concat([existing_df, new_cities_df], ignore_index=True)
    print(f"\nTotal cities in combined dataset: {len(combined_df)}")

    # Shuffle the combined dataset
    print(f"Shuffling combined dataset with seed {seed}...")
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Initialize output directory
    output_dir = init_directory(output_dir, overwrite=args.overwrite)

    # Save combined dataset
    output_csv = output_dir / 'cities.csv'
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved combined dataset to {output_csv}")

    # Save metadata
    metadata = {
        'config_file': args.config,
        'input_csv': str(input_csv),
        'seed': seed,
        'max_id': max_id,
        'created': pd.Timestamp.now(tz='UTC').isoformat(),
        'original_cities': len(existing_df),
        'new_cities': len(new_cities_df),
        'total_cities': len(combined_df),
        'pacificus_config': pacificus,
        'used_ids_sample': [int(x) for x in sorted(selected_ids)[:10]],  # Convert to Python int
        'columns': list(combined_df.columns)
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy config file
    config_copy_path = output_dir / 'config.yaml'
    shutil.copy(args.config, config_copy_path)

    print(f"\nSaved files:")
    print(f"  - cities.csv: Combined dataset")
    print(f"  - metadata.json: Dataset metadata")
    print(f"  - config.yaml: Configuration used")

    # Display summary statistics
    print(f"\nSummary:")
    print(f"  Original cities: {len(existing_df):,}")
    print(f"  New Pacificus cities: {len(new_cities_df):,}")
    print(f"  Total cities: {len(combined_df):,}")
    print(f"  City ID range: 0-{max_id-1}")

    if args.debug:
        print(f"\nFirst 5 Pacificus cities added:")
        print(new_cities_df.head())
        print(f"\nSample of assigned city IDs: {sorted(selected_ids)[:20]}")

    print("\nDataset successfully created!")


if __name__ == "__main__":
    main()