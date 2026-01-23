#!/usr/bin/env python3
"""
Create filtered city dataset with optional Atlantis regions using YAML configuration.
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
import yaml
import json
import shutil
from pathlib import Path
sys.path.append('.')  # Add root to path
from src.utils import extract_coordinates, init_directory

def main():
    parser = argparse.ArgumentParser(description='Create city dataset based on YAML configuration')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output if it already exists')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get config values
    pop_threshold = config.get('threshold', 100000)
    seed = config.get('seed', 42)
    max_id = config.get('max_id', None)
    output_path = config['output_dir']
    source_csv = config.get('source_csv', '/n/home12/cfpark00/WM_1/data/geonames-all-cities-with-a-population-1000.csv')
    atlantis_regions = config.get('atlantis_regions', [])
    region_mapping_path = config.get('region_mapping', None)
    
    # Read the original CSV file
    print("Loading city data...")
    # keep_default_na=False prevents "NA" (Namibia) from being interpreted as NaN
    df = pd.read_csv(source_csv, sep=';', encoding='utf-8-sig', keep_default_na=False)
    
    # Filter for population >= threshold
    print(f"Filtering cities with population >= {pop_threshold:,}...")
    filtered_df = df[df['Population'] >= pop_threshold].copy()
    
    # Extract coordinates
    print("Extracting coordinates...")
    filtered_df = extract_coordinates(filtered_df)
    
    # Standardize column names (extract_coordinates already added x,y columns)
    filtered_df = filtered_df.rename(columns={'id': 'Geoname ID', 'asciiname': 'ASCII Name'})
    
    # Load and generate Atlantis regions if config provided
    atlantis_region_mapping = {}
    if region_mapping_path:
        print(f"Loaded region mapping from {region_mapping_path}")
        with open(region_mapping_path, 'r') as f:
            atlantis_region_mapping = json.load(f)
    
    atlantis_dfs = []

    # Handle scattered_atlantis (uniform distribution) if specified
    scattered_atlantis = config.get('scattered_atlantis', None)
    if scattered_atlantis:
        region = scattered_atlantis
        print(f"Generating {region['n_cities']} SCATTERED cities with prefix '{region['city_prefix']}'...")

        # Generate random points using UNIFORM distribution
        np.random.seed(seed + hash(region['city_prefix']) % 1000)
        x_atlantis = np.random.uniform(region['x_min'], region['x_max'], region['n_cities'])
        y_atlantis = np.random.uniform(region['y_min'], region['y_max'], region['n_cities'])

        # Create city names using the specified prefix
        city_names = [f"{region['city_prefix']}_{i:03d}" for i in range(1, region['n_cities'] + 1)]

        # Create DataFrame for this region
        region_df = pd.DataFrame({
            'Geoname ID': [-1000000 - hash(region['city_prefix']) % 100000 - i for i in range(region['n_cities'])],
            'ASCII Name': city_names,
            'x': np.round(x_atlantis, 5),
            'y': np.round(y_atlantis, 5),
            'Country Code': [region['country_code']] * region['n_cities']
        })
        atlantis_dfs.append(region_df)
        print(f"  Generated: UNIFORM x=[{region['x_min']}, {region['x_max']}], y=[{region['y_min']}, {region['y_max']}], country_code={region['country_code']}")

    # Handle atlantis_regions (Gaussian distribution) - original behavior
    for region in atlantis_regions:
        print(f"Generating {region['n_cities']} CLUSTERED cities with prefix '{region['city_prefix']}'...")

        # Generate random points using Gaussian distribution
        # Note: Config values are in original scale (-180 to 180, -90 to 90)
        np.random.seed(seed + hash(region['city_prefix']) % 1000)  # Unique seed per region
        x_atlantis = np.random.normal(region['center_x'], region['std_dev'], region['n_cities'])
        y_atlantis = np.random.normal(region['center_y'], region['std_dev'], region['n_cities'])

        # Clip to valid coordinate ranges (original scale)
        x_atlantis = np.clip(x_atlantis, -180, 180)
        y_atlantis = np.clip(y_atlantis, -90, 90)

        # Create city names using the specified prefix
        city_names = [f"{region['city_prefix']}_{i:03d}" for i in range(1, region['n_cities'] + 1)]

        # Create DataFrame for this region
        region_df = pd.DataFrame({
            'Geoname ID': [-1000000 - hash(region['city_prefix']) % 100000 - i for i in range(region['n_cities'])],
            'ASCII Name': city_names,
            'x': np.round(x_atlantis, 5),
            'y': np.round(y_atlantis, 5),
            'Country Code': [region['country_code']] * region['n_cities']
        })
        atlantis_dfs.append(region_df)
        print(f"  Generated: GAUSSIAN center=({region['center_x']}, {region['center_y']}), std={region['std_dev']}, country_code={region['country_code']}")
    
    # Combine all Atlantis regions with world cities
    if atlantis_dfs:
        atlantis_combined = pd.concat(atlantis_dfs, ignore_index=True)
        print(f"Total Atlantis cities generated: {len(atlantis_combined)}")
        
        # Combine with filtered world cities (already renamed)
        filtered_df = pd.concat([
            filtered_df[['Geoname ID', 'ASCII Name', 'x', 'y', 'Country Code']],
            atlantis_combined
        ], ignore_index=True)
        print(f"Total cities (world + Atlantis): {len(filtered_df)}")
    else:
        filtered_df = filtered_df[['Geoname ID', 'ASCII Name', 'x', 'y', 'Country Code']]
    
    # Shuffle all rows together with seed
    print(f"\nShuffling all {len(filtered_df)} rows with seed={seed}...")
    filtered_df = filtered_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Generate random city IDs
    n_cities = len(filtered_df)
    if max_id is None:
        max_id = n_cities
        print(f"Using max_id={max_id} (number of cities)")
    else:
        print(f"Using specified max_id={max_id}")
        if max_id < n_cities:
            raise ValueError(f"max_id ({max_id}) must be >= number of cities ({n_cities})")
    
    # Generate random unique city IDs from [0, max_id-1]
    print(f"Assigning random city_ids from range [0, {max_id-1}]...")
    np.random.seed(seed)  # Use same seed for reproducibility
    city_ids = np.random.choice(max_id, size=n_cities, replace=False)
    
    # Load world region mapping
    print("Loading region mappings...")
    with open('/n/home12/cfpark00/WM_1/data/geographic_mappings/country_to_region.json', 'r') as f:
        world_region_mapping = json.load(f)
    
    # Combine world and Atlantis region mappings
    full_region_mapping = {**world_region_mapping, **atlantis_region_mapping}
    
    # Map country codes to regions
    filtered_df['Region'] = filtered_df['Country Code'].map(full_region_mapping)
    
    # Handle any unmapped country codes
    unmapped = filtered_df[filtered_df['Region'].isna()]['Country Code'].unique()
    if len(unmapped) > 0:
        print(f"Warning: Unmapped country codes: {unmapped}")
        filtered_df['Region'] = filtered_df['Region'].fillna('Unknown')
    
    # Create the new dataset with specified columns
    # Scale x,y by 10 to avoid decimals (x: -1800 to 1800, y: -900 to 900)
    result_df = pd.DataFrame({
        'id': filtered_df['Geoname ID'],
        'asciiname': filtered_df['ASCII Name'],
        'x': (filtered_df['x'] * 10).round().astype(int),  # Scale by 10 and round to int
        'y': (filtered_df['y'] * 10).round().astype(int),  # Scale by 10 and round to int
        'country_code': filtered_df['Country Code'],
        'region': filtered_df['Region'],  # Region name from mapping
        'city_id': city_ids  # Random IDs from [0, max_id-1]
    })
    
    # Handle output path - if it's a directory, create folder structure
    if not output_path.endswith('.csv'):
        # Treat as directory path
        output_dir = init_directory(output_path, overwrite=args.overwrite)
        csv_path = output_dir / 'cities.csv'
        
        # Save metadata
        metadata = {
            'config_file': args.config,
            'seed': seed,
            'max_id': max_id if max_id else n_cities,
            'population_threshold': pop_threshold,
            'created': pd.Timestamp.now(tz='UTC').isoformat(),
            'total_cities': n_cities,
            'columns': list(result_df.columns)
        }
        
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Copy config file
        config_copy_path = output_dir / 'config.yaml'
        shutil.copy(args.config, config_copy_path)
        
        print(f"\nSaving to directory: {output_dir}")
        print(f"  - cities.csv: Main dataset")
        print(f"  - metadata.json: Dataset metadata")
        print(f"  - config.yaml: Configuration used")
    else:
        # Regular CSV file path
        csv_path = Path(output_path)
        output_dir = csv_path.parent
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    result_df.to_csv(csv_path, index=False)
    
    print(f"\nDataset created successfully!")
    print(f"Total cities: {len(result_df):,}")
    print(f"Rows shuffled with seed: {seed}")
    print(f"\nFirst 5 rows:")
    print(result_df.head())
    print(f"\nColumn info:")
    print(result_df.info())

if __name__ == "__main__":
    main()