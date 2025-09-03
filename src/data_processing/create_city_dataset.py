#!/usr/bin/env python3
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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create filtered city dataset with optional Atlantis regions')
parser.add_argument('--threshold', type=int, default=100000,
                   help='Population threshold (default: 100000)')
parser.add_argument('--seed', type=int, default=42,
                   help='Random seed for shuffling rows (default: 42)')
parser.add_argument('--max-id', type=int, default=None,
                   help='Maximum city ID value. IDs will be randomly assigned from [0, max_id-1]. If not specified, uses number of cities.')
parser.add_argument('--atlantis-config', type=str, default=None,
                   help='Path to YAML config file defining Atlantis regions to add')
parser.add_argument('--output', type=str, required=True,
                   help='Output CSV file path or directory')
parser.add_argument('--overwrite', action='store_true',
                   help='Overwrite output if it already exists')
args = parser.parse_args()

pop_threshold = args.threshold
seed = args.seed
max_id = args.max_id
atlantis_config = args.atlantis_config
output_path = args.output
overwrite = args.overwrite

# Read the original CSV file
print("Loading city data...")
df = pd.read_csv('/n/home12/cfpark00/WM_1/data/geonames-all-cities-with-a-population-1000.csv', 
                 sep=';', encoding='utf-8-sig')

# Filter for population >= threshold
print(f"Filtering cities with population >= {pop_threshold:,}...")
filtered_df = df[df['Population'] >= pop_threshold].copy()

# Extract coordinates
print("Extracting coordinates...")
filtered_df = extract_coordinates(filtered_df)

# Convert to x,y coordinates before shuffling
x_coords = filtered_df['longitude'].values
y_coords = filtered_df['latitude'].values

# Create temporary dataframe with x,y and standardize column names
filtered_df['x'] = x_coords
filtered_df['y'] = y_coords
filtered_df = filtered_df.rename(columns={'id': 'Geoname ID', 'asciiname': 'ASCII Name'})

# Load and generate Atlantis regions if config provided
atlantis_region_mapping = {}
if atlantis_config:
    print(f"\nLoading Atlantis config from {atlantis_config}...")
    with open(atlantis_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load Atlantis region mapping if specified
    if 'region_mapping' in config:
        with open(config['region_mapping'], 'r') as f:
            atlantis_region_mapping = json.load(f)
        print(f"Loaded region mapping from {config['region_mapping']}")
    
    atlantis_dfs = []
    for region in config['atlantis_regions']:
        print(f"Generating {region['n_cities']} cities with prefix '{region['city_prefix']}'...")
        
        # Generate random points using Gaussian distribution
        np.random.seed(seed + hash(region['city_prefix']) % 1000)  # Unique seed per region
        x_atlantis = np.random.normal(region['center_x'], region['std_dev'], region['n_cities'])
        y_atlantis = np.random.normal(region['center_y'], region['std_dev'], region['n_cities'])
        
        # Clip to valid coordinate ranges
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
        print(f"  Generated: center=({region['center_x']}, {region['center_y']}), std={region['std_dev']}, country_code={region['country_code']}")
    
    # Combine all Atlantis regions with world cities
    atlantis_combined = pd.concat(atlantis_dfs, ignore_index=True)
    print(f"Total Atlantis cities generated: {len(atlantis_combined)}")
    
    # Combine with filtered world cities (already renamed)
    filtered_df = pd.concat([
        filtered_df[['Geoname ID', 'ASCII Name', 'x', 'y', 'Country Code']],
        atlantis_combined
    ], ignore_index=True)
    print(f"Total cities (world + Atlantis): {len(filtered_df)}")

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
# Only x,y coordinates (no longitude/latitude)
result_df = pd.DataFrame({
    'id': filtered_df['Geoname ID'],
    'asciiname': filtered_df['ASCII Name'],
    'x': filtered_df['x'],  # Cartesian x coordinate
    'y': filtered_df['y'],  # Cartesian y coordinate
    'country_code': filtered_df['Country Code'],
    'region': filtered_df['Region'],  # Region name from mapping
    'city_id': city_ids  # Random IDs from [0, max_id-1]
})

# Handle output path - if it's a directory, create folder structure
if not output_path.endswith('.csv'):
    # Treat as directory path
    output_dir = init_directory(output_path, overwrite=overwrite)
    csv_path = output_dir / 'cities.csv'
    
    # Save metadata
    metadata = {
        'seed': seed,
        'max_id': max_id if max_id else n_cities,
        'population_threshold': pop_threshold,
        'atlantis_config': atlantis_config,
        'created': pd.Timestamp.now(tz='UTC').isoformat(),
        'total_cities': n_cities,
        'columns': list(result_df.columns)
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy Atlantis config if used
    if atlantis_config:
        atlantis_copy_path = output_dir / 'atlantis_config.yaml'
        shutil.copy(atlantis_config, atlantis_copy_path)
    
    print(f"\nSaving to directory: {output_dir}")
    print(f"  - cities.csv: Main dataset")
    print(f"  - metadata.json: Dataset metadata")
    if atlantis_config:
        print(f"  - atlantis_config.yaml: Atlantis configuration used")
else:
    # Regular CSV file path
    csv_path = output_path
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

# Save to CSV
result_df.to_csv(csv_path, index=False)

print(f"\nDataset created successfully!")
print(f"Total cities: {len(result_df):,}")
print(f"Rows shuffled with seed: {seed}")
print(f"\nFirst 5 rows:")
print(result_df.head())
print(f"\nColumn info:")
print(result_df.info())