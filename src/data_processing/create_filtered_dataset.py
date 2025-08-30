#!/usr/bin/env python3
import pandas as pd
import os
import sys
sys.path.append('.')  # Add root to path
from src.utils import extract_coordinates

# Get population threshold from command line argument
if len(sys.argv) > 1:
    pop_threshold = int(sys.argv[1])
else:
    pop_threshold = 50000  # Default value

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

# Create the new dataset with specified columns
result_df = pd.DataFrame({
    'id': filtered_df['Geoname ID'],
    'asciiname': filtered_df['ASCII Name'],
    'longitude': filtered_df['longitude'],
    'latitude': filtered_df['latitude'],
    'country_code': filtered_df['Country Code'],
    'row_id': range(len(filtered_df))
})

# Create output directory if it doesn't exist
os.makedirs('outputs/datasets', exist_ok=True)

# Save to CSV with threshold in filename
threshold_str = f"{pop_threshold//1000}k" if pop_threshold < 1000000 else f"{pop_threshold//1000000}m"
output_path = f'outputs/datasets/cities_{threshold_str}_plus.csv'
result_df.to_csv(output_path, index=False)

print(f"\nDataset created successfully!")
print(f"Output file: {output_path}")
print(f"Total cities: {len(result_df):,}")
print(f"\nFirst 5 rows:")
print(result_df.head())
print(f"\nColumn info:")
print(result_df.info())