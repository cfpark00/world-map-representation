#!/usr/bin/env python3
"""
Create dataset with ONLY inter-Atlantis city pairs for distance prediction.
This creates pairs where both cities are from Atlantis.
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')  # Add root to path
from src.utils import haversine
from datasets import Dataset, DatasetDict
import os
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create inter-Atlantis distance dataset')
parser.add_argument('output_dir', type=str, help='Output directory for the dataset')
parser.add_argument('--atlantis-csv', type=str, 
                   default='outputs/datasets/atlantis_XX0_100_seed42.csv',
                   help='Path to Atlantis CSV file')
parser.add_argument('--n_train', type=int, default=4000, 
                   help='Number of training samples (default: 4000)')
parser.add_argument('--n_val', type=int, default=128, 
                   help='Number of validation samples (default: 128)')
parser.add_argument('--n_test', type=int, default=500, 
                   help='Number of test samples (default: 500)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--offset', type=int, default=None, 
                   help='Offset for Atlantis city IDs (default: auto-detect from world cities)')
parser.add_argument('--cities-csv', type=str, default='outputs/datasets/cities_100k_plus_seed42.csv',
                   help='Path to world cities CSV to determine offset')

args = parser.parse_args()

n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
n_tot = n_train + n_val + n_test
output_dir = args.output_dir
seed = args.seed
# Auto-detect offset if not provided
if args.offset is None:
    # Load world cities to get the max row_id
    import pandas as pd
    world_df = pd.read_csv(args.cities_csv)
    offset = len(world_df)  # This will be 5075 (since row_id goes from 0 to 5074)
    print(f"Auto-detected offset: {offset} (continuing from world cities)")
else:
    offset = args.offset

# Load the Atlantis dataset
print("Loading Atlantis cities data...")
try:
    df_atlantis = pd.read_csv(args.atlantis_csv)
except FileNotFoundError as e:
    print(f"Error: Could not find Atlantis CSV at {args.atlantis_csv}")
    print("Please run: python src/data_processing/generate_atlantis.py")
    sys.exit(1)

# Apply offset to row_ids to continue from world cities
# World cities have row_ids 0-5074, so Atlantis starts at 5075
df_atlantis['row_id'] = df_atlantis.index + offset

n_cities = len(df_atlantis)

print(f"Number of Atlantis cities: {n_cities:,}")
print(f"Row ID range: {offset} to {offset + n_cities - 1}")
print(f"Requested pairs: {n_tot:,} (train: {n_train:,}, val: {n_val:,}, test: {n_test:,})")

# Generate random pairs without replacement (only upper triangle, excluding diagonal)
print(f"\nGenerating random inter-Atlantis pairs (seed={seed})...")
np.random.seed(seed)

# Calculate total unique pairs possible
n_unique_pairs = n_cities * (n_cities - 1) // 2
print(f"Total unique unordered pairs: {n_unique_pairs:,}")

if n_tot > n_unique_pairs:
    print(f"Warning: Requested {n_tot} pairs but only {n_unique_pairs} unique pairs exist")
    print(f"Reducing total to {n_unique_pairs}")
    # Adjust splits proportionally
    ratio_train = n_train / n_tot
    ratio_val = n_val / n_tot
    n_tot = n_unique_pairs
    n_train = int(n_tot * ratio_train)
    n_val = int(n_tot * ratio_val)
    n_test = n_tot - n_train - n_val
    print(f"New splits - train: {n_train:,}, val: {n_val:,}, test: {n_test:,}")

triu_indices = np.triu_indices(n_cities, k=1)  # k=1 excludes diagonal

# Sample from the unique pairs
selected_pair_indices = np.random.choice(n_unique_pairs, size=n_tot, replace=False)
indices = (triu_indices[0][selected_pair_indices], triu_indices[1][selected_pair_indices])

# Split into train, val, and test
train_i = indices[0][:n_train]
train_j = indices[1][:n_train]
val_i = indices[0][n_train:n_train+n_val]
val_j = indices[1][n_train:n_train+n_val]
test_i = indices[0][n_train+n_val:]
test_j = indices[1][n_train+n_val:]

# Function to create dataset from indices
def create_dataset_dict(indices_i, indices_j, df):
    """Create a dictionary suitable for HuggingFace Dataset using vectorized operations"""
    
    # Get coordinates in degrees for all city pairs at once
    lon1 = df.iloc[indices_i]['longitude'].values
    lat1 = df.iloc[indices_i]['latitude'].values
    lon2 = df.iloc[indices_j]['longitude'].values
    lat2 = df.iloc[indices_j]['latitude'].values
    
    # Use vectorized haversine from utils (handles degrees -> radians internally)
    distances_km = haversine(lon1, lat1, lon2, lat2)
    
    # Round to nearest km
    distances_km = np.round(distances_km).astype(int)
    
    # Get city IDs (using the offset row_ids)
    city1_ids = df.iloc[indices_i]['row_id'].values.astype(int)
    city2_ids = df.iloc[indices_j]['row_id'].values.astype(int)
    
    # Create text lists with progress bar
    text_list = []
    prompt_list = []
    completion_list = []
    
    for c1, c2, d in tqdm(zip(city1_ids, city2_ids, distances_km), 
                          total=len(city1_ids), 
                          desc="Formatting samples", 
                          leave=False):
        text_list.append(f"dist(c_{c1},c_{c2})={d}")
        prompt_list.append(f"<bos>dist(c_{c1},c_{c2})=")
        completion_list.append(f"{d}<eos>")
    
    return {
        'text': text_list,
        'prompt': prompt_list,
        'completion': completion_list
    }

# Create train, val, and test datasets
print("\nCreating train dataset...")
train_data = create_dataset_dict(train_i, train_j, df_atlantis)
train_dataset = Dataset.from_dict(train_data)

print("Creating validation dataset...")
val_data = create_dataset_dict(val_i, val_j, df_atlantis)
val_dataset = Dataset.from_dict(val_data)

print("Creating test dataset...")
test_data = create_dataset_dict(test_i, test_j, df_atlantis)
test_dataset = Dataset.from_dict(test_data)

# Combine into DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Save in HuggingFace format directly to output_dir
print(f"\nSaving HF dataset to {output_dir}...")
dataset_dict.save_to_disk(output_dir)

print(f"\nDataset created successfully!")
print(f"HuggingFace dataset: {output_dir}")
print(f"Train size: {len(train_dataset):,}")
print(f"Val size: {len(val_dataset):,}")
print(f"Test size: {len(test_dataset):,}")

# Print dataset info
print("\nDataset structure:")
print(dataset_dict)
print("\nFeatures:")
print(train_dataset.features)

# Show sample rows with explanation
print("\nSample train rows (inter-Atlantis pairs only):")
for i in range(min(5, len(train_dataset))):
    sample = train_dataset[i]
    print(f"  {sample['text']}")

print("\nSample val rows:")
for i in range(min(3, len(val_dataset))):
    sample = val_dataset[i]
    print(f"  {sample['text']}")

print("\nSample test rows:")
for i in range(min(3, len(test_dataset))):
    sample = test_dataset[i]
    print(f"  {sample['text']}")

print("\nTo load this dataset:")
print(">>> from datasets import load_from_disk")
print(f">>> dataset = load_from_disk('{output_dir}')")
print(">>> train_data = dataset['train']")
print(">>> val_data = dataset['validation']")
print(">>> test_data = dataset['test']")

print("\nFormat: dist(c_X,c_Y)=Z")
print(f"  - c_X, c_Y: Atlantis city IDs ({offset} to {offset+n_cities-1})")
print("  - Z: haversine distance in km (rounded to nearest integer)")
print("  - All pairs are inter-Atlantis (both cities from Atlantis)")